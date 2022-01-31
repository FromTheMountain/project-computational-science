import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator
from matplotlib import colors
import matplotlib.patches as mpatches
import cv2

# Model
ITERATIONS = 2000
SNAP_INTERVAL = 1
SNAPSHOTS = (ITERATIONS - 1)//SNAP_INTERVAL + 1

# LBM parameters
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

viscosity = 0.2  # kinematic lattice viscosity
TAU = 3*viscosity + 0.5  # section 7.2.1.1 in LBPP
DELTA_T = 1
DELTA_X = 1
Q = 9
cssq = (1/3) * (DELTA_X / DELTA_T)**2

AIR, WALL, INLET, OUTLET, INFECTED, SUSCEPTIBLE = [0, 1, 2, 3, 4, 5]

class LBM:
    def __init__(self, wall, inlet, outlet, infected, susceptible, size,
                 num_particles=0, inlet_handler=None, outlet_handler=None):
        # Get the map details
        assert wall.shape == inlet.shape == outlet.shape
        self.width = self.height = size

        # Resize all the arrays
        self.wall = cv2.resize(wall.astype('uint8'), (size, size), cv2.INTER_NEAREST).astype(bool)
        self.inlet = cv2.resize(inlet.astype('uint8'), (size, size), cv2.INTER_NEAREST).astype(bool)
        self.outlet = cv2.resize(outlet.astype('uint8'), (size, size), cv2.INTER_NEAREST).astype(bool)
        self.infected = cv2.resize(infected.astype('uint8'), (size, size), cv2.INTER_NEAREST).astype(bool)
        self.susceptible = cv2.resize(susceptible.astype('uint8'), (size, size), cv2.INTER_NEAREST).astype(bool)

        self.num_particles = num_particles

        self.inlet_handler = inlet_handler if inlet_handler is not None else \
            LBM.inlet_handler
        self.outlet_handler = outlet_handler if outlet_handler is not None \
            else LBM.outlet_handler

        if inlet_handler == None:
            self.inlet_handler = LBM.inlet_handler

        # Set the initial macroscopic quantities
        self.rho = np.ones((self.width, self.height))
        # self.rho += 0.05 * np.random.randn(WIDTH, HEIGHT)
        self.ux = np.full((self.width, self.height), 0.0)

        self.uy = np.zeros((self.width, self.height))

        self.f = LBM.get_equilibrium(self.width * self.height,
                                     self.rho.flatten(), self.ux.flatten(),
                                     self.uy.flatten()).reshape(
            (self.width, self.height, Q))

        # Know variables
        self.w = 10                 # meter
        self.delta_x = 0.01        # meter
        self.w_star = self.w / self.delta_x     # needed grid width 500
        self.C_l = self.delta_x

        # self.air_visc = 1.225   # kg/m3
        # self.C_rho = self.air_visc

        self.C_L = self.delta_x
        rho_0_start = 1

        # between 0.5 - 2
        self.tau = self.tau_star = 0.51

        self.kin_visc_air = 1.48e-5
        self.cs = (1/3)
        self.delta_t = self.cs * (self.tau_star - 0.5) * (self.delta_x**2 / self.kin_visc_air)
        self.C_t = self.delta_t
        self.dt = 1

        print("delta", self.dt)

        self.C_u = self.C_l  / self.C_t
        print("C u ",self.C_u)

        self.airflow_u = 0.1                      # airflow m/s in building 5-8 m/s
        self.u_lb = self.airflow_u / self.C_u
        print(self.u_lb)

        # Model parameters
        self.compute_lbm_parameters()


    ### Compute remaining lbm parameters
    def compute_lbm_parameters(self):

        print("---------Model para----------")
        self.u_star = self.airflow_u * self.w**2 / 8 * self.kin_visc_air
        self.Re = self.u_star * self.w / self.kin_visc_air
        print(self.u_star)
        print(self.Re)

        # spwan_rate per iteration
        self.spawn_rate = int(self.num_particles / ITERATIONS)

    def read_map_from_file(filename):
        with open(filename, 'r') as f:
            iterator = enumerate(f)

            _, firstline = next(iterator)
            width, height = [int(x) for x in firstline.strip().split(',')]

            wall = np.zeros((width, height), bool)
            inlet = np.zeros((width, height), bool)
            outlet = np.zeros((width, height), bool)
            infected = np.zeros((width, height), bool)
            susceptible = np.zeros((width, height), bool)

            for i, line in iterator:
                for j, c in enumerate(line.strip()):
                    c = int(c)
                    if c == WALL:
                        wall[j, width-i] = True
                    elif c == INLET:
                        inlet[j, width-i] = True
                    elif c == OUTLET:
                        outlet[j, width-i] = True
                    elif c == INFECTED:
                        infected[j, width-i] = True
                    elif c == SUSCEPTIBLE:
                        susceptible[j, width-i] = True

        return wall, inlet, outlet, infected, susceptible

    def moment_update(f):
        """
        Update the rho and velocity values of each grid position.
        """
        rho = np.sum(f, 2)

        ux = np.sum(c[:, 0] * f, axis=2) / rho
        uy = np.sum(c[:, 1] * f, axis=2) / rho

        return rho, ux, uy

    def get_equilibrium(n, rho, ux, uy):
        """
        Calculate the equalibrium distribution for the BGK operator.
        """
        udotu = ux * ux + uy * uy

        udotc = np.zeros((n, Q), dtype=float)
        for i in range(Q):
            udotc[:, i] = ux * c[i, 0] + uy * c[i, 1]

        f_eq = np.zeros((n, 9), dtype=float)

        for i in range(Q):
            f_eq[:, i] = w[i] * rho * (1 + udotc[:, i] / cssq +
                                       (udotc[:, i])**2 / (2 * cssq**2) -
                                       udotu / (2 * cssq))

        return f_eq

    def lbm_iteration(self, it):
        """
        Perform an iteration of the Lattice-Boltzmann method.
        """
        # moment update
        self.rho, self.ux, self.uy = LBM.moment_update(self.f)

        # equilibrium
        f_eq = LBM.get_equilibrium(self.width * self.height,
                                   self.rho.flatten(), self.ux.flatten(),
                                   self.uy.flatten()).reshape(
                                       (self.width, self.height, Q))

        # Check stability condition
        assert np.min(f_eq) >= 0, "Simulation violated stability condition"

        # collision
        self.f = self.f * (1 - (self.dt / self.tau_star)) + (self.dt / self.tau_star) * f_eq

        # streaming
        for i in range(Q):
            self.f[:, :, i] = np.roll(self.f[:, :, i], c[i], axis=(0, 1))

        # bounce back
        self.ux[self.wall] = 0
        self.uy[self.wall] = 0

        boundary_f = self.f[self.wall, :]
        boundary_f = boundary_f[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
        self.f[self.wall, :] = boundary_f

        # Handle inlets and outlets. Note that "self.inlet_handler" does not
        # necessarily refer to LBM.inlet_handler, it could also be a custom
        # callback function provided by the user when initialising the model.
        self.inlet_handler(self)
        self.outlet_handler(self)

    def inlet_handler(model):
        """
        The default inlet handler for an LBM model.
        """
        # Set the velocity vector at inlets
        inlet_ux = model.u_lb
        inlet_uy = 0.0
        inlet_rho = model.rho[model.inlet]

        model.f[model.inlet] = LBM.get_equilibrium(len(inlet_rho),
                                                 model.rho[model.inlet],
                                                 inlet_ux, inlet_uy)

    def outlet_handler(model):
        """
        The default outlet handler for an LBM model.
        """
        # Set the density at outlets
        outlet_rho = 0.9
        outlet_ux = model.ux[model.outlet]
        outlet_uy = model.uy[model.outlet]
        model.f[model.outlet] = LBM.get_equilibrium(len(outlet_ux), outlet_rho,
                                                    outlet_ux, outlet_uy)

    def render(self, kind="density", vectors=False, save_file=None):
        """
        Render the values collected by the model with matplotlib. Argument
        "kind" should be of value "density" or "mag"
        """
        # Initialize plots
        fig, ax = plt.subplots()

        # First layer: fluid plot
        vmin = 0 if kind == "mag" else 0.8
        vmax = 0.2 if kind == "mag" else 1.2

        self.fluid_plot = plt.imshow(np.zeros((self.width, self.height),
                                              dtype=float),
                                     vmin=vmin, vmax=vmax,
                                     cmap=plt.get_cmap("jet"))
        plt.colorbar(self.fluid_plot)

        # Second layer: vector plot
        if vectors:
            x, y = np.meshgrid(np.linspace(0, self.width-1, 20, dtype=int),
                               np.linspace(0, self.height-1, 20, dtype=int))
            u = self.ux[x, y]
            v = self.uy[x, y]

            # Set scale to 0.5 for lid driven cavity, 4 for Karman vortex
            self.vector_plot = plt.quiver(x, y, u, v, scale=4)

        # Third layer: particle plots
        if self.num_particles:
            self.particle_locations = np.zeros((self.num_particles, 2), float)
            self.particle_plots = [plt.plot(0, 0, 'ro', markersize=10)[0]
                                   for i in range(self.num_particles)]

        # Fourth layer: map plot
        map_data = (WALL * self.wall + INLET * self.inlet +
                    OUTLET * self.outlet + INFECTED * self.infected +
                    SUSCEPTIBLE * self.susceptible)

        clr = ["lightgreen", "blue", "red", "purple", "yellow", "lightcoral"]
        cmap = colors.ListedColormap(clr)

        self.map_plot = plt.imshow(map_data.T, alpha=0.6, origin="lower",
                                   cmap=cmap)

        patches = [mpatches.Patch(color=c, label=name) for c, name in
                   zip(clr[1:],
                       ['Wall', 'Inlet', 'Outlet', 'Infected', 'Susceptible'])]

        ax.legend(handles=patches, loc='upper center',
                  bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True,
                  ncol=len(clr))
        fig.tight_layout()

        # Initialise particles
        self.infections = np.zeros((ITERATIONS))
        self.removed = np.zeros((ITERATIONS))
        self.particles_exited = set()

        anim = FuncAnimation(fig, self.animate, interval=1, frames=ITERATIONS,
                             repeat=True, fargs=[ax, kind, vectors])

        if save_file:
            anim.save(save_file)
            # anim.save("simulation/1/" + str(idx) + ".png", writer="imagemagick")
            fig.savefig('last_frame.png')
        else:
            for i in range(ITERATIONS):
                self.animate(i, ax, kind, vectors)

        infection_rate = np.cumsum(self.infections)
        removed_rate = np.cumsum(self.removed)

        return infection_rate, removed_rate

    def animate(self, it, ax, kind, vectors):
        print("Running animate on iteration {} of kind {}".format(it, kind),
              end="\r")
        # Perform an LBM iteration and update fluid plot
        self.lbm_iteration(it)

        vals = np.sqrt(self.ux**2 + self.uy**2) if kind == "mag" else self.rho
        self.fluid_plot.set_data(vals.T)

        # Update the vector plot
        if vectors:
            x, y = np.meshgrid(np.linspace(0, self.width-1, 20, dtype=int),
                               np.linspace(0, self.height-1, 20, dtype=int))
            u, v = self.ux[x, y], self.uy[x, y]

            self.vector_plot.set_UVC(u, v)

        # Update particle locations and plots
        self.update_particles(it)

        for i, loc in enumerate(self.particle_locations):
            self.particle_plots[i].set_data(*loc)

        # Update the plot title
        ax.set_title("{}, iteration {}".format(kind, it))

    def update_particles(self, it):
        """
        Tracks the motions of particles through the airflow.
        """
        if it == 0:
            # initialize the particles
            for i in range(self.num_particles):
                # Spawn a new particle
                # Randomly choose an infected cell.
                infected_indices = np.where(self.infected)
                idx = np.random.randint(len(infected_indices[0]))

                self.particle_locations[i] = \
                    infected_indices[0][idx], infected_indices[1][idx]

            return

        # it > 0
        ux_func = RegularGridInterpolator((np.arange(self.width),
                                           np.arange(self.height)),
                                          self.ux)

        uy_func = RegularGridInterpolator((np.arange(self.width),
                                           np.arange(self.height)),
                                          self.uy)

        # Add the linearly interpolated velocity vector to the location of
        # the point.
        for i in range(self.num_particles):
            x, y = self.particle_locations[i]
            # check whether particle intercepted a person

            if i in self.particles_exited:
                continue

            if self.susceptible[int(round(x)), int(round(y))]:
                self.infections[i] += 1
                self.particles_exited.add(i)
                self.particle_locations[i] = [0, 0]
            elif self.outlet[int(round(x)), int(round(y))]:
                self.removed[i] += 1
                self.particles_exited.add(i)
                self.particle_locations[i] = [0, 0]
            else:
                dx, dy = ux_func([x, y])[0], uy_func([x, y])[0]

                # Keep particles inside boundaries
                new_x = min(max(0, x + dx), self.width - 1)
                new_y = min(max(0, y + dy), self.height - 1)

                self.particle_locations[i] = [new_x, new_y]


if __name__ == '__main__':
    wall, inlet, outlet, infected, susceptible = \
        LBM.read_map_from_file('./maps/concept1')

    model = LBM(wall, inlet, outlet, infected, susceptible, 100, num_particles=10)

    infection_rate, removed_rate = \
        model.render(kind="mag", vectors=True, save_file='animation.gif')

    fig, ax = plt.subplots()

    ax.plot(infection_rate)
    ax.plot(removed_rate)
