import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator
from matplotlib import colors
import matplotlib.patches as mpatches
import cv2

# LBM constants
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
Q = 9

AIR, WALL, INLET, OUTLET, INFECTED, SUSCEPTIBLE = [0, 1, 2, 3, 4, 5]

# for know hardcoden
susceptible_centroids = np.array([(18, 93), (79, 80), (79, 51), (80, 26),
                                  (49, 5)])
NUM_SUSCEP_CENTROIDS = len(susceptible_centroids)


class LBM:
    def __init__(self, params, inlet_handler=None, outlet_handler=None):
        # Get the map details
        self.width = self.height = params['size']
        self.map_scaling_factor = 1.0
        self.wall, self.inlet, self.outlet, self.infected, self.susceptible = \
            self.read_map_from_file('./maps/' + params['map'])
        self.iters = params['iterations']

        assert self.wall.shape == self.inlet.shape == self.outlet.shape

        # Whether or not particles need to be simulated.
        self.simulate_particles = params['simulate_particles']

        # every 20 it spawn particles
        if self.simulate_particles:
            self.spawn_rate = 20            # every x iterations
            self.spawn_amount_at_rate = 5   # x particles

            self.num_particles = (self.iters // self.spawn_rate) * \
                self.spawn_amount_at_rate
            self.particle_nr = 0

        self.inlet_handler = inlet_handler if inlet_handler is not None else \
            LBM.inlet_handler
        self.outlet_handler = outlet_handler if outlet_handler is not None \
            else LBM.outlet_handler

        # If we do know the Reynolds number, then use different calculations
        # to determine the other values.
        if 'reynolds' in params:
            self.init_reynolds(params)
        else:
            self.init_default(params)

        print("=" * 10 + " Model values " + "=" * 10)
        print(f"{'Physical length':<15} {self.width * self.dx:>10.4f} meters")
        print(f"{'Total time':<15} {self.iters * self.dt:>10.4f} seconds")
        print()
        print(f"{'dx':<10} {self.dx:>10.4f} m/unit")
        print(f"{'dt':<10} {self.dt:>10.4f} s/step")
        print(f"{'tau':<10} {self.tau:>10.4f}")
        print(f"{'Re':<10} {self.Re:>10.4f}")
        print(f"{'nu_lb':<10} {self.nu_lb:>10.4f} units^2/step")
        print(f"{'u_lb':<10} {self.u_lb:>10.4f} units/step")
        print()

        # Set the initial macroscopic quantities
        self.rho = np.ones((self.width, self.height))
        self.ux = np.full((self.width, self.height), 0.0)
        self.uy = np.zeros((self.width, self.height))

        self.f = self.get_equilibrium(self.width * self.height,
                                      self.rho.flatten(), self.ux.flatten(),
                                      self.uy.flatten()).reshape(
            (self.width, self.height, Q))

        """Initialise the model based on a couple of physical values.
        """
    def init_default(self, params):
        # Known parameters (SI units)
        self.L_p = params['L_p']
        self.nu_p = params['nu_p']
        self.u_p = params['u_p']
        self.dt = params['dt']

        # Compute other variables
        self.dx = self.L_p / self.width
        self.Re = (self.u_p * self.L_p) / self.nu_p
        self.u_lb = self.u_p * (self.dt / self.dx)
        self.L_lb = self.L_p / self.dx
        self.nu_lb = (self.u_lb * self.L_lb) / self.Re

        self.cssq = 1/3
        self.tau = self.nu_lb / self.cssq + 0.5

        """Initialise the model based on the known Reynolds number and other
        values.
        """
    def init_reynolds(self, params):
        # Known parameters (SI units)
        self.Re = params['reynolds']
        self.L_lb = params['L_lb']
        self.L_p = params['L_p']
        self.nu_p = params['nu_p']
        self.u_lb = params['u_lb']

        # Compute other variables
        self.u_p = (self.Re * self.nu_p) / self.L_p
        self.dx = self.L_p / self.L_lb
        self.nu_lb = (self.u_lb * self.L_lb) / self.Re
        self.dt = self.Re * self.nu_lb / self.L_lb**2

        self.cssq = 1/3
        self.tau = self.nu_lb / self.cssq + 0.5

        """Read a map made by the map editor from a file.
        """
    def read_map_from_file(self, filename):
        with open(filename, 'r') as f:
            iterator = enumerate(f)
            _, firstline = next(iterator)
            width, height = [int(x) for x in firstline.strip().split(',')]
            assert width == height, "Map width does not match map height"
            self.map_scaling_factor = self.width / width

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

        # Resize all the arrays
        wall = cv2.resize(wall.astype('uint8'), (self.width, self.height),
                          cv2.INTER_NEAREST).astype(bool)
        inlet = cv2.resize(inlet.astype('uint8'), (self.width, self.height),
                           cv2.INTER_NEAREST).astype(bool)
        outlet = cv2.resize(outlet.astype('uint8'), (self.width, self.height),
                            cv2.INTER_NEAREST).astype(bool)
        infected = cv2.resize(infected.astype('uint8'),
                              (self.width, self.height),
                              cv2.INTER_NEAREST).astype(bool)
        susceptible = cv2.resize(susceptible.astype('uint8'),
                                 (self.width, self.height),
                                 cv2.INTER_NEAREST).astype(bool)

        return wall, inlet, outlet, infected, susceptible

        """Update the macroscopic quantities of the model and return them.
        """
    def moment_update(f):
        rho = np.sum(f, 2)

        ux = np.sum(c[:, 0] * f, axis=2) / rho
        uy = np.sum(c[:, 1] * f, axis=2) / rho

        return rho, ux, uy

        """Calculate the equilibrium values of the model and return them.
        """
    def get_equilibrium(self, n, rho, ux, uy):
        """
        Calculate the equalibrium distribution for the BGK operator.
        """
        udotu = ux * ux + uy * uy

        udotc = np.zeros((n, Q), dtype=float)
        for i in range(Q):
            udotc[:, i] = ux * c[i, 0] + uy * c[i, 1]

        f_eq = np.zeros((n, 9), dtype=float)

        for i in range(Q):
            f_eq[:, i] = w[i] * rho * (1 + udotc[:, i] / self.cssq +
                                       (udotc[:, i])**2 / (2 * self.cssq**2) -
                                       udotu / (2 * self.cssq))

        return f_eq

        """Perform an iteration of the Lattice-Boltzmann method. Also checks
        whether the model is stable.

        Performs inlet and outlet handling according to the specified handlers.
        """
    def lbm_iteration(self, it):
        # moment update
        self.rho, self.ux, self.uy = LBM.moment_update(self.f)

        # equilibrium
        f_eq = self.get_equilibrium(self.width * self.height,
                                    self.rho.flatten(), self.ux.flatten(),
                                    self.uy.flatten()).reshape(
                                       (self.width, self.height, Q))

        # Check stability condition
        assert np.min(f_eq) >= 0, f"Simulation violated stability \
            condition at {np.unravel_index(np.argmin(f_eq), f_eq.shape)}"

        # collision
        self.f = self.f * (1 - (1 / self.tau)) + (1 / self.tau) * f_eq

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
        self.inlet_handler(self, it)
        self.outlet_handler(self, it)

        """Calculate the values of the inlets based on the specified velocity.
        """
    def inlet_handler(model, it):
        """
        The default inlet handler for an LBM model.
        """
        # Set the velocity vector at inlets
        inlet_ux = model.u_lb
        inlet_uy = 0.0
        inlet_rho = np.ones_like(model.rho[model.inlet], dtype=float)

        model.f[model.inlet] = model.get_equilibrium(len(inlet_rho),
                                                     inlet_rho,
                                                     inlet_ux, inlet_uy)

    """Keep the values of the outlets constant, so no bounce back occurs
       and the fluid exits the computational domain.
    """
    def outlet_handler(model, it):
        """
        The default outlet handler for an LBM model.
        """
        # Set the density at outlets
        outlet_rho = model.rho[model.outlet]
        outlet_ux = model.ux[model.outlet]
        outlet_uy = model.uy[model.outlet]
        model.f[model.outlet] = model.get_equilibrium(len(outlet_ux),
                                                      outlet_rho,
                                                      outlet_ux, outlet_uy)

        """Render the model.

        Vectors: Whether to draw vector arrows on the visualisation.
        kind: What to visualise, options are: mag, density
        """
    def render(self, kind="density", vectors=False, save_file=None):
        """
        Render the values collected by the model with matplotlib. Argument
        "kind" should be of value "density" or "mag"
        """
        # Initialize plots
        fig, ax = plt.subplots()

        # First layer: fluid plot
        self.fluid_plot = plt.imshow(np.zeros((self.width, self.height),
                                              dtype=float),
                                     vmin=0.0, vmax=self.u_lb,
                                     cmap=plt.get_cmap("jet"))
        cbar = plt.colorbar(self.fluid_plot)
        cbar.set_label("Speed (units/steps)", rotation=270, labelpad=15)

        # adding numbers at susceptible_centroids
        for idx, val in enumerate(susceptible_centroids):
            x, y = val
            plt.text(x, y, str(idx), fontsize=10, color='white')

        # Second layer: vector plot
        if vectors:
            x, y = np.meshgrid(np.linspace(0, self.width-1, 20, dtype=int),
                               np.linspace(0, self.height-1, 20, dtype=int))
            u = self.ux[x, y]
            v = self.uy[x, y]

            # Set scale to 0.5 for lid driven cavity, 4 for Karman vortex
            self.vector_plot = plt.quiver(x, y, u, v, scale=4)

        # Third layer: particle plots
        if self.simulate_particles:
            self.particle_locations = np.zeros((self.num_particles, 2), float)
            self.particle_plots = [plt.plot(0, 0, 'ro', markersize=2)[0]
                                   for i in range(self.num_particles)]

        # Fourth layer: map plot
        map_data = (WALL * self.wall + INLET * self.inlet +
                    OUTLET * self.outlet + INFECTED * self.infected +
                    SUSCEPTIBLE * self.susceptible)

        clr = ["lightgreen", "blue", "red", "purple", "yellow", "darkgreen"]
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
        if self.simulate_particles:
            self.infections = np.zeros((NUM_SUSCEP_CENTROIDS, self.iters))
            self.removed = np.zeros((self.iters))
            self.particles_exited = set()

        anim = FuncAnimation(fig, self.animate, interval=1,
                             frames=range(1, self.iters),
                             repeat=True, fargs=[ax, kind, vectors],
                             init_func=lambda: self.animate(0, ax, kind,
                                                            vectors))

        if save_file:
            anim.save("simulation.html", writer="html")
        else:
            for i in range(self.iters):
                self.animate(i, ax, kind, vectors)

        if self.simulate_particles:
            fig, ax = plt.subplots()

            infection_rate = np.cumsum(self.infections, axis=1)
            removed_rate = np.cumsum(self.removed)

            ax.plot(infection_rate.T)
            # TODO: Add spawn rate to title and parameters
            ax.set_title(f"Infection rate in a building")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Infected count")
            ax.legend(range(len(susceptible_centroids)))
            fig.savefig('infection_rate.png')
            ax.plot(removed_rate)
            fig.savefig('removed_rate.png')

        """The animate function that is called for every step of the model.
        """
    def animate(self, it, ax, kind, vectors):
        print("Running animate on iteration {} of {} of kind {}".format(it + 1,
              self.iters, kind),
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
        if self.simulate_particles:
            self.update_particles(it)

            for i, loc in enumerate(self.particle_locations):
                self.particle_plots[i].set_data(*loc)

        # Update the plot title
        ax.set_title("{}, i={}, t={:.4f}s".format(kind, it, it * self.dt))

        """Update the particles that move according to the velocities of the
        fluid.
        """
    def update_particles(self, it):
        """
        Tracks the motions of particles through the airflow.
        """

        if it % self.spawn_rate == 0:
            for _ in range(self.spawn_amount_at_rate):
                # Spawn a new particle
                # Randomly choose an infected cell.
                if self.particle_nr < self.num_particles:
                    infected_indices = np.where(self.infected)

                    # If there are no infected grid cells, then stop.
                    if len(infected_indices[0]) == 0:
                        return

                    idx = np.random.randint(len(infected_indices[0]))

                    self.particle_locations[self.particle_nr] = \
                        infected_indices[0][idx], infected_indices[1][idx]
                    self.particle_nr += 1

        # it > 0
        ux_func = RegularGridInterpolator((np.arange(self.width),
                                           np.arange(self.height)),
                                          self.ux)

        uy_func = RegularGridInterpolator((np.arange(self.width),
                                           np.arange(self.height)),
                                          self.uy)

        # Add the linearly interpolated velocity vector to the location of
        # the point.
        for i in range(self.particle_nr):
            x, y = self.particle_locations[i]

            if i in self.particles_exited:
                continue

            # check whether particle intercepted a person
            if self.susceptible[int(round(x)), int(round(y))]:
                # FIND CLOSEST NODE
                node = int(x), int(y)
                nodes = susceptible_centroids
                dist_2 = np.sum((nodes - node)**2, axis=1)
                closest = np.argmin(dist_2)

                self.infections[closest][it] += 1

                self.particles_exited.add(i)
                self.particle_locations[i] = [0, 0]
            elif self.outlet[int(round(x)), int(round(y))]:
                self.removed[i] += 1
                self.particles_exited.add(i)
                self.particle_locations[i] = [0, 0]
            else:
                dx, dy = ux_func([x, y])[0], uy_func([x, y])[0]

                dx, dy = self.map_scaling_factor * dx, \
                    self.map_scaling_factor * dy
                # Keep particles inside boundaries
                new_x = min(max(0, x + dx), self.width - 1)
                new_y = min(max(0, y + dy), self.height - 1)

                self.particle_locations[i] = [new_x, new_y]


if __name__ == '__main__':
    model_params = {
        "iterations": 10000,
        "size": 100,
        "simulate_particles": True,
        "map": "concept4",
        "reynolds": 100,
        "L_lb": 200,
        "L_p": 1,
        "nu_p": 1.48e-5,
        "u_lb": 0.4
    }

    model = LBM(model_params)

    model.render(kind="mag", vectors=True, save_file='animation')
