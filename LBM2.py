from cProfile import label
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator
from matplotlib import colors

import matplotlib.patches as mpatches

# The initial distribution function takes values from a normal distribution.
# Large values for NORMAL_DIST_MEAN (>= 0.5, more or less) cause the
# distribution function to explode; this behaviour does not show up with
# smaller values.
NORMAL_DIST_STDDEV = 0.3

# Model
ITERATIONS = 400
SNAP_INTERVAL = 1
SNAPSHOTS = (ITERATIONS - 1)//SNAP_INTERVAL + 1

# LBM parameters
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

viscosity = 0.2
TAU = 3*viscosity + 0.5
DELTA_T = 1
DELTA_X = 1
Q = 9
cssq = (1/3) * (DELTA_X / DELTA_T)**2

# INT for diff. cell in map
WALL = 1
INLET = 2
OUTLET = 3
PERSON = 4 


def read_map_from_file(filename):
    with open(filename, 'r') as f:
        iterator = enumerate(f)

        _, firstline = next(iterator)
        width, height = [int(x) for x in firstline.strip().split(',')]

        wall = np.zeros((width, height), bool)
        inlet = np.zeros((width, height), bool)
        outlet = np.zeros((width, height), bool)
        person = np.zeros((width, height), bool)

        for i, line in iterator:
            for j, c in enumerate(line.strip()):
                c = int(c)
                if c == WALL:
                    wall[j, width-i] = True
                elif c == INLET:
                    inlet[j, width-i] = True
                elif c == OUTLET:
                    outlet[j, width-i] = True
                elif c == PERSON:
                    person[j, width-i] = True

    return person, wall, inlet, outlet


class LBM:
    def __init__(self, person, wall, inlet, outlet):
        # Get the map details
        assert wall.shape == inlet.shape == outlet.shape
        self.width, self.height = wall.shape

        self.wall = wall
        self.inlet = inlet
        self.outlet = outlet
        self.person = person

        # Set the initial macroscopic quantities
        self.rho = np.ones((self.width, self.height))
        # self.rho += 0.05 * np.random.randn(WIDTH, HEIGHT)
        self.ux = np.full((self.width, self.height), 0.0)

        # Lid driven cavity
        # self.ux[:, -2:] = 0.3

        self.uy = np.zeros((self.width, self.height))

        self.f = LBM.get_equilibrium(self.width * self.height,
                                     self.rho.flatten(), self.ux.flatten(),
                                     self.uy.flatten()).reshape(
            (self.width, self.height, Q))

        self.rho_snapshots = np.zeros((SNAPSHOTS, self.width, self.height))
        self.ux_snapshots = np.copy(self.rho_snapshots)
        self.uy_snapshots = np.copy(self.rho_snapshots)

    def run(self):
        """
        Initialise and run the simulation.
        """
        for i in range(ITERATIONS):
            self.lbm_iteration(i)

            if i % SNAP_INTERVAL == 0 or i == ITERATIONS - 1:
                self.rho_snapshots[i//SNAP_INTERVAL] = self.rho
                self.ux_snapshots[i//SNAP_INTERVAL] = self.ux
                self.uy_snapshots[i//SNAP_INTERVAL] = self.uy

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
        self.f = self.f * (1 - (DELTA_T / TAU)) + (DELTA_T / TAU) * f_eq

        # streaming
        for i in range(Q):
            self.f[:, :, i] = np.roll(self.f[:, :, i], c[i], axis=(0, 1))

        # bounce back
        boundary_f = self.f[self.wall, :]
        boundary_f = boundary_f[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
        self.f[self.wall, :] = boundary_f

        # Set the velocity vector at inlets
        inlet_ux = 0.5
        inlet_uy = 0.0
        inlet_rho = self.rho[self.inlet]

        self.f[self.inlet] = LBM.get_equilibrium(len(inlet_rho),
                                                 self.rho[self.inlet],
                                                 inlet_ux, inlet_uy)

        # Set the density at outlets
        outlet_rho = 0.5
        outlet_ux = self.ux[self.outlet]
        outlet_uy = self.uy[self.outlet]
        self.f[self.outlet] = LBM.get_equilibrium(len(outlet_ux), outlet_rho,
                                                  outlet_ux, outlet_uy)

    def render(self, particle_locations=None, kind="density",
               vectors=False, save=False):
        """
        Render the values collected by the model with matplotlib. Argument
        "kind" should be of value "density" or "mag"
        """
    
        fig, ax = plt.subplots()

        if particle_locations is not None:

            particles = particle_locations.shape[1]

            particle_plots = [plt.plot(particle_locations[0, i, 0] + 1/2,
                                    particle_locations[0, i, 1] + 1/2,
                                    'ro', markersize=10)[0]
                            for i in range(particles)]


        init_vals = np.sqrt(model.ux_snapshots[0]**2 +
                            model.uy_snapshots[0]**2) if kind == "mag" \
            else model.rho_snapshots[0]
        vmin = 0 if kind == "mag" else 0.8
        vmax = 0.2 if kind == "mag" else 1.2
        fluid_plot = plt.imshow(init_vals.T, origin="lower", vmin=vmin,
                                vmax=vmax, cmap=plt.get_cmap("jet"))
        plt.colorbar(fluid_plot)

        people_data = (WALL * self.wall + INLET * self.inlet + OUTLET * self.outlet + PERSON * self.person)

        labels, labelscolors = ["WALL","INLET", "OUTLET","PERSON"], ["lightgreen", "blue", "red", "purple", "yellow"]
        
        cmap = colors.ListedColormap(
            labelscolors
            )

        people_plot = plt.imshow(people_data.T, origin="lower", alpha=0.6, cmap=cmap)
        
        red_patch = mpatches.Patch(color=labelscolors[1], label='Wall')
        blue_patch = mpatches.Patch(color=labelscolors[2], label='Inlet')
        purple_patch = mpatches.Patch(color=labelscolors[3], label='Outlet')
        yellow_patch = mpatches.Patch(color=labelscolors[4], label='Person')

        ax.legend(handles=[red_patch, blue_patch, purple_patch, yellow_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=len(labels))
        fig.tight_layout()


        if vectors:
            x, y = np.meshgrid(np.linspace(0, self.width-1, 20, dtype=int),
                               np.linspace(0, self.height-1, 20, dtype=int))
            u = model.ux_snapshots[0, x, y]
            v = model.uy_snapshots[0, x, y]

            # Set scale to 0.5 for lid driven cavity, 4 for Karman vortex
            vector_plot = plt.quiver(x, y, u, v, scale=4)

       
        def animate(i):
            ax.set_title("{}, iteration {}".format(kind, i))

            if particle_locations is not None:
                for j in range(particles):
                    particle_plots[j].set_data(
                        particle_locations[i, j, 0] + 1/2,
                        particle_locations[i, j, 1] + 1/2)

          
            vals = np.sqrt(model.ux_snapshots[i//SNAP_INTERVAL]**2 +
                           model.uy_snapshots[i//SNAP_INTERVAL]**2) \
                if kind == "mag" else model.rho_snapshots[i//SNAP_INTERVAL]
            fluid_plot.set_data(vals.T)

            people_data = (WALL * self.wall + INLET * self.inlet + OUTLET * self.outlet + PERSON * self.person).T
            people_plot.set_data(people_data)

     
            if vectors:
                u = model.ux_snapshots[i//SNAP_INTERVAL, x, y]
                v = model.uy_snapshots[i//SNAP_INTERVAL, x, y]

                vector_plot.set_UVC(u, v)

            
        anim = FuncAnimation(fig, animate, interval=1, frames=ITERATIONS,
                             repeat=True)

        plt.show()

        if save:
            anim.save(time.strftime("%Y%m%d-%H%M%S.gif"))

    def track_particles(self):
        """
        Tracks the motions of particles through the airflow.
        """
        num_particles = 10

        # Spawn num_particles particles at evenly spaced intervals.
        particle_locations = np.zeros((ITERATIONS, num_particles, 2))

        infections = np.zeros((ITERATIONS))

        ux_func = RegularGridInterpolator(
            (np.arange(0, ITERATIONS, SNAP_INTERVAL), np.arange(self.width),
             np.arange(self.height)), model.ux_snapshots)

        uy_func = RegularGridInterpolator(
            (np.arange(0, ITERATIONS, SNAP_INTERVAL), np.arange(self.width),
             np.arange(self.height)), model.uy_snapshots)

        infection_counter = 0  

        particles_exited = []
        for i in range(ITERATIONS - 1):
            
            if i % (ITERATIONS // num_particles) == 0:
                # Spawn a new particle
                # Randomly choose an inlet cell.
                inlet_indices = np.where(model.inlet)
                idx = np.random.randint(len(inlet_indices[0]))

                particle_locations[i, i // (ITERATIONS // num_particles)] = \
                    inlet_indices[0][idx], inlet_indices[1][idx]

            # Add the linearly interpolated velocity vector to the location of
            # the point.


            for j in range(i // (ITERATIONS // num_particles) + 1):
                x, y = particle_locations[i, j]

                # check whether particle intercepted a person

                if j not in particles_exited:
                    if model.person[int(x)][int(y)]:
                        infection_counter += 1 
                        infections[i] += 1 
                        particle_locations[i+1, j] = [self.width-1, self.height-1]
                        particles_exited.append(int(j))

                    else:
                        dx, dy = ux_func([i, x, y])[0], uy_func([i, x, y])[0]

                        # Keep particles inside boundaries
                        new_x = min(max(0, x + dx), self.width - 1)
                        new_y = min(max(0, y + dy), self.height - 1)
                        particle_locations[i+1, j] = [new_x, new_y]


        print("Infection counter", infection_counter)
        print(particles_exited)
        infection_rate = np.cumsum(infections)

        """"
        nd array of infection of time/it.
        
        """
        
        return infection_rate, particle_locations


if __name__ == '__main__':
    # To change from lid driven cavity to Karman vortex, only two changes need
    # to be made. First, the filename below needs to be modified to
    # './maps/karmanvortex'. Second, the scale parameter in line 201 needs to
    # be adjusted to 4.
    person, wall, inlet, outlet = read_map_from_file('./maps/person_in_room')

    model = LBM(person, wall, inlet, outlet)
    model.run()

    infection_rate, particle_locations = model.track_particles()

    plt.plot(infection_rate)

    model.render(kind="mag", particle_locations=particle_locations,
                 vectors=True)
