import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator

# The initial distribution function takes values from a normal distribution.
# Large values for NORMAL_DIST_MEAN (>= 0.5, more or less) cause the
# distribution function to explode; this behaviour does not show up with
# smaller values.
NORMAL_DIST_STDDEV = 0.3

# Model
ITERATIONS = 3201
SNAP_INTERVAL = 1
SNAPSHOTS = (ITERATIONS - 1)//SNAP_INTERVAL + 1

# LBM parameters
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

viscosity = 0.02
TAU = 3*viscosity + 0.5
DELTA_T = 1
DELTA_X = 1
Q = 9
cssq = (1/3) * (DELTA_X / DELTA_T)**2

WALL = 1
INLET = 2
OUTLET = 3


def read_map_from_file(filename):
    with open(filename, 'r') as f:
        iterator = enumerate(f)

        _, firstline = next(iterator)
        width, height = [int(x) for x in firstline.strip().split(',')]

        wall = np.zeros((width, height), bool)
        inlet = np.zeros((width, height), bool)
        outlet = np.zeros((width, height), bool)

        for i, line in iterator:
            for j, c in enumerate(line.strip()):
                c = int(c)
                if c == WALL:
                    wall[j, width-i] = True
                elif c == INLET:
                    inlet[j, width-i] = True
                elif c == OUTLET:
                    outlet[j, width-i] = True

    return wall, inlet, outlet


class LBM:
    def __init__(self, wall, inlet, outlet):
        # Get the map details
        assert wall.shape == inlet.shape == outlet.shape
        self.width, self.height = wall.shape

        self.wall = wall
        self.inlet = inlet
        self.outlet = outlet

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
        self.f[wall, :] = boundary_f

        # Set the velocity vector at inlets
        inlet_ux = 0.05
        inlet_uy = 0.0
        inlet_rho = self.rho[self.inlet]

        self.f[self.inlet] = LBM.get_equilibrium(len(inlet_rho),
                                                 self.rho[self.inlet],
                                                 inlet_ux, inlet_uy)

        # Set the density at outlets
        outlet_rho = 0.9
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
        if particle_locations is not None:
            particles = particle_locations.shape[1]

        fig, ax = plt.subplots()

        init_vals = np.sqrt(model.ux_snapshots[0]**2 +
                            model.uy_snapshots[0]**2) if kind == "mag" \
            else model.rho_snapshots[0]
        vmin = 0 if kind == "mag" else 0.8
        vmax = 0.2 if kind == "mag" else 1.2
        fluid_plot = plt.imshow(init_vals.T, origin="lower", vmin=vmin,
                                vmax=vmax, cmap=plt.get_cmap("jet"))
        plt.colorbar(fluid_plot)

        if vectors:
            x, y = np.meshgrid(np.linspace(0, self.width-1, 20, dtype=int),
                               np.linspace(0, self.height-1, 20, dtype=int))
            u = model.ux_snapshots[0, x, y]
            v = model.uy_snapshots[0, x, y]

            # Set scale to 0.5 for lid driven cavity, 4 for Karman vortex
            vector_plot = plt.quiver(x, y, u, v, scale=0.5)

        if particle_locations is not None:
            particle_plots = [plt.plot(particle_locations[0, i, 0] + 1/2,
                                       particle_locations[0, i, 1] + 1/2,
                                       'ro', markersize=10)[0]
                              for i in range(particles)]

        def animate(i):
            ax.set_title("{}, iteration {}".format(kind, i))

            vals = np.sqrt(model.ux_snapshots[i//SNAP_INTERVAL]**2 +
                           model.uy_snapshots[i//SNAP_INTERVAL]**2) \
                if kind == "mag" else model.rho_snapshots[i//SNAP_INTERVAL]

            fluid_plot.set_data(vals.T)

            if vectors:
                u = model.ux_snapshots[i//SNAP_INTERVAL, x, y]
                v = model.uy_snapshots[i//SNAP_INTERVAL, x, y]

                vector_plot.set_UVC(u, v)

            if particle_locations is not None:
                for j in range(particles):
                    particle_plots[j].set_data(
                        particle_locations[i, j, 0] + 1/2,
                        particle_locations[i, j, 1] + 1/2)

        anim = FuncAnimation(fig, animate, interval=1, frames=ITERATIONS,
                             repeat=True)

        plt.show()

        if save:
            anim.save(time.strftime("%Y%m%d-%H%M%S.gif"))

    def track_particles(self):
        """
        Tracks the motions of particles through the airflow.
        """
        num_particles = 20
        mass = 0.1
        drag_coefficient = 0.47 # coefficient of a sphere

        # Spawn num_particles particles at evenly spaced intervals.
        particle_locations = np.zeros((ITERATIONS, num_particles, 2))
        particle_velocities = np.zeros((ITERATIONS, num_particles, 2))

        ux_func = RegularGridInterpolator(
            (np.arange(0, ITERATIONS, SNAP_INTERVAL), np.arange(self.width),
             np.arange(self.height)), model.ux_snapshots)

        uy_func = RegularGridInterpolator(
            (np.arange(0, ITERATIONS, SNAP_INTERVAL), np.arange(self.width),
             np.arange(self.height)), model.uy_snapshots)

        for i in range(ITERATIONS - 1):
            if i % (ITERATIONS // num_particles) == 0:
                # Spawn a new particle
                # Randomly choose an inlet cell.
                inlet_indices = np.where(model.inlet)
                idx = np.random.randint(len(inlet_indices[0]))

                particle_locations[i, i // (ITERATIONS // num_particles)] = \
                    inlet_indices[0][idx], inlet_indices[1][idx]

                particle_velocities[i, i // (ITERATIONS // num_particles)] = \
                    [0.0, 0.0]

            # Add the linearly interpolated velocity vector to the location of
            # the point.
            for j in range(i // (ITERATIONS // num_particles) + 1):
                x, y = particle_locations[i, j]
                vx, vy = particle_velocities[i, j] # Particle velocity
                ux, uy = ux_func([i, x, y])[0], uy_func([i, x, y])[0] # Air velocity

                # Velocity of particle relative to fluid
                rel_vx = ux - vx
                rel_vy = uy - vy

                # Drag force
                density = model.rho_snapshots[i//SNAP_INTERVAL, int(x), int(y)]
                f_x = 0.5 * density * rel_vx**2 * drag_coefficient
                f_y = 0.5 * density * rel_vy**2 * drag_coefficient

                # Acceleration
                a_x = f_x / mass
                a_y = f_y / mass

                # Compute new velocity
                new_vx = vx + a_x * DELTA_T
                new_vy = vy + a_y * DELTA_T

                if i < 150 and j == 0:
                    print(vx)
                    print(f_x)
                    print(rel_vx)
                    print(new_vx)
                    print()

                particle_velocities[i + 1, j] = [new_vx, new_vy]

                dx = new_vx * DELTA_T
                dy = new_vy * DELTA_T

                # Keep particles inside boundaries
                new_x = min(max(0, x + dx), self.width - 1)
                new_y = min(max(0, y + dy), self.height - 1)
                particle_locations[i + 1, j] = [new_x, new_y]

        return particle_locations


if __name__ == '__main__':
    # To change from lid driven cavity to Karman vortex, only two changes need
    # to be made. First, the filename below needs to be modified to
    # './maps/karmanvortex'. Second, the scale parameter in line 201 needs to
    # be adjusted to 4.
    # wall, inlet, outlet = read_map_from_file('./maps/liddrivencavity')
    wall, inlet, outlet = read_map_from_file('./maps/concept1')

    model = LBM(wall, inlet, outlet)
    model.run()

    particle_locations = model.track_particles()

    model.render(kind="mag", particle_locations=particle_locations)
