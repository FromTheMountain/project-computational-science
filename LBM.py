import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp2d

# The initial distribution function takes values from a normal distribution.
# Large values for NORMAL_DIST_MEAN (>= 0.5, more or less) cause the
# distribution function to explode; this behaviour does not show up with
# smaller values.
NORMAL_DIST_STDDEV = 0.3

# Model
ITERATIONS = 200
SNAPSHOT_INTERVAL = 5

# LBM parameters
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

viscosity = 0.01
TAU = 3*viscosity + 0.5
DELTA_T = 1
DELTA_X = 1
NUM_LOOPS = 3
LATTICE_WIDTH = 200
LATTICE_HEIGHT = 200
Q = 9
cssq = (1/3) * (DELTA_X / DELTA_T)**2


class LBM:
    def __init__(self):
        self.rho_snapshots = []
        self.ux_snapshots = []
        self.uy_snapshots = []

    """Initialise and run the simulation.
    """
    def run(self):
        f = np.ones((LATTICE_WIDTH,LATTICE_HEIGHT,Q))
        f += NORMAL_DIST_STDDEV * np.random.randn(LATTICE_WIDTH,LATTICE_HEIGHT,Q)

        for i in range(ITERATIONS):
            f, rho, ux, uy = LBM.lbm_iteration(f)

            if i % SNAPSHOT_INTERVAL == 0:
                self.rho_snapshots.append(rho)
                self.ux_snapshots.append(ux)
                self.uy_snapshots.append(uy)

    """Update the rho and velocity values of each grid position.
    """
    def moment_update(f):
        rho = np.sum(f, 2)

        ux = np.sum(c[:,0] * f, axis=2) / rho
        uy = np.sum(c[:,1] * f, axis=2) / rho

        return rho, ux, uy

    """Calculate the equalibrium values of the grid.
    """
    def get_equilibrium(rho, ux, uy):
        udotu = ux * ux + uy * uy

        udotc = np.zeros([LATTICE_WIDTH, LATTICE_HEIGHT, Q])
        for i in range(Q):
            udotc[:,:,i] = ux * c[i,0] + uy * c[i,1]

        f_eq = np.zeros((LATTICE_WIDTH, LATTICE_HEIGHT, 9), dtype=float)

        for i in range(Q):
            f_eq[:,:,i] = w[i] * rho * (1 + udotc[:,:,i] / cssq +
                (udotc[:,:,i])**2 / (2 * cssq**2) - udotu / (2 * cssq))

        return f_eq

    """Perform an iteration of the Lattice-Boltzmann method.
    """
    def lbm_iteration(f):
        # moment update
        rho, ux, uy = LBM.moment_update(f)

        # equilibrium
        f_eq = LBM.get_equilibrium(rho, ux, uy)

        # Check stability condition
        assert np.min(f_eq) >= 0, "Simulation violated stability condition"

        # collision
        f = f * (1 - (DELTA_T / TAU)) + (DELTA_T / TAU) * f_eq

        # streaming
        for i in range(Q):
            f[:, :, i] = np.roll(f[:, :, i], c[i], axis=(1, 0))

        return f, rho, ux, uy


"""Render the values collected by the model with matplotlib.
"""
def render_lbm_model(model, particle_locations, save=False):
    fig, ax = plt.subplots()
    rho_plot = plt.imshow(model.rho_snapshots[0], extent=(0, LATTICE_WIDTH, 0, LATTICE_HEIGHT),
                    cmap=plt.get_cmap("Greys"))

    particle_plots = [plt.plot(particle_locations[0,i,0],
                               particle_locations[0,i,1],
                               'ro', markersize=10)[0]
                      for i in range(4)]

    def animate(i):
        ax.set_title(i)
        rho_plot.set_data(model.rho_snapshots[i//SNAPSHOT_INTERVAL])

        for j in range(4):
            particle_plots[j].set_data(particle_locations[i,j,0],
                                       particle_locations[i,j,1])

    anim = FuncAnimation(fig, animate, interval=200, frames=ITERATIONS,
                         repeat=False)

    plt.show()

    if save:
        anim.save(time.strftime("%Y%m%d-%H%M%S.gif"))


def track_particles(model):
    """
    TrackS the motions of four particles through the airflow.
    """
    num_particles = 4

    particle_locations = np.zeros((ITERATIONS, num_particles, 2))
    particle_locations[0] = np.array([
        [LATTICE_WIDTH / 3, LATTICE_HEIGHT / 3],
        [LATTICE_WIDTH / 3, 2 * LATTICE_HEIGHT / 3],
        [2 * LATTICE_WIDTH / 3, LATTICE_HEIGHT / 3],
        [2 * LATTICE_WIDTH / 3, 2 * LATTICE_HEIGHT / 3]
    ])

    for i in range(ITERATIONS - 1):
        if i % SNAPSHOT_INTERVAL == 0:
            # Get linear interpolation function for ux and uy
            ux_func = interp2d(np.arange(LATTICE_WIDTH),
                               np.arange(LATTICE_HEIGHT),
                               model.ux_snapshots[i//SNAPSHOT_INTERVAL])
            uy_func = interp2d(np.arange(LATTICE_WIDTH),
                               np.arange(LATTICE_HEIGHT),
                               model.uy_snapshots[i//SNAPSHOT_INTERVAL])

        # Add the linearly interpolated velocity vector to the location of the
        # point.
        for j in range(num_particles):
            x, y = particle_locations[i,j]
            dx, dy = ux_func(x, y)[0], uy_func(x, y)[0]
            x += dx
            y += dy

            particle_locations[i+1, j] = [x, y]

    return particle_locations

if __name__ == '__main__':
    model = LBM()

    model.run()

    particle_locations = track_particles(model)

    render_lbm_model(model, particle_locations)

