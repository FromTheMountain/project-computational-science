import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# The initial distribution function takes values from a normal distribution.
# Large values for NORMAL_DIST_MEAN (>= 0.5, more or less) cause the
# distribution function to explode; this behaviour does not show up with
# smaller values.
NORMAL_DIST_STDDEV = 0.3

# Model
ITERATIONS = 500

# LBM parameters
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

viscosity = 0.01
TAU = 3*viscosity + 0.5
DELTA_T = 1
DELTA_X = 1
NUM_LOOPS = 3
LATTICE_WIDTH = 100
LATTICE_HEIGHT = 100
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

        for _ in range(ITERATIONS):
            f, rho, ux, uy = LBM.lbm_iteration(f)

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
def render_lbm_model(model, save=False):
    fig, ax = plt.subplots()
    img = plt.imshow(model.rho_snapshots[0], extent=(0, LATTICE_WIDTH, 0, LATTICE_HEIGHT),
                    cmap=plt.get_cmap("Greys"))

    def animate(i):
        ax.set_title(f"i={i}")
        img.set_data(model.rho_snapshots[i])

    anim = FuncAnimation(fig, animate, interval=10, frames=len(model.rho_snapshots),
                            repeat=False)

    plt.show()

    if save:
        anim.save(time.strftime("%Y%m%d-%H%M%S.gif"))


if __name__ == '__main__':
    model = LBM()

    model.run()

    render_lbm_model(model)
