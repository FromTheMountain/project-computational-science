import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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
LATTICE_WIDTH = 200
LATTICE_HEIGHT = 200
Q = 9
cssq = (1/3) * (DELTA_X / DELTA_T)**2


class LBM:
    def __init__(self):
        self.snapshots = []

    """Initialise and run the simulation.
    """
    def run(self):
        f = np.random.random([LATTICE_WIDTH, LATTICE_HEIGHT, Q])

        for _ in range(ITERATIONS):
            f, rho, u = LBM.lbm_iteration(f)

            self.snapshots.append(rho)

    """Update the rho and velocity values of each grid position.
    """
    def moment_update(f):
        rho = np.sum(f, 2)
        u_rho = np.repeat(np.repeat(np.sum(c, 0)[np.newaxis,:], LATTICE_WIDTH, 0)[np.newaxis,:], LATTICE_HEIGHT, 0)
        u = u_rho / np.repeat(rho[:,:,np.newaxis], 2, 2)

        return rho, u

    """Calculate the equalibrium values of the grid.
    """
    def get_equilibrium(rho, u):
        f_eq = np.zeros((LATTICE_WIDTH, LATTICE_HEIGHT, 9), dtype=float)
        udotu = np.repeat(np.einsum("xyk,xyk->xy", u, u)[:,:,np.newaxis], 9, 2)
        udotc = np.einsum("xyk, vk -> xyv", u, c)
        w_rho = np.einsum("k, xy -> xyk", w, rho)

        f_eq = w_rho * (1 + (udotc / cssq) + (udotc**2 / (2 * cssq**2)) - (udotu / (2 * cssq)))

        return f_eq

    """Perform an iteration of the Lattice-Boltzmann method.
    """
    def lbm_iteration(f):
        # moment update
        rho, u = LBM.moment_update(f)

        # # equilibrium
        f_eq = LBM.get_equilibrium(rho, u)

        # # collision
        f = f * (1 - (DELTA_T / TAU)) + (DELTA_T / TAU) * f_eq

        # streaming
        for i in range(Q):
            f[:, :, i] = np.roll(f[:, :, i], c[i], axis=(1, 0))

        return f, rho, u


"""Render the values collected by the model with matplotlib.
"""
def render_lbm_model(model, save=False):
    fig, ax = plt.subplots()
    img = plt.imshow(model.snapshots[0], extent=(0, LATTICE_WIDTH, 0, LATTICE_HEIGHT),
                    cmap=plt.get_cmap("Greys"))

    def animate(i):
        ax.set_title(i)
        img.set_data(model.snapshots[i])

    anim = FuncAnimation(fig, animate, interval=1, frames=len(model.snapshots) - 1,
                            repeat=False)

    plt.show()

    if save:
        anim.save(time.strftime("%Y%m%d-%H%M%S.gif"))


if __name__ == '__main__':
    model = LBM()

    model.run()

    render_lbm_model(model, True)
