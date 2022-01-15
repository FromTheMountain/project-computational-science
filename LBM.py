import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_LOOPS = 3
LATTICE_WIDTH = 200
LATTICE_HEIGHT = 200
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Some alternative velocity sets for testing:

# c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)])
# w = np.array([4/9, 1/9, 1/9, 1/9, 1/9])

# c = np.array([(0, 0), (1, 0)])
# w = np.array([4/9, 1/9])

NUM_VELOCITIES = len(c)
cssq = 1/3

# delta_t was chosen to be 1
# omega was taken as defined in 03_karman_vortex.py

viscosity = 0.02
tau = 3*viscosity + 0.5

def moment_update(f):
    rho = np.sum(f, 2)
    u_rho = np.repeat(np.repeat(np.sum(c, 0)[np.newaxis,:], LATTICE_WIDTH, 0)[np.newaxis,:], LATTICE_HEIGHT, 0)
    u = u_rho / np.repeat(rho[:,:,np.newaxis], 2, 2)

    return rho, u


def get_equilibrium(rho, u):
    f_eq = np.zeros((LATTICE_WIDTH, LATTICE_HEIGHT, 9), dtype=float)
    udotu = np.repeat(np.einsum("xyk,xyk->xy", u, u)[:,:,np.newaxis], 9, 2) # CORRECT
    udotc = np.einsum("xyk, vk -> xyv", u, c) # CORRECT
    w_rho = np.einsum("k, xy -> xyk", w, rho) # CORRECT

    f_eq = w_rho * (1 + (udotc / cssq) + (udotc**2 / (2 * cssq**2)) - (udotu / (2 * cssq)))

    return f_eq


def collision(f, f_eq):
    f = f * (1 - (1 / tau)) + (1 / tau) * f_eq

    return f

def streaming(f):
    for i in range(NUM_VELOCITIES):
        f[:, :, i] = np.roll(f[:, :, i], c[i], axis=(1, 0))

    return f


def lbm_iteration(f):
    # moment update
    rho, u = moment_update(f)

    # # equilibrium
    f_eq = get_equilibrium(rho, u)

    # # collision
    f = collision(f, f_eq)

    # streaming
    f = streaming(f)

    return f, rho, u


# initialisation
# f = np.full([LATTICE_WIDTH, LATTICE_HEIGHT, NUM_VELOCITIES], 1.0)
f = np.random.random([LATTICE_WIDTH, LATTICE_HEIGHT, NUM_VELOCITIES])

rho, u = moment_update(f)

fig, ax = plt.subplots()
img = plt.imshow(rho, extent=(0, LATTICE_WIDTH, 0, LATTICE_HEIGHT),
                 cmap=plt.get_cmap("Greys"))


def init():
    ax.set_xlim(0, LATTICE_WIDTH)
    ax.set_ylim(0, LATTICE_HEIGHT)
    return img,


def update(frame):
    global f

    f, rho, _ = lbm_iteration(f)
    img.set_data(rho)
    return img,


anim = FuncAnimation(fig, update, frames=500, interval=1,
                     init_func=init, repeat=False, blit=True)
plt.show()
