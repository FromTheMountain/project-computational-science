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
    rho = np.sum(f, axis=2)
    u = np.einsum('xyk,kc->cxy', f, c) / rho
    return rho, u


def get_equilibrium(rho, u):
    udotc = np.einsum('axy,ia->xyi', u, c)
    udotu = np.linalg.norm(u, axis=0)**2

    # f_eq = np.einsum('i,xy->xyi', w, rho) * (1 + cdot3u*(1 + 0.5*cdot3u) - 1.5*usq[:,:,np.newaxis])

    f_eq = np.full([LATTICE_WIDTH, LATTICE_HEIGHT, NUM_VELOCITIES], 0.0)
    # for i in range(NUM_VELOCITIES):
    #     f_eq[:, :, i] = w[i] * rho * (1 + udotc[:, :, i] / cssq
    #         + udotc[:, :, i] / (2 * cssq**2) - udotu / (2 * cssq))

    ux = u[0,:,:]
    uy = u[1,:,:]

    f_eq[:,:,0] = 2 * rho / 9 * (2 - 3 * udotu)
    f_eq[:,:,1] = rho / 18 * (2 + 6 * ux + 9 * ux**2 - 3 * udotu)
    f_eq[:,:,2] = rho / 18 * (2 + 6 * uy + 9 * uy**2 - 3 * udotu)
    f_eq[:,:,3] = rho / 18 * (2 - 6 * ux + 9 * ux**2 - 3 * udotu)
    f_eq[:,:,4] = rho / 18 * (2 - 6 * uy + 9 * uy**2 - 3 * udotu)
    f_eq[:,:,5] = rho / 36 * (1 + 3 * (ux + uy) + 9 * ux * uy + 3 * udotu)
    f_eq[:,:,6] = rho / 36 * (1 - 3 * (ux - uy) - 9 * ux * uy + 3 * udotu)
    f_eq[:,:,7] = rho / 36 * (1 - 3 * (ux + uy) + 9 * ux * uy + 3 * udotu)
    f_eq[:,:,8] = rho / 36 * (1 + 3 * (ux - uy) - 9 * ux * uy + 3 * udotu)

    # TODO: remove this, it's just here so we can inspect it to see if they
    # eq_rho matches rho and eq_rho_u matches rho_u (which does seem to be the
    # case).
    eq_rho = np.sum(f_eq, axis=2)
    rho_u = np.einsum('xyk,kc->xyc', f, c)
    eq_rho_u = np.einsum('xyk,kc->xyc', f_eq, c)

    print(np.sum(rho))

    return f_eq
    # for x in range(LATTICE_WIDTH):
    #     for y in range(LATTICE_HEIGHT):
    #         for i in range(NUM_VELOCITIES):
    #             f_eq[x, y, i] = w[i] * rho[x, y] * (
    #                 1 + np.dot(u[x, y, :], c[i])
    #                 + np.dot(u[x, y, :], c[i]**2) / (2 * c_s**4)
    #                 - np.dot(u[x, y, :], u[x, y, :]) / (2 * c_s**2)
    #             )

    return f_eq


def collision(f, f_eq):
    f = f * (1 - 1 / tau) + 1 / tau * f_eq
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
    print(frame)
    f, rho, _ = lbm_iteration(f)
    img.set_data(rho)
    return img,


anim = FuncAnimation(fig, update, frames=500, interval=1,
                     init_func=init, repeat=False, blit=True)
plt.show()
