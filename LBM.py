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
ITERATIONS = 201
SNAP_INTERVAL = 1
SNAPSHOTS = (ITERATIONS - 1)//SNAP_INTERVAL + 1

# LBM parameters
c = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
              (-1, 1), (-1, -1), (1, -1)])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

viscosity = 0.01
TAU = 3*viscosity + 0.5
DELTA_T = 1
DELTA_X = 1
WIDTH = 100
HEIGHT = 100
Q = 9
cssq = (1/3) * (DELTA_X / DELTA_T)**2

wall = np.zeros((WIDTH, HEIGHT), bool)                     # Set to True wherever there's a wall
# wall[WIDTH//3:(2 * WIDTH//3), HEIGHT//3:(2*HEIGHT//3)] = True

# Set up cylinder
for y in range(0, HEIGHT) :
    for x in range(0, WIDTH) :
        if np.sqrt((x-WIDTH/4)**2 + (y-HEIGHT/2)**2) < 10.0: 
            wall[x,y] = True

class LBM:
    def __init__(self):
        self.rho_snapshots = np.zeros((SNAPSHOTS, WIDTH, HEIGHT))
        self.ux_snapshots = np.copy(self.rho_snapshots)
        self.uy_snapshots = np.copy(self.rho_snapshots)

        self.rho = np.ones((WIDTH, HEIGHT))
        self.rho += 0.05 * np.random.randn(WIDTH, HEIGHT)
        self.ux = np.full((WIDTH, HEIGHT), 0.1)
        self.uy = np.zeros((WIDTH, HEIGHT)) 

        self.f = LBM.get_equilibrium(self.rho, self.ux, self.uy)

    """Initialise and run the simulation.
    """
    def run(self):
        for i in range(ITERATIONS):
            self.lbm_iteration()

            if i % SNAP_INTERVAL == 0 or i == ITERATIONS - 1:
                self.rho_snapshots[i//SNAP_INTERVAL] = self.rho
                self.ux_snapshots[i//SNAP_INTERVAL] = self.ux
                self.uy_snapshots[i//SNAP_INTERVAL] = self.uy

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

        udotc = np.zeros([WIDTH, HEIGHT, Q])
        for i in range(Q):
            udotc[:,:,i] = ux * c[i,0] + uy * c[i,1]

        f_eq = np.zeros((WIDTH, HEIGHT, 9), dtype=float)

        for i in range(Q):
            f_eq[:,:,i] = w[i] * rho * (1 + udotc[:,:,i] / cssq +
                (udotc[:,:,i])**2 / (2 * cssq**2) - udotu / (2 * cssq))

        return f_eq

    """Perform an iteration of the Lattice-Boltzmann method.
    """
    def lbm_iteration(self):
        # moment update
        self.rho, self.ux, self.uy = LBM.moment_update(self.f)

        # equilibrium
        f_eq = LBM.get_equilibrium(self.rho, self.ux, self.uy)

        # Check stability condition
        assert np.min(f_eq) >= 0, "Simulation violated stability condition"

        # collision
        self.f = self.f * (1 - (DELTA_T / TAU)) + (DELTA_T / TAU) * f_eq

        # streaming
        for i in range(Q):
            self.f[:, :, i] = np.roll(self.f[:, :, i], c[i], axis=(1, 0))

        # bounce back
        boundary_f = self.f[wall, :]
        boundary_f = boundary_f[:,[0,3,4,1,2,7,8,5,6]]
        self.f[wall,:] = boundary_f


"""Render the values collected by the model with matplotlib.
"""
def render_lbm_model(model, particle_locations, save=False):
    particles = particle_locations.shape[1]

    fig, ax = plt.subplots()
    # mag_plot = plt.imshow(np.sqrt(model.ux_snapshots[0]**2 + 
    #                               model.uy_snapshots[0]**2),
    #                       extent=(0, WIDTH, 0, HEIGHT),
    #                       norm=plt.Normalize(-0.2, 0.2),
    #                       cmap=plt.get_cmap("jet"))
    mag_plot = plt.imshow(model.rho_snapshots[0],
                          extent=(0, WIDTH, 0, HEIGHT),
                          vmin=np.min(model.rho_snapshots),
                          vmax=np.max(model.rho_snapshots),
                        #   norm=plt.Normalize(-0.2, 0.2),
                          cmap=plt.get_cmap("jet"))

    particle_plots = [plt.plot(particle_locations[0,i,0] + 1/2,
                               particle_locations[0,i,1] + 1/2,
                               'ro', markersize=10)[0]
                      for i in range(particles)]

    def animate(i):
        ax.set_title(i)
        # mag_plot.set_data(np.sqrt(model.ux_snapshots[i//SNAP_INTERVAL]**2 + 
        #                           model.uy_snapshots[i//SNAP_INTERVAL]**2))
        
        mag_plot.set_data(model.rho_snapshots[i//SNAP_INTERVAL])

        for j in range(particles):
            particle_plots[j].set_data(particle_locations[i,j,0] + 1/2,
                                       particle_locations[i,j,1] + 1/2)

    anim = FuncAnimation(fig, animate, interval=1, frames=ITERATIONS,
                         repeat=False)

    plt.show()

    if save:
        anim.save(time.strftime("%Y%m%d-%H%M%S.gif"))


def track_particles(model):
    """
    Tracks the motions of particles through the airflow.
    """
    gridsize = 6

    particle_locations = np.zeros((ITERATIONS, gridsize**2, 2))
    xs = ys = np.linspace((1/(gridsize + 1)) * WIDTH, 
                          (1 - 1/(gridsize + 1)) * WIDTH, gridsize)
    particle_locations[0] = np.dstack(np.meshgrid(xs, ys)).reshape(-1, 2)

    ux_func = RegularGridInterpolator((np.arange(0, ITERATIONS, SNAP_INTERVAL),
                                       np.arange(WIDTH), np.arange(HEIGHT)),
                                      model.ux_snapshots)

    uy_func = RegularGridInterpolator((np.arange(0, ITERATIONS, SNAP_INTERVAL),
                                       np.arange(WIDTH), np.arange(HEIGHT)),
                                      model.uy_snapshots)

    for i in range(ITERATIONS - 1):
        # Add the linearly interpolated velocity vector to the location of the
        # point.
        for j in range(gridsize**2):
            x, y = particle_locations[i,j]
            dx, dy = ux_func([i, x, y])[0], uy_func([i, x, y])[0]

            # Keep particles inside boundaries
            new_x = min(max(0, x + dx), WIDTH - 1)
            new_y = min(max(0, y + dy), HEIGHT - 1)
            particle_locations[i+1, j] = [new_x, new_y]

    return particle_locations

if __name__ == '__main__':
    model = LBM()

    model.run()

    particle_locations = track_particles(model)

    render_lbm_model(model, particle_locations)
