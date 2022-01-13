import numpy as np

NUM_LOOPS = 2
LATTICE_WIDTH = 10
LATTICE_HEIGHT = 10
VELOCITY_VECS = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), 
                          (-1, 1), (-1, -1), (1, -1)])
VELOCITY_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
NUM_VELOCITIES = len(VELOCITY_VECS)

def moment_update(f):
    rho = np.sum(f, axis=2)
    u = np.einsum('xyk,kc->xyc', f, VELOCITY_VECS)
    return rho, u

# initialisation
f = np.full([LATTICE_WIDTH, LATTICE_HEIGHT, NUM_VELOCITIES], 1.0)

# simulation loop
for i in range(NUM_LOOPS):
    # moment update
    rho, u = moment_update(f)
    # equilibrium
    # collision
    # streaming
