import sys
import os
import matplotlib.pyplot as plt

import LBM


def lid_driven_cavity():
    pass


def karman_vortex():
    pass


def experiment1():
    """
    This experiment aims to anwer the question: what is the influence of the
    velocity that particles get at an inlet, on the number of particles that
    end up infecting susceptible people?
    """
    def inlet_handler(model, inlet_ux):
        inlet_rho = model.rho[model.inlet]

        model.f[model.inlet] = LBM.LBM.get_equilibrium(
            len(inlet_rho), model.rho[model.inlet], inlet_ux, 0.0)

    # First simulation: vary the inlet velocity from 0 to 0.5
    for i, inlet_ux in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        print("Iteration {}, inlet_ux {}".format(i, inlet_ux))

        wall, inlet, outlet, infected, susceptible = \
            LBM.LBM.read_map_from_file('maps/concept1')

        model = LBM.LBM(wall, inlet, outlet, infected, susceptible,
                        num_particles=100,
                        inlet_handler=lambda m: inlet_handler(m, inlet_ux))

        infection_rate, removed_rate = \
            model.render(kind="mag", vectors=True, save_file=None)

        _, ax = plt.subplots()

        ax.plot(infection_rate, label="Particle infections")
        ax.plot(removed_rate, label="Removed particles")
        ax.legend()
        plt.savefig('results/exp1-{}.png'.format(i))

if len(sys.argv) <= 1:
    print("Please provide experiment name.")

name = sys.argv[1]

if not os.path.exists('results'):
    os.mkdir('results')

if name == "1":
    experiment1()
elif name == "ldc":
    lid_driven_cavity()
elif name == "karman":
    karman_vortex()
else:
    raise ValueError("Experiment name not recognized.")
