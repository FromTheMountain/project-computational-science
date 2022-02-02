import sys
import os
import matplotlib.pyplot as plt

from LBM import LBM


def lid_driven_cavity():
    model_params = {
        "iterations": 50,
        "size": 100,
        "simulate_particles": False,
        "map": "liddrivencavity",
        "reynolds": 1000.0,
        "L_lb": 100,
        "L_p": 1,
        "nu_p": 1.48e-5,
        "u_lb": 0.1
    }

    model = LBM(model_params)
    model.render(kind="mag", save_file="animation")


def karman_vortex():
    pass


def experiment1():
    """
    This experiment aims to answer the question: what is the influence of the
    velocity that particles get at an inlet, on the number of particles that
    end up infecting susceptible people?
    """
    def inlet_handler(model, inlet_ux):
        inlet_rho = model.rho[model.inlet]

        model.f[model.inlet] = LBM.LBM.get_equilibrium(
            len(inlet_rho), model.rho[model.inlet], inlet_ux, 0.0)

    # First simulation: vary the inlet velocity from 0 to 0.5
    for i, inlet_ux in enumerate([0.2]):
        print("Iteration {}, inlet_ux {}".format(i, inlet_ux))

        wall, inlet, outlet, infected, susceptible = \
            LBM.read_map_from_file('maps/concept2')

        model = LBM(wall, inlet, outlet, infected, susceptible,
                    num_particles=100,
                    inlet_handler=lambda m: inlet_handler(m, inlet_ux))

        infection_rate, removed_rate = \
            model.render(kind="mag", vectors=True,
                         save_file='results/exp1-{}.gif'.format(i))

        _, ax = plt.subplots()

        ax.plot(infection_rate, label="Particle infections")
        ax.plot(removed_rate, label="Removed particles")
        ax.legend()
        plt.savefig('results/exp1-{}.png'.format(i))


if __name__ == '__main__':
    experiment_options = {
        "cavity": lid_driven_cavity,
        "karman": karman_vortex
    }

    if len(sys.argv) < 2:
        print("Please provide an experiment name. Options are:")

        for key in experiment_options:
            print(f"- {key}")

        exit(1)

    try:
        exp_func = experiment_options[sys.argv[1].lower()]
    except KeyError:
        print("Experiment does not exist, please try again.")

        exit(1)

    if not os.path.exists('results'):
        os.mkdir('results')

    # Run the experiment
    exp_func()
