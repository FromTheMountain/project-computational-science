import sys
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from LBM import LBM


def lid_driven_cavity():
    model_params = {
        "iterations": 10000,
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
    model.render(kind="mag", show_realtime=True)


def validation():
    """
    check line strip [50, 0-100] and [0-100, 50] in LDC test to compare

    """
    model_params = {
        "iterations": 10000,
        "size": 100,
        "simulate_particles": False,
        "map": "liddrivencavity",
        "reynolds": 100.0,
        "L_lb": 100,
        "L_p": 1,
        "nu_p": 1.48e-5,
        "u_lb": 0.2
    }

    model = LBM(model_params)
    model.render(kind="mag")

    nx = model_params['size']
    ny = model_params['size']

    vx_error = np.zeros((nx))
    uy_error = np.zeros((ny))
    half_nx = math.floor(nx / 2)
    half_ny = math.floor(ny / 2)

    for i in range(nx):
        vx_error[i] = model.ux[half_ny, i]/model.u_lb

    for j in range(ny):
        uy_error[j] = model.uy[j, half_nx]/model.u_lb

    output_dir = 'validation/'

    # Write to files
    filename = output_dir+'cavity_vx'
    with open(filename, 'w') as f:
        for i in range(nx):
            f.write('{} {}\n'.format(i*model.dx, uy_error[i]))

    filename = output_dir+'cavity_uy'
    with open(filename, 'w') as f:
        for j in range(ny):
            f.write('{} {}\n'.format(j*model.dx, vx_error[j]))

    # plot against reference
    all_files = ['cavity_vx', 'cavity_vx_ref', 'cavity_uy', 'cavity_uy_ref']

    plt.figure()

    all_y = []
    for filename in all_files:
        data = pd.read_csv('validation/' + str(filename), sep=' ',
                           header=None)

        x = data[0]
        y = data[1]
        all_y.append(y)
        plt.plot(x, y, label=filename)

    plt.title(f"U/V Profile vs Ghia et al. (ref), Re = {model.Re}")

    MSE_vx = round(np.square(all_y[0] - all_y[1]).mean(), 5)
    MSE_uy = round(np.square(all_y[2] - all_y[3]).mean(), 5)

    all_files = [f'cavity_vx (MSE={MSE_vx})', 'cavity_vx_ref',
                 f'cavity_uy (MSE={MSE_uy})', 'cavity_uy_ref']

    plt.legend(all_files)
    plt.savefig('validation/comparison.png')
    plt.show()


def karman_vortex():
    model_params = {
        "iterations": 10000,
        "size": 100,
        "simulate_particles": False,
        "map": "karmanvortex",
        "reynolds": 1000.0,
        "L_lb": 100,
        "L_p": 1,
        "nu_p": 1.48e-5,
        "u_lb": 0.1
    }

    model = LBM(model_params)
    model.render(kind="mag", show_realtime=True)


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


def experiment2():
    model_params = {
        "iterations": 10000,
        "size": 100,
        "simulate_particles": False,
        "map": "concept4",
        "reynolds": 10.0,
        "L_lb": 100,
        "L_p": 1,
        "nu_p": 1.48e-5,
        "u_lb": 0.05
    }

    model = LBM(model_params)

    model.render(kind="mag", vectors=True, show_realtime=True)

    period_length = 800
    open_window_frac = 1

    def inlet_handler(model, it):
        if it % period_length < open_window_frac * period_length:
            # The windows are open, the inlet is acting like an actual inlet.
            inlet_ux = model.u_lb
            inlet_uy = 0.0
            inlet_rho = np.ones_like(model.rho[model.inlet], dtype=float)

            model.f[model.inlet] = model.get_equilibrium(len(inlet_rho),
                                                         inlet_rho,
                                                         inlet_ux, inlet_uy)
        else:
            # The windows are closed, the inlet is acting like a wall.
            model.ux[model.inlet] = 0
            model.uy[model.inlet] = 0

            inlet_f = model.f[model.inlet, :]
            inlet_f = inlet_f[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
            model.f[model.inlet, :] = inlet_f

    def outlet_handler(model, it):
        if it % period_length < open_window_frac * period_length:
            # The windows are open, the outlet is acting like an actual outlet.
            # Set the density at outlets
            outlet_rho = 0.9
            outlet_ux = model.ux[model.outlet]
            outlet_uy = model.uy[model.outlet]
            model.f[model.outlet] = model.get_equilibrium(len(outlet_ux),
                                                          outlet_rho,
                                                          outlet_ux, outlet_uy)
        else:
            model.ux[model.outlet] = 0
            model.uy[model.outlet] = 0

            outlet_f = model.f[model.outlet, :]
            outlet_f = outlet_f[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
            model.f[model.outlet, :] = outlet_f


if __name__ == '__main__':
    experiment_options = {
        "cavity": lid_driven_cavity,
        "karman": karman_vortex,
        "validation": validation,
        "1": experiment1,
        "2": experiment2
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
