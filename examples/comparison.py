"""
This script compares the absolute energy error between the different ode_solvers at different jump heights.
The comparison is done between the pickle files in the same directory. The "output" of this script is two plots ()
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import biorbd

from somersault import Models

from typing import Callable


def RK4(t, f: Callable, y0: np.ndarray, args=()):
    """
    Runge-Kutta 4th order method

    Parameters
    ----------
    t : array_like
    time steps
    f : Callable
    function to be integrated in the form f(t, y, *args)
    y0 : np.ndarray
    initial conditions of states

    Returns
    -------
    y : array_like
    states for each time step

    """
    n = len(t)
    y = np.zeros((len(y0), n))
    y[:, 0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        yi = np.squeeze(y[:, i])
        k1 = f(t[i], yi, args)
        k2 = f(t[i] + h / 2.0, yi + k1 * h / 2.0, args)
        k3 = f(t[i] + h / 2.0, yi + k2 * h / 2.0, args)
        k4 = f(t[i] + h, yi + k3 * h, args)
        y[:, i + 1] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def fun(t, yi, qddotj):
    q = yi[:15]
    qdot = yi[15:]
    return np.concatenate((qdot, model.ForwardDynamicsFreeFloatingBase(q, qdot, qddotj).to_array(), qddotj))


def get_created_data_from_pickle(file: str, ode_solver_name: str, intermediary_states: bool = True):
    """
    Creates data from pickle

    Parameters
    ----------
    file: str
        File where data has been saved
    ode_solver_name: str
        Name of the ode_solver in the pickle file names.
    intermediary_states: bool
        If True, the data will be created from the states with intermediary nodes. If False, the data will be created
        from the states without intermediary nodes.

    Returns
    -------
    Data
    """
    with open(file, "rb") as f:
        data_tmp = pickle.load(f)

        print(f"{file}: status: {data_tmp['status']}, cost:{data_tmp['cost']}, time to optimize: "
              f"{data_tmp['real_time_to_optimize']}, "
              f"nb_it: {data_tmp['iterations']}")
        print(
            f"1ère phase : {data_tmp['time'][0][-1] - data_tmp['time'][0][0]}, "
            f"{data_tmp['states_no_intermediate'][0]['q'].shape[1]} nodes")
        print(
            f"2ère phase : {data_tmp['time'][1][-1] - data_tmp['time'][1][0]}, "
            f"{data_tmp['states_no_intermediate'][1]['q'].shape[1]} nodes")

        shape_0_1 = data_tmp["states_no_intermediate"][0]["q"].shape[1] - 1

        datas_shape = (
            data_tmp["states_no_intermediate"][0]["q"].shape[0],
            shape_0_1 + data_tmp["states_no_intermediate"][1]["q"].shape[1]
        )

        # q
        datas_q = np.zeros(datas_shape)
        datas_q[:, :shape_0_1] = data_tmp["states_no_intermediate"][0]["q"][:, :-1]
        datas_q[:, shape_0_1:] = data_tmp["states_no_intermediate"][1]["q"]

        # qdot
        datas_qdot = np.zeros(datas_shape)
        datas_qdot[:, :shape_0_1] = data_tmp["states_no_intermediate"][0]["qdot"][:, :-1]
        datas_qdot[:, shape_0_1:] = data_tmp["states_no_intermediate"][1]["qdot"]

        # Time
        datas_time = np.zeros(datas_shape[1])
        if data_tmp["states_no_intermediate"][0]["q"].shape[1] == data_tmp["states"][0]["q"].shape[1]:
            step = 1
        else:
            step = 5
        datas_time[:shape_0_1] = data_tmp["time"][0][:-1:step]
        datas_time[shape_0_1:] = data_tmp["time"][1][::step]

        # Torques
        if ode_solver_name == "ACC-DRIVEN_RK4_0327" and intermediary_states:
            nb_reint = 100
            datas_q_int = np.zeros((datas_shape[0], (datas_shape[1] - 1) * nb_reint + 1))
            datas_qdot_int = np.zeros((datas_shape[0], (datas_shape[1] - 1) * nb_reint + 1))
            datas_time_int = np.zeros((datas_shape[1] - 1) * nb_reint + 1)
            datas_qddot_joints_int = np.zeros((9, (datas_shape[1] - 1) * nb_reint + 1))

            # qddotj
            qddot_joints_shape = (
                data_tmp["controls"][0]["qddot_joints"].shape[0],
                data_tmp["controls"][0]["qddot_joints"].shape[1] - 1 + data_tmp["controls"][1]["qddot_joints"].shape[1])

            datas_qddot_joints = np.zeros((qddot_joints_shape[0], qddot_joints_shape[1]))
            datas_qddot_joints[:, :data_tmp["controls"][0]["qddot_joints"].shape[1] - 1] = \
                data_tmp["controls"][0]["qddot_joints"][:, :-1]
            datas_qddot_joints[:, data_tmp["controls"][0]["qddot_joints"].shape[1] - 1:] = \
                data_tmp["controls"][1]["qddot_joints"]

            # Intermediary states
            for i in range(datas_shape[1] - 1):
                datas_time_int[i*nb_reint:(i+1)*nb_reint + 1] = \
                    np.linspace(datas_time[i], datas_time[i+1], nb_reint + 1)
                for j in range(i * nb_reint, (i + 1) * nb_reint):
                    datas_qddot_joints_int[:, j] = datas_qddot_joints[:, i]
                q_int = RK4(
                    datas_time_int[i*nb_reint:(i+1)*nb_reint],
                    fun,
                    np.concatenate((datas_q[:, i], datas_qdot[:, i])),
                    args=(datas_qddot_joints[:, i]))
                datas_q_int[:, i*nb_reint:(i+1)*nb_reint] = q_int[:15, :nb_reint]
                datas_qdot_int[:, i*nb_reint:(i+1)*nb_reint] = q_int[15:, :nb_reint]

            datas_q_int[:, -1] = datas_q[:, -1]
            datas_qdot_int[:, -1] = datas_qdot[:, -1]
            datas_qddot_joints_int[:, -1] = datas_qddot_joints[:, -1]
            # qddotb
            datas_qddot_floating_base_int = np.zeros((6, (datas_shape[1] - 1) * nb_reint + 1))
            for i in range(datas_q_int.shape[1]):
                datas_qddot_floating_base_int[:, i] = model.ForwardDynamicsFreeFloatingBase(
                    datas_q_int[:, i], datas_qdot_int[:, i], datas_qddot_joints_int[:, i]
                ).to_array()

            datas_qddot_int = np.concatenate((datas_qddot_floating_base_int, datas_qddot_joints_int), axis=0)

            datas_tau_int = np.zeros((9, (datas_shape[1] - 1) * nb_reint + 1))

            for i in range(datas_q_int.shape[1]):
                datas_tau_int[:, i] = \
                    model.InverseDynamics(datas_q_int[:, i], datas_qdot_int[:, i], datas_qddot_int[:, i]).to_array()[6:]

            datas_q = datas_q_int
            datas_qdot = datas_qdot_int
            datas_time = datas_time_int
            datas_tau = datas_tau_int

        elif ode_solver_name == "ACC-DRIVEN_RK4_0327" and not intermediary_states:
            # qddotj
            qddot_joints_shape = (
                data_tmp["controls"][0]["qddot_joints"].shape[0],
                data_tmp["controls"][0]["qddot_joints"].shape[1] - 1 + data_tmp["controls"][1]["qddot_joints"].shape[1])

            datas_qddot_joints = np.zeros((qddot_joints_shape[0], qddot_joints_shape[1]))
            datas_qddot_joints[:, :data_tmp["controls"][0]["qddot_joints"].shape[1] - 1] = \
                data_tmp["controls"][0]["qddot_joints"][:, :-1]
            datas_qddot_joints[:, data_tmp["controls"][0]["qddot_joints"].shape[1] - 1:] = \
                data_tmp["controls"][1]["qddot_joints"]

            # qddotb
            datas_qddot_floating_base = np.zeros((6, datas_shape[1]))
            for i in range(datas_shape[1]):
                datas_qddot_floating_base[:, i] = model.ForwardDynamicsFreeFloatingBase(
                    datas_q[:, i], datas_qdot[:, i], datas_qddot_joints[:, i]
                ).to_array()

            datas_qddot = np.concatenate((datas_qddot_floating_base, datas_qddot_joints), axis=0)

            datas_tau = np.zeros((9, datas_shape[1]))

            for i in range(datas_shape[1]):
                datas_tau[:, i] = \
                    model.InverseDynamics(datas_q[:, i], datas_qdot[:, i], datas_qddot[:, i]).to_array()[6:]

        else:
            tau_shape = (
                data_tmp["controls"][0]["tau"].shape[0],
                data_tmp["controls"][0]["tau"].shape[1] - 1 + data_tmp["controls"][1]["tau"].shape[1])
            datas_tau = np.zeros((tau_shape[0], tau_shape[1]))
            datas_tau[:, :data_tmp["controls"][0]["tau"].shape[1] - 1] = data_tmp["controls"][0]["tau"][:, :-1]
            datas_tau[:, data_tmp["controls"][0]["tau"].shape[1] - 1:] = data_tmp["controls"][1]["tau"]

        return np.asarray(datas_q), np.asarray(datas_qdot), np.asarray(datas_time), np.asarray(datas_tau)


def work_f_dx(
        tau1: np.ndarray,
        q1: np.ndarray,
):
    """
    Calculates the work produced by the athlete during the jump.

    Parameters
    ----------
    tau1: np.ndarray
        The generalized controls.
    q1: np.ndarray
        The generalized coordinates.

    Returns
    -------
    work: np.ndarray
        The work produced since the timestep 0.
    """
    dq = np.zeros(q1[6:, :].shape)
    dq[:, 1:] = q1[6:, 1:] - q1[6:, :-1]
    dw = np.zeros(dq.shape)
    dw[:, 1:] = tau1[:, :-1] * dq[:, 1:]
    W = np.zeros(dw.shape)
    for i in range(dw.shape[1] - 1):
        W[:, i+1] = W[:, i] + dw[:, i+1]

    return W.sum(axis=0)


def discrete_mechanical_energy(
        biorbd_model: biorbd.Model,
        q1: np.ndarray,
        qdot1: np.ndarray,
) -> np.ndarray:
    """
    Computes the discrete mechanical energy (kinetic energy + potential gravity energy) of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q1: np.ndarray
        The generalized coordinates
    qdot1: np.ndarray
        The generalized velocities

    Returns
    -------
    The discrete total energy
    """
    n_frames = q1.shape[1]
    d_total_energy = np.zeros(n_frames)
    for i in range(n_frames):
        d_total_energy[i] = biorbd_model.KineticEnergy(q1[:, i], qdot1[:, i]) + biorbd_model.PotentialEnergy(q1[:, i])
    return d_total_energy


if __name__ == "__main__":
    model = biorbd.Model(Models.ACROBAT.value)

    # Energy error % time
    fig, ax = plt.subplots()

    dic_heights = {
        "RK4_0317": [3, 5, 10, 15, 20, 25],
        "COLLOCATION_0324": [3, 5, 10, 15, 20, 25],
        "ACC-DRIVEN_RK4_0327": [3, 5, 10, 15, 20, 25],
    }

    dic_markers = {
        "RK4_0317": "+",
        "COLLOCATION_0324": ".",
        "ACC-DRIVEN_RK4_0327": "x"
    }

    dic_colors = {
        "3": sns.color_palette()[0],
        "5": sns.color_palette()[1],
        "10": sns.color_palette()[2],
        "15": sns.color_palette()[3],
        "20": sns.color_palette()[4],
        "25": sns.color_palette()[5],
    }

    data = []

    for j, (ode_solver, heights) in enumerate(dic_heights.items()):
        for height in heights:
            q, qdot, time, tau = get_created_data_from_pickle(f"{height}m_{ode_solver}", ode_solver)

            energy = discrete_mechanical_energy(model, q, qdot) - work_f_dx(tau, q)
            energy = abs(energy - energy[0]) + 1e-10

            # if ode_solver == "ACC-DRIVEN_RK4_0327":
            #     q2, qdot2, time2, tau2 = get_created_data_from_pickle(f"{height}m_{ode_solver}", ode_solver, False)
            #
            #     energy2 = discrete_mechanical_energy(model, q2, qdot2) - work_f_dx(tau2, q2)
            #     energy2 = abs(energy2 - energy2[0]) + 1e-10
            #     for e2 in energy2:
            #         data.append({'height': height, 'ACC-DRIVEN_RK4': e2})

            for e in energy:
                if ode_solver == "RK4_0317":
                    data.append({'height': height, 'RK4': e})
                elif ode_solver == "COLLOCATION_0324":
                    data.append({'height': height, 'COLLOCATION': e})
                elif ode_solver == "ACC-DRIVEN_RK4_0327":
                    data.append({'height': height, 'ACC-DRIVEN_RK4_REINT': e})

            # Energy error % time
            ax.semilogy(
                time, energy,
                color=dic_colors[f"{height}"], marker=dic_markers[ode_solver], markersize=4, linestyle='-',
                label=f"{height}m {ode_solver[:-5]}"
            )

    # Energy error % time
    ax.set_title("Energy conservation in function of the jump height and ODE solver")
    ax.set_xlabel("Times (s)")
    ax.set_ylabel("Absolute energy error compared to the energy at the first iteration (J)")

    lines = [Line2D([0], [0], marker=line_style, color='black', linestyle='-') for line_style in dic_markers.values()]
    legend1 = ax.legend(
        lines, [ode_solver[:-5] for ode_solver in dic_markers.keys()], loc='upper left', title='ODE solver'
    )

    colors = [Patch(facecolor=color) for color in dic_colors.values()]
    legend2 = ax.legend(colors, [height + " m" for height in dic_colors.keys()], loc='upper right', title='Heights')

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Boxplot energy error
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='height', y='value', hue='variable',
                data=pd.melt(df, id_vars=['height'], var_name='variable', value_name='value'), ax=ax, showfliers=False)
    ax.set_yscale('log')
    sns.stripplot(x='height', y='value', hue='variable',
                  data=pd.melt(df, id_vars=['height'], var_name='variable', value_name='value'), dodge=True,
                  jitter=True, color='black', alpha=0.5, ax=ax, size=2.0, label=None)
    ax.set_xlabel('Height (m)')
    ax.set_ylabel("Absolute energy error compared to the energy at the first iteration (J)")
    ax.set_title("Energy conservation in function of the jump height and ODE solver")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], labels[:3] + ['Data'], title='ODE solver', bbox_to_anchor=(0, 1), loc='upper left')

    plt.show()
