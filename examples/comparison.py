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


def get_created_data_from_pickle(file: str, ode_solver_name: str):
    """
    Creates data from pickle

    Parameters
    ----------
    file: str
        File where data has been saved
    ode_solver_name: str
        Name of the ode_solver in the pickle file names.

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
        if ode_solver_name == "ACC-DRIVEN_RK4_0327":
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

            for e in energy:
                if ode_solver == "RK4_0317":
                    data.append({'height': height, 'RK4': e})
                elif ode_solver == "COLLOCATION_0324":
                    data.append({'height': height, 'COLLOCATION': e})
                elif ode_solver == "ACC-DRIVEN_RK4_0327":
                    data.append({'height': height, 'ACC-DRIVEN_RK4': e})

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
