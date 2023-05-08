import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


import biorbd
import bioviz
from somersault import Models

import pickle


def get_created_data_from_pickle(file: str, ode_solver):
    """
    Creates data from pickle

    Parameters
    ----------
    file: str
        File where data has been saved

    Returns
    -------
    Data
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
        print(f"{file}: status: {data_tmp['status']}, cost:{data_tmp['cost']}, time to optimize: "
              f"{data_tmp['real_time_to_optimize']}, "
              f"nb_it: {data_tmp['iterations']}")
        print(
            f"1ère phase : {data_tmp['time'][0][-1] - data_tmp['time'][0][0]}, {data_tmp['states_no_intermediate'][0]['q'].shape[1]} "
            f"nodes")
        print(
            f"2ère phase : {data_tmp['time'][1][-1] - data_tmp['time'][1][0]}, {data_tmp['states_no_intermediate'][1]['q'].shape[1]} "
            f"nodes")

        shape1 = data_tmp["states_no_intermediate"][0]["q"].shape[1] - 1

        datas_shape = (
            data_tmp["states_no_intermediate"][0]["q"].shape[0],
            shape1 + data_tmp["states_no_intermediate"][1]["q"].shape[1]
        )

        datas_q = np.zeros(datas_shape)
        datas_q[:, :shape1] = data_tmp["states_no_intermediate"][0]["q"][:, :-1]
        datas_q[:, shape1:] = data_tmp["states_no_intermediate"][1]["q"]

        datas_shape_colloc = (
            data_tmp["states"][0]["q"].shape[0],
            data_tmp["states"][0]["q"].shape[1] - 1 + data_tmp["states"][1]["q"].shape[1]
        )

        datas_q_colloc = np.zeros(datas_shape_colloc)
        datas_q_colloc[:, :data_tmp["states"][0]["q"].shape[1] - 1] = data_tmp["states"][0]["q"][:, :-1]
        datas_q_colloc[:, data_tmp["states"][0]["q"].shape[1] - 1:] = data_tmp["states"][1]["q"]

        datas_qdot = np.zeros(datas_shape)
        datas_qdot[:, :shape1] = data_tmp["states_no_intermediate"][0]["qdot"][:, :-1]
        datas_qdot[:, shape1:] = data_tmp["states_no_intermediate"][1]["qdot"]

        datas_qdot_colloc = np.zeros(datas_shape_colloc)
        datas_qdot_colloc[:, :data_tmp["states"][0]["qdot"].shape[1] - 1] = data_tmp["states"][0]["qdot"][:, :-1]
        datas_qdot_colloc[:, data_tmp["states"][0]["qdot"].shape[1] - 1:] = data_tmp["states"][1]["qdot"]

        datas_time = np.zeros(datas_shape[1])
        if data_tmp["states_no_intermediate"][0]["q"].shape[1] == data_tmp["states"][0]["q"].shape[1]:
            step = 1
        else:
            step = 5
        datas_time[:shape1] = data_tmp["time"][0][:-1:step]
        datas_time[shape1:] = data_tmp["time"][1][::step]

        datas_time_colloc = np.zeros(datas_shape_colloc[1])
        datas_time_colloc[:data_tmp["states"][0]["qdot"].shape[1] - 1] = data_tmp["time"][0][:-1]
        datas_time_colloc[data_tmp["states"][0]["qdot"].shape[1] - 1:] = data_tmp["time"][1][:]

        if ode_solver == "ACC-DRIVEN_RK4_0327":
            # qddotj
            qddot_joints_shape = (
                data_tmp["controls"][0]["qddot_joints"].shape[0],
                data_tmp["controls"][0]["qddot_joints"].shape[1] - 1 + data_tmp["controls"][1]["qddot_joints"].shape[1])

            datas_qddot_joints = np.zeros((qddot_joints_shape[0], qddot_joints_shape[1]))
            datas_qddot_joints[:, :data_tmp["controls"][0]["qddot_joints"].shape[1] - 1] = data_tmp["controls"][0]["qddot_joints"][:, :-1]
            datas_qddot_joints[:, data_tmp["controls"][0]["qddot_joints"].shape[1] - 1:] = data_tmp["controls"][1]["qddot_joints"]

            datas_qddot_floating_base = np.zeros((6, datas_shape[1]))
            for i in range(datas_shape[1]):
                datas_qddot_floating_base[:, i] = model.ForwardDynamicsFreeFloatingBase(
                    datas_q[:, i], datas_qdot[:, i], datas_qddot_joints[:, i]
                ).to_array()

            datas_qddot = np.concatenate((datas_qddot_floating_base, datas_qddot_joints), axis=0)

            datas_tau = np.zeros((9, datas_shape[1]))

            for i in range(datas_shape[1]):
                datas_tau[:, i] = model.InverseDynamics(datas_q[:, i], datas_qdot[:, i], datas_qddot[:, i]).to_array()[6:]
        else:
            tau_shape = (
                data_tmp["controls"][0]["tau"].shape[0],
                data_tmp["controls"][0]["tau"].shape[1] - 1 + data_tmp["controls"][1]["tau"].shape[1])
            datas_tau = np.zeros((tau_shape[0], tau_shape[1]))
            datas_tau[:, :data_tmp["controls"][0]["tau"].shape[1] - 1] = data_tmp["controls"][0]["tau"][:, :-1]
            datas_tau[:, data_tmp["controls"][0]["tau"].shape[1] - 1:] = data_tmp["controls"][1]["tau"]

        # data_status = data_tmp["status"]
        # data_mus = data_tmp["controls"]["muscles"]
        # data_time = data_tmp["real_time_to_optimize"]
        # data_it = data_tmp["iterations"]
        # data_cost = data_tmp["detailed_cost"]

        return (
            np.asarray(datas_q), np.asarray(datas_qdot), np.asarray(datas_time), np.asarray(datas_tau), datas_shape[1],
            np.asarray(datas_q_colloc), np.asarray(datas_qdot_colloc), np.asarray(datas_time_colloc)
        )


def discrete_angular_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        qdot: np.ndarray,
) -> np.ndarray:
    """
    Computes the discrete angular momentum of workflows biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    qdot: np.ndarray
        The generalized velocities

    Returns
    -------
    The discrete angular_momentum
    """
    n_frames = q.shape[1]
    d_angular_momentum_x = np.zeros(n_frames)
    d_angular_momentum_y = np.zeros(n_frames)
    d_angular_momentum_z = np.zeros(n_frames)
    d_angular_momentum = np.zeros(n_frames)
    for i in range(n_frames):
        am = biorbd_model.angularMomentum(q[:, i], qdot[:, i]).to_array()
        d_angular_momentum_x[i] = am[0]
        d_angular_momentum_y[i] = am[1]
        d_angular_momentum_z[i] = am[2]
        d_angular_momentum[i] = np.linalg.norm(am)
    return d_angular_momentum_x, d_angular_momentum_y, d_angular_momentum_z, d_angular_momentum

if __name__ == "__main__":
    delta_am = []
    delta_lm = []

    fig, ax = plt.subplots()

    model = biorbd.Model(Models.ACROBAT.value)

    q, qdot, time, tau, ns, q_colloc, qdot_colloc, time_colloc = get_created_data_from_pickle(f"20m_COLLOCATION_0324", "20m_COLLOCATION_0324")

    ax.plot(time_colloc, discrete_angular_momentum(model, q_colloc, qdot_colloc)[1], marker="+", linestyle="None", label="y-angular momentum on the intermediary nodes")
    ax.plot(time, discrete_angular_momentum(model, q, qdot)[1], label="y-angular momentum on the principal nodes")
    plt.xlabel("Time (s)")
    plt.ylabel("y-angular momentum (kg m^2 / s)")
    plt.title("Angular momentum with collocation")
    plt.legend()

    plt.show()
