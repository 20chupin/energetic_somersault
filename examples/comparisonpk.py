import matplotlib.pyplot as plt
import numpy as np

import biorbd
import bioviz
from somersault import Models

import pickle


def get_created_data_from_pickle(file: str):
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
        print(f"{file}: cost:{data_tmp['cost']}, time to optimize: {data_tmp['real_time_to_optimize']}, "
              f"nb_it: {data_tmp['iterations']}")
        print(
            f"1ère phase : {data_tmp['time'][0][-1] - data_tmp['time'][0][0]}, {data_tmp['states'][0]['q'].shape[1]} "
            f"noeuds")
        print(
            f"2ère phase : {data_tmp['time'][1][-1] - data_tmp['time'][1][0]}, {data_tmp['states'][1]['q'].shape[1]} "
            f"noeuds")

        datas_shape = (
            data_tmp["states"][0]["q"].shape[0],
            data_tmp["states"][0]["q"].shape[1] + data_tmp["states"][1]["q"].shape[1])

        datas_q = np.zeros(datas_shape)
        datas_q[:, :data_tmp["states"][0]["q"].shape[1]] = data_tmp["states"][0]["q"]
        datas_q[:, data_tmp["states"][0]["q"].shape[1]:] = data_tmp["states"][1]["q"]

        datas_qdot = np.zeros(datas_shape)
        datas_qdot[:, :data_tmp["states"][0]["qdot"].shape[1]] = data_tmp["states"][0]["qdot"]
        datas_qdot[:, data_tmp["states"][0]["qdot"].shape[1]:] = data_tmp["states"][1]["qdot"]

        datas_time = np.zeros(datas_shape[1])
        datas_time[:data_tmp["time"][0].shape[0]] = data_tmp["time"][0]
        datas_time[data_tmp["time"][0].shape[0]:] = data_tmp["time"][1]

        # datas_tau = data_tmp["controls"]["tau"]
        # data_status = data_tmp["status"]
        # data_mus = data_tmp["controls"]["muscles"]
        # data_time = data_tmp["real_time_to_optimize"]
        # data_it = data_tmp["iterations"]
        data_cost = data_tmp["detailed_cost"]

        return np.asarray(datas_q), np.asarray(datas_qdot), np.asarray(datas_time), datas_shape[1]


def discrete_linear_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        qdot: np.ndarray,
):
    """
    Compute the discrete linear momentum of a biorbd model

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
    The discrete total energy
    """
    n_frames = qdot.shape[1]
    d_linear_momentum = np.zeros(n_frames)
    d_linear_momentum_x = np.zeros(n_frames)
    d_linear_momentum_y = np.zeros(n_frames)
    d_linear_momentum_z = np.zeros(n_frames)
    for i in range(n_frames):
        v = biorbd_model.CoMdot(q[:, i], qdot[:, i]).to_array()
        d_linear_momentum_x[i] = biorbd_model.mass() * v[0]
        d_linear_momentum_y[i] = biorbd_model.mass() * v[1]
        d_linear_momentum_z[i] = biorbd_model.mass() * v[2]
        d_linear_momentum[i] = biorbd_model.mass() * np.linalg.norm(v)
    return d_linear_momentum_x, d_linear_momentum_y, d_linear_momentum_z, d_linear_momentum


def discrete_angular_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        qdot: np.ndarray,
) -> np.ndarray:
    """
    Computes the discrete angular momentum of a biorbd model

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


def discrete_total_energy(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        qdot: np.ndarray,
) -> np.ndarray:
    """
    Computes the discrete total energy of a biorbd model

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
    The discrete total energy
    """
    n_frames = q.shape[1]
    d_total_energy = np.zeros(n_frames)
    for i in range(n_frames):
        d_total_energy[i] = biorbd_model.KineticEnergy(q[:, i], qdot[:, i]) + biorbd_model.PotentialEnergy(q[:, i])
    return d_total_energy


if __name__ == "__main__":
    height = 15
    q, qdot, time, ns = get_created_data_from_pickle(f"{height}m_RK4_0317")

    model = biorbd.Model(Models.ACROBAT.value)

    # Video
    b = bioviz.Viz(Models.ACROBAT.value, show_now=False, show_meshes=True, show_global_center_of_mass=False,
                   show_gravity_vector=False, show_floor=False, show_segments_center_of_mass=False,
                   show_global_ref_frame=True, show_local_ref_frame=False, show_markers=False,
                   show_muscles=False,
                   show_wrappings=False, mesh_opacity=1.0, )
    b.load_movement(q)
    b.set_camera_roll(b.get_camera_roll() - np.pi / 2)
    b.set_camera_position(b.get_camera_position()[0], b.get_camera_position()[1], height)
    b.resize(1000, 2000)
    b.start_recording(f"{height}m_RK4_new_pen_video")

    for f in range(ns + 2):
        b.movement_slider[0].setValue(f)
        b.add_frame()
    b.stop_recording()
    b.quit()

    # # height = 15
    # #
    # delta_energy = []
    # delta_am = []
    # delta_lm = []
    #
    # plt.figure(1)
    # plt.title("Total energy")
    # plt.figure(2)
    # fig_time, axs_time = plt.subplots(1, 3, sharex=True)
    # fig_delta, axs_delta = plt.subplots(1, 3, sharex=True)
    # fig_lm, axs_lm = plt.subplots(1, 3, sharex=True)
    # fig_am, axs_am = plt.subplots(1, 3, sharex=True)
    #
    # dic_heights = {
    #     "RK4": [1, 3, 5, 10, 15, 20, 25],
    #     "RK8": [3],
    #     "COLLOCATION": [3, 5],
    # }
    #
    # model = biorbd.Model(Models.ACROBAT.value)
    #
    # for ode_solver, heights in dic_heights.items():
    #     for height in heights:
    #         q, qdot, time, ns = get_created_data_from_pickle(f"{height}m_{ode_solver}")
    #
    #         # Energies
    #         energy = discrete_total_energy(model, q, qdot)
    #         angular_momentum_x, angular_momentum_y, angular_momentum_z, angular_momentum = discrete_angular_momentum(model, q, qdot)
    #         linear_momentum_x, linear_momentum_y, linear_momentum_z, linear_momentum = discrete_linear_momentum(model, q, qdot)
    #         delta_energy.append(energy)
    #         delta_am.append(angular_momentum)
    #         delta_lm.append(linear_momentum)
    #
    #         plt.figure(1)
    #         plt.plot(time, energy, label=f"{height}m_{ode_solver}")
    #
    #         axs_time[0].plot(time, energy, label=f"{height}m_{ode_solver}")
    #         axs_time[1].plot(time, angular_momentum, label=f"{height}m_{ode_solver}")
    #         axs_time[2].plot(time, linear_momentum, label=f"{height}m_{ode_solver}")
    #
    #         axs_lm[0].plot(time, linear_momentum_x, marker="o", ms=3, label=f"{height}m_{ode_solver}")
    #         axs_lm[1].plot(time, linear_momentum_y, label=f"{height}m_{ode_solver}")
    #         axs_lm[2].plot(time, linear_momentum_z, label=f"{height}m_{ode_solver}")
    #
    #         axs_am[0].plot(time, angular_momentum_x, label=f"{height}m_{ode_solver}")
    #         axs_am[1].plot(time, angular_momentum_y, label=f"{height}m_{ode_solver}")
    #         axs_am[2].plot(time, angular_momentum_z, label=f"{height}m_{ode_solver}")
    #
    # # axs_delta[0].boxplot(delta_energy, labels=heights)
    # # axs_delta[1].boxplot(delta_am, labels=heights)
    # # axs_delta[2].boxplot(delta_lm, labels=heights)
    # # axs_time[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    #
    # plt.figure(1)
    # plt.legend()
    #
    # axs_time[0].set_title("Total energy")
    # axs_time[1].set_title("Angular momentum norm")
    # axs_time[2].set_title("Linear momentum norm")
    #
    # axs_lm[0].set_title("Linear momentum x")
    # axs_lm[1].set_title("Linear momentum y")
    # axs_lm[2].set_title("Linear momentum z")
    # axs_lm[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    #
    # axs_am[0].set_title("Angular momentum x")
    # axs_am[1].set_title("Angular momentum y")
    # axs_am[2].set_title("Angular momentum z")
    # axs_am[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    #
    # axs_delta[0].set_title("Energy")
    # axs_delta[1].set_title("Angular momentum")
    # axs_delta[2].set_title("Linear momentum")
    # fig_delta.suptitle("RK4")
    # axs_time[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    # plt.show()
