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
        datas_q = data_tmp["states"]["q"]
        datas_qdot = data_tmp["states"]["qdot"]
        datas_time = data_tmp["time"]
        # datas_tau = data_tmp["controls"]["tau"]
        # data_status = data_tmp["status"]
        # data_mus = data_tmp["controls"]["muscles"]
        # data_time = data_tmp["real_time_to_optimize"]
        # data_it = data_tmp["iterations"]
        # data_cost = data_tmp["detailed_cost"]

        return datas_q, datas_qdot, datas_time,  # datas_tau, data_status, data_it, data_time, data_cost


def discrete_linear_momentum(
        biorbd_model: biorbd.Model,
        qdot: np.ndarray,
):
    """
    Compute the discrete linear momentum of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    qdot: np.ndarray
        The generalized velocities

    Returns
    -------
    The discrete total energy
    """
    n_frames = qdot.shape[1]
    d_linear_momentum = np.zeros(n_frames)
    for i in range(n_frames):
        d_linear_momentum[i] = biorbd_model.mass() * np.linalg.norm(qdot[:2, i])
    return d_linear_momentum


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
    d_angular_momentum = np.zeros(n_frames)
    for i in range(n_frames):
        d_angular_momentum[i] = np.linalg.norm(biorbd_model.angularMomentum(q[:, i], qdot[:, i]).to_array())
    return d_angular_momentum


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
    q, qdot, time = get_created_data_from_pickle(f"2m")
    # b = bioviz.Viz(Models.ACROBAT.value, show_floor=True, show_meshes=True)
    # b.load_movement(q)
    # b.exec()

    plt.plot(q[3, :] / (2 * np.pi))
    plt.show()

    delta_energy = []
    delta_am = []
    delta_lm = []

    fig_time, axs_time = plt.subplots(1, 3, sharex=True)
    fig_delta, axs_delta = plt.subplots(1, 3, sharex=True)

    heights = [1, 2]

    for height in heights:
        q, qdot, time = get_created_data_from_pickle(f"{height}m")

        model = biorbd.Model(Models.ACROBAT.value)

        energy = discrete_total_energy(model, q, qdot)
        angular_momentum = discrete_angular_momentum(model, q, qdot)
        linear_momentum = discrete_linear_momentum(model, qdot)
        delta_energy.append(energy[-1] - energy[0])
        delta_am.append(angular_momentum[-1] - angular_momentum[0])
        delta_lm.append(linear_momentum[-1] - linear_momentum[0])

        axs_time[0].plot(time, energy, label=f"{height}m")
        axs_time[1].plot(time, angular_momentum, label=f"{height}m")
        axs_time[2].plot(time, linear_momentum, label=f"{height}m")

    axs_delta[0].plot(heights, delta_energy, label=f"RK4")
    axs_delta[1].plot(heights, delta_am, label=f"RK4")
    axs_delta[2].plot(heights, delta_lm, label=f"RK4")

    axs_time[0].set_title("Total energy")
    axs_time[1].set_title("Angular momentum norm")
    axs_time[2].set_title("Linear momentum norm")
    axs_delta[0].set_title("Delta energy")
    axs_delta[1].set_title("Delta angular momentum")
    axs_delta[2].set_title("Delta linear momentum")
    axs_time[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    axs_delta[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    plt.show()