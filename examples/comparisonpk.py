import matplotlib.pyplot as plt
import numpy as np

import biorbd
import bioviz
from somersault import Models

import pickle


def get_created_data_from_pickle(file: str):
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
    linear_momentum = np.zeros((n_frames - 1, 1))
    for i in range(n_frames - 1):
        linear_momentum[i] = biorbd_model.mass() * np.linalg.norm(qdot[:2, i])
    return linear_momentum


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
    angular_momentum = np.zeros((n_frames, 1))
    for i in range(n_frames):
        angular_momentum[i] = np.linalg.norm(biorbd_model.angularMomentum(q[:, i], qdot[:, i]).to_array())
    return angular_momentum


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
    discrete_total_energy = np.zeros((n_frames, 1))
    for i in range(n_frames - 1):
        discrete_total_energy[i] = biorbd_model.KineticEnergy(q[:, i], qdot[:, i]) + biorbd_model.PotentialEnergy(q[:, i])
    return discrete_total_energy


if __name__ == "__main__":
    q, qdot, time = get_created_data_from_pickle("1m")
    print(Models.ACROBAT.value)
    # b = bioviz.Viz(Models.ACROBAT.value, show_floor=True, show_meshes=True)
    # b.load_movement(q)
    # b.exec()

    fig_time, axs_time = plt.subplots(1, 3, sharex=True)
    fig_delta, axs_delta = plt.subplots(1, 3, sharex=True)

    model = biorbd.Model(Models.ACROBAT.value)

    energy = discrete_total_energy(model, q, qdot)
    angular_momentum = discrete_angular_momentum(model, q, qdot)
    linear_momentum = discrete_linear_momentum(model, qdot)
    delta_energy = energy[-1] - energy[0]
    delta_am = angular_momentum[-1] - angular_momentum[0]
    delta_lm = linear_momentum[-1] - linear_momentum[0]

    axs_time[0].plot(time, energy)  #, label=ode_solver)
    axs_time[1].plot(time, angular_momentum)  #, label=ode_solver)
    axs_time[2].plot(time[:-1], linear_momentum)  #, label=ode_solver)
    axs_delta[0].plot(1.0, delta_energy, "+")  #, label=ode_solver)
    axs_delta[1].plot(1.0, delta_am, "+")  #, label=ode_solver)
    axs_delta[2].plot(1.0, delta_lm, "+")  #, label=ode_solver)

    axs_time[0].set_title("Total energy")
    axs_time[1].set_title("Angular momentum norm")
    axs_time[2].set_title("Linear momentum norm")
    axs_delta[0].set_title("Delta energy")
    axs_delta[1].set_title("Delta angular momentum")
    axs_delta[2].set_title("Delta linear momentum")
    axs_time[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    axs_delta[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
    plt.show()
