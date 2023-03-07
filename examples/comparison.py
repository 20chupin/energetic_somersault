import matplotlib.pyplot as plt
import numpy as np
import json

import biorbd
from varint.enums import QuadratureRule
from somersault import Models

import pickle
import bioviz


def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
        datas_q = data_tmp["states"]["q"]
        datas_qdot = data_tmp["states"]["qdot"]
        # datas_tau = data_tmp["controls"]["tau"]
        # data_status = data_tmp["status"]
        # data_mus = data_tmp["controls"]["muscles"]
        # data_time = data_tmp["real_time_to_optimize"]
        # data_it = data_tmp["iterations"]
        # data_cost = data_tmp["detailed_cost"]

        return datas_q, datas_qdot,  # datas_tau, data_status, data_it, data_time, data_cost


q, q_dot = get_created_data_from_pickle("1m")
b = bioviz.Viz(Models.ACROBAT.value, show_floor=True, show_meshes=True)
b.load_movement(q)
b.exec()


def linear_momentum_i(
        biorbd_model: biorbd.Model,
        q1: np.ndarray,
        q2: np.ndarray,
        time_step,
) -> np.ndarray:
    """
    Compute the angular momentum of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q1: np.ndarray
        The generalized coordinates at the first time step
    q2: np.ndarray
        The generalized coordinates at the second time step
    time_step: float
        The time step
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration.

    Returns
    -------
    The discrete total energy
    """
    qdot = (q2 - q1) / time_step

    return biorbd_model.mass() * np.linalg.norm(qdot[:2])


def discrete_linear_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        time: np.ndarray,
):
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time: np.ndarray
        The times

    Returns
    -------
    The discrete total energy
    """
    n_frames = q.shape[1]
    linear_momentum = np.zeros((n_frames - 1, 1))
    for i in range(n_frames - 1):
        linear_momentum[i] = linear_momentum_i(biorbd_model, q[:, i], q[:, i + 1], time[i + 1] - time[i])
    return linear_momentum


def delta_linear_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        time: np.ndarray,
) -> np.ndarray:
    """
    Compute the delta total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time: np.ndarray
        The times

    Returns
    -------
    The discrete total energy
    """
    discrete_angular_momentum_start = linear_momentum_i(biorbd_model, q[:, 0], q[:, 1], time[1] - time[0])
    discrete_angular_momentum_end = linear_momentum_i(biorbd_model, q[:, -2], q[:, -1], time[-2] - time[-1])
    return discrete_angular_momentum_end - discrete_angular_momentum_start


def angular_momentum_i(
        biorbd_model: biorbd.Model,
        q1: np.ndarray,
        q2: np.ndarray,
        time_step,
        discrete_approximation
) -> np.ndarray:
    """
    Compute the angular momentum of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q1: np.ndarray
        The generalized coordinates at the first time step
    q2: np.ndarray
        The generalized coordinates at the second time step
    time_step: float
        The time step
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration.

    Returns
    -------
    The discrete total energy
    """
    if discrete_approximation == QuadratureRule.MIDPOINT:
        q = (q1 + q2) / 2
    elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
        q = q1
    elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
        q = q2
    elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
        q = (q1 + q2) / 2
    else:
        raise NotImplementedError(
            f"Discrete energy computation {discrete_approximation} is not implemented"
        )
    qdot = (q2 - q1) / time_step
    return np.linalg.norm(biorbd_model.angularMomentum(q, qdot).to_array())


def angular_momentum_v(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        qdot: np.ndarray,
) -> np.ndarray:
    """
    Compute the angular momentum of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    qdot: np.ndarray
        The generalized velocity

    Returns
    -------
    The discrete total energy
    """
    return np.linalg.norm(biorbd_model.angularMomentum(q, qdot).to_array())


def discrete_angular_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        time: np.ndarray,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,

) -> np.ndarray:
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time: np.ndarray
        The times
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration, trapezoidal by default.

    Returns
    -------
    The discrete total energy
    """
    n_frames = q.shape[1]
    angular_momentum = np.zeros((n_frames - 1, 1))
    for i in range(n_frames - 1):
        angular_momentum[i] = angular_momentum_i(biorbd_model, q[:, i], q[:, i + 1], time[i + 1] - time[i],
                                                 discrete_approximation)
    return angular_momentum


def delta_angular_momentum(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        time: np.ndarray,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
) -> np.ndarray:
    """
    Compute the delta total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time: np.ndarray
        The times
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration, trapezoidal by default.

    Returns
    -------
    The discrete total energy
    """
    discrete_angular_momentum_start = angular_momentum_i(biorbd_model, q[:, 0], q[:, 1], time[1] - time[0],
                                                         discrete_approximation)
    discrete_angular_momentum_end = angular_momentum_i(biorbd_model, q[:, -2], q[:, -1], time[-2] - time[-1],
                                                       discrete_approximation)
    return discrete_angular_momentum_end - discrete_angular_momentum_start


def discrete_total_energy_i(
        biorbd_model: biorbd.Model,
        q1: np.ndarray,
        q2: np.ndarray,
        time_step,
        discrete_approximation
) -> np.ndarray:
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q1: np.ndarray
        The generalized coordinates at the first time step
    q2: np.ndarray
        The generalized coordinates at the second time step
    time_step: float
        The time step
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration.

    Returns
    -------
    The discrete total energy
    """
    if discrete_approximation == QuadratureRule.MIDPOINT:
        q = (q1 + q2) / 2
    elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
        q = q1
    elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
        q = q2
    elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
        q = (q1 + q2) / 2
    else:
        raise NotImplementedError(
            f"Discrete energy computation {discrete_approximation} is not implemented"
        )
    qdot = (q2 - q1) / time_step
    return biorbd_model.KineticEnergy(q, qdot) + biorbd_model.PotentialEnergy(q)


def discrete_total_energy(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        time: np.ndarray,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
) -> np.ndarray:
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time: np.ndarray
        The times
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration, trapezoidal by default.

    Returns
    -------
    The discrete total energy
    """
    n_frames = q.shape[1]
    discrete_total_energy = np.zeros((n_frames - 1, 1))
    for i in range(n_frames - 1):
        discrete_total_energy[i] = discrete_total_energy_i(biorbd_model, q[:, i], q[:, i + 1], time[i + 1] - time[i],
                                                           discrete_approximation)
    return discrete_total_energy


def delta_total_energy(
        biorbd_model: biorbd.Model,
        q: np.ndarray,
        time: np.ndarray,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
) -> np.ndarray:
    """
    Compute the delta total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time: np.ndarray
        The times
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration, trapezoidal by default.

    Returns
    -------
    The discrete total energy
    """
    discrete_total_energy_start = discrete_total_energy_i(biorbd_model, q[:, 0], q[:, 1], time[1] - time[0],
                                                          discrete_approximation)
    discrete_total_energy_end = discrete_total_energy_i(biorbd_model, q[:, -2], q[:, -1], time[-2] - time[-1],
                                                        discrete_approximation)
    return discrete_total_energy_end - discrete_total_energy_start

ode_solvers = [
    "RK4_1",
    "Collocation_explicit",
    "Collocation_implicit",
]

fig_time, axs_time = plt.subplots(1, 3, sharex=True)
fig_delta, axs_delta = plt.subplots(1, 3, sharex=True)

for ode_solver in ode_solvers:
    with open(ode_solver + ".json", "r") as file:
        time_pos = json.load(file)

    model = biorbd.Model(Models.ACROBAT.value)

    time = np.asarray(time_pos["time"])
    q = np.asarray(time_pos["q"])
    energy = discrete_total_energy(model, q, time)
    angular_momentum = discrete_angular_momentum(model, q, time)
    linear_momentum = discrete_linear_momentum(model, q, time)
    delta_energy = delta_total_energy(model, q, time)
    delta_am = delta_angular_momentum(model, q, time)
    delta_lm = delta_linear_momentum(model, q, time)

    axs_time[0].plot(time[:-1], energy, label=ode_solver)
    axs_time[1].plot(time[:-1], angular_momentum, label=ode_solver)
    axs_time[2].plot(time[:-1], linear_momentum, label=ode_solver)
    axs_delta[0].plot(1.0, delta_energy, "+", label=ode_solver)
    axs_delta[1].plot(1.0, delta_am, "+", label=ode_solver)
    axs_delta[2].plot(1.0, delta_lm, "+", label=ode_solver)

axs_time[0].set_title("Total energy")
axs_time[1].set_title("Angular momentum norm")
axs_time[2].set_title("Linear momentum norm")
axs_delta[0].set_title("Delta energy")
axs_delta[1].set_title("Delta angular momentum")
axs_delta[2].set_title("Delta linear momentum")
axs_time[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
axs_delta[2].legend(ncols=3, bbox_to_anchor=(0, -0.05))
plt.show()
