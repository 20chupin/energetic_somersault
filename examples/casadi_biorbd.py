import time
import numpy as np
from casadi import MX, Function, jacobian, vertcat
import biorbd_casadi as biorbd
from typing import Callable

from somersault import Models

model = biorbd.Model(Models.ACROBAT.value)

q_mx = MX.sym("q", model.nbQ(), 1)
qdot_mx = MX.sym("qdot", model.nbQ(), 1)
u = MX.sym("u", model.nbQ(), 1)


def RK4(fun: Callable, x_prev: np.ndarray, u: np.ndarray, h : float = 0.05):
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
    k1 = fun(x_prev, u)
    k2 = fun(x_prev + h / 2 * k1, u)
    k3 = fun(x_prev + h / 2 * k2, u)
    k4 = fun(x_prev + h * k3, u)
    return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def fun(x, u):
    q = x[:model.nbQ()]
    qdot = x[model.nbQ():]
    return vertcat(qdot, model.ForwardDynamics(q, qdot, u).to_mx())


for cse in [True, False]:
    t0 = time.time()

    for i in range(10):
        casadi_issue = Function(
            "casadi_issue",
            [q_mx, qdot_mx, u],
            [RK4(fun, vertcat(q_mx, qdot_mx), u)],
            {"cse": cse}
        ).expand()

    print(cse, time.time() - t0)

    # m = int(1e5)
    #
    # q_list = []
    # qdot_list = []
    # qddot_list = []
    #
    # for i in range(m):
    #     q_list.append(np.random.rand(model.nbQ(), 1))
    #     qdot_list.append(np.random.rand(model.nbQ(), 1))
    #     qddot_list.append(np.random.rand(model.nbQ(), 1))
    #
    # t0 = time.time()
    #
    # for i in range(m):
    #     casadi_issue(q_list[i], qdot_list[i], qddot_list[i])
    #
    # print(cse, time.time() - t0)
