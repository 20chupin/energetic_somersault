from bioptim import CostType, Solver, DynamicsFcn, OdeSolver
from somersault import MillerOCP, Models
from bioptim.misc.enums import DefectType

import pickle
import numpy as np


def save_results(sol, c3d_file_path):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
    The solution to the ocp at the current pool
    c3d_file_path: str
    The path to the c3d file of the task
    """
    data = dict(
        states=sol.states,
        states_no_intermediate=sol.states_scaled_no_intermediate,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=sol.cost,
        detailed_cost=sol.detailed_cost,
        real_time_to_optimize=sol.real_time_to_optimize,
        status=sol.status,
        time=sol.time,
    )
    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def main():
    for height in (25, ):

        equation_of_motion = DynamicsFcn.JOINTS_ACCELERATION_DRIVEN

        model_path = Models.ACROBAT.value

        # --- Solve the program --- #
        miller = MillerOCP(
            ode_solver=OdeSolver.RK4(),
            biorbd_model_path=model_path,
            dynamics_function=equation_of_motion,
            n_threads=3,  # if your computer has enough cores, otherwise it takes them all
            jump_height=height
        )

        miller.ocp.add_plot_penalty(CostType.ALL)

        print("number of states: ", miller.ocp.v.n_all_x)
        print("number of controls: ", miller.ocp.v.n_all_u)

        miller.ocp.print(to_console=True, to_graph=False)

        solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
        solv.set_maximum_iterations(300 * height)
        solv.set_linear_solver("ma57")
        solv.set_print_level(5)
        sol = miller.ocp.solve(solv)

        save_results(sol, f"{height}m_ACC-DRIVEN_RK4_0327")

    # for height in (25,):
    #     equation_of_motion = DynamicsFcn.TORQUE_DRIVEN
    #
    #     model_path = Models.ACROBAT.value
    #
    #     ode_solver_obj = OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT)
    #
    #     # --- Solve the program --- #
    #     miller = MillerOCP(
    #         ode_solver=ode_solver_obj,
    #         biorbd_model_path=model_path,
    #         dynamics_function=equation_of_motion,
    #         n_threads=3,  # if your computer has enough cores, otherwise it takes them all
    #         jump_height=height,
    #     )
    #
    #     miller.ocp.add_plot_penalty(CostType.ALL)
    #
    #     print("number of states: ", miller.ocp.v.n_all_x)
    #     print("number of controls: ", miller.ocp.v.n_all_u)
    #
    #     miller.ocp.print(to_console=True, to_graph=False)
    #
    #     solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    #     solv.set_maximum_iterations(1000 * height)
    #     solv.set_linear_solver("ma57")
    #     solv.set_print_level(5)
    #     sol = miller.ocp.solve(solv)
    #
    #     save_results(sol, f"{height}m_COLLOCATION_0324")

    # --- Show results --- #
    # sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
