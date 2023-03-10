from bioptim import CostType, Solver, DynamicsFcn
from somersault import MillerOCP, MillerOcpOnePhase, Models

import pickle


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
    height = 20

    equation_of_motion = DynamicsFcn.TORQUE_DRIVEN

    model_path = Models.ACROBAT.value

    # --- Solve the program --- #
    miller = MillerOCP(
        biorbd_model_path=model_path,
        dynamics_function=equation_of_motion,
        n_threads=32,  # if your computer has enough cores, otherwise it takes them all
        seed=42,  # The sens of life
        jump_height=height
    )

    miller.ocp.add_plot_penalty(CostType.ALL)

    print("number of states: ", miller.ocp.v.n_all_x)
    print("number of controls: ", miller.ocp.v.n_all_u)

    miller.ocp.print(to_console=True, to_graph=False)

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(2000 * height)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = miller.ocp.solve(solv)

    save_results(sol, f"{height}m")

    # --- Show results --- #
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
