import matplotlib.pyplot as plt
import numpy as np

from bioptim import OdeSolver, CostType, RigidBodyDynamics, Solver, DefectType, Shooting, SolutionIntegrator, DynamicsFcn
from somersault import MillerOcpOnePhase, Models



def main():
    # --- ODE SOLVER Options --- #
    # One can pick any of the following ODE solvers:

    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4)
    ode_solver = OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4)

    # --- EQUATIONS OF MOTION --- #
    # One can pick any of the following equations of motion:
    equation_of_motion = DynamicsFcn.JOINTS_ACCELERATION_DRIVEN
    # equation_of_motion = DynamicsFcn.TORQUE_DRIVEN

    model_path = Models.ACROBAT.value

    # --- Solve the program --- #
    miller = MillerOcpOnePhase(
        biorbd_model_path=model_path,
        n_shooting=125,
        ode_solver=ode_solver,
        dynamics_function=equation_of_motion,
        twists=2 * np.pi,  # try to add more twists with : 4 * np.pi or 6 * np.pi
        n_threads=32,  # if your computer has enough cores, otherwise it takes them all
        seed=42,  # The sens of life
    )

    miller.ocp.add_plot_penalty(CostType.ALL)

    print("number of states: ", miller.ocp.v.n_all_x)
    print("number of controls: ", miller.ocp.v.n_all_u)

    miller.ocp.print(to_console=True, to_graph=False)

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(500)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = miller.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()
    # sol.graphs(show_bounds=True)

    out = sol.integrate(
        shooting_type=Shooting.SINGLE,
        keep_intermediate_points=False,
        merge_phases=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    sol.animate(show_floor=False, show_gravity=False)

    # plt.figure()
    # plt.plot(sol.time, sol.states["q"].T, label="ocp", marker=".")
    # plt.plot(out.time, out.states["q"].T, label="integrated", marker="+")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
