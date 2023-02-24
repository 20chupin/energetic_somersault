"""
This script runs the miller optimal control problem with a given set of parameters and save the results.
The main function is used in main_comparison.py and main_convergence.py. to run the different Miller optimal control problem.
"""
import os
from typing import Any
import numpy as np
import pickle
from time import time

import biorbd
from bioptim import (
    Solver,
    Shooting,
    SolutionIntegrator,
    CostType,
    DynamicsFcn,
)


def torque_driven_dynamics(
    model: biorbd.Model,
    states: np.array,
    controls: np.array,
    params: np.array,
    fext: np.array,
) -> np.ndarray:
    q = states[: model.nbQ()]
    qdot = states[model.nbQ() :]
    tau = controls
    if fext is None:
        qddot = model.ForwardDynamics(q, qdot, tau).to_array()
    else:
        fext_vec = biorbd.VecBiorbdVector()
        fext_vec.append(fext)
        qddot = model.ForwardDynamics(q, qdot, tau, biorbd.VecBiorbdSpatialVector(), fext_vec).to_array()
    return np.hstack((qdot, qddot))


class RunOCP:
    def __init__(
        self,
        ocp_class: Any,
        show_optim: bool = False,
        iteration: int = 10000,
        print_level: int = 5,
        ignore_already_run: bool = True,
    ):

        self.ocp_class = ocp_class
        self.show_optim = show_optim
        self.iteration = iteration
        self.print_level = print_level
        self.ignore_already_run = ignore_already_run

    def main(self, args: list = None):
        """
        Main function for the run_miller.py script.
        It runs the optimization and saves the results of a Miller Optimal Control Problem.

        Parameters
        ----------
        args : list
            List of arguments containing the following:
            args[0] : biorbd_model_path
                Path to the biorbd model.
            args[1] : ode_solver
                Ode solver to use.
            args[2] : n_shooting
                Number of shooting points.
            args[3] : n_threads
                Number of threads to use.
            args[4] : dynamics_type
                Dynamics type to use.
            args[5] : dynamics_function
                Dynamics function to use.
            args[6] : out_path
                Path to the output folder.
            args[7] : i_rand
                Random seed to use.

        """
        if args:
            biorbd_model_path = args[0]
            ode_solver = args[1]
            n_shooting = args[2]
            n_threads = args[3]
            dynamics_type = args[4]
            dynamics_function = args[5]
            twists = args[6]
            out_path_raw = args[7]
            i_rand = args[8]
        else:
            biorbd_model_path = args[0]
            ode_solver = args[1]
            n_shooting = args[2]
            n_threads = args[3]
            dynamics_type = args[4]
            dynamics_function = args[5]
            twists = args[6]
            out_path_raw = args[7]
            i_rand = args[8]

        str_ode_solver = ode_solver.__str__().replace("\n", "_").replace(" ", "_")
        str_dynamics_type = (
            dynamics_function.__str__().replace("DynamicsFcn.", "").replace("\n", "_").replace(" ", "_")
        )
        filename = (
            f"sol_irand{i_rand}_{n_shooting}_{str_ode_solver}_{ode_solver.defects_type.value}"
            f"_{str_dynamics_type}_{int(np.round(twists/(2*np.pi),decimals=0))}"
        )
        outpath = f"{out_path_raw}/" + filename

        # check if this file already exists if yes return
        if self.ignore_already_run:
            print(outpath)
            if os.path.isfile(outpath + ".pckl"):
                print("Already run ! Skipping...")
                return

        # --- Solve the program --- #
        my_ocp = self.ocp_class(
            biorbd_model_path=biorbd_model_path,
            rigidbody_dynamics=dynamics_type,
            dynamics_function=dynamics_function,
            twists=twists,
            n_shooting=n_shooting,
            ode_solver=ode_solver,
            n_threads=n_threads,
            seed=i_rand,
        )

        # --- Solve the program --- #
        print("Show online optimization", self.show_optim)
        solver = Solver.IPOPT(show_online_optim=self.show_optim, show_options=dict(show_bounds=True))

        solver.set_maximum_iterations(self.iteration)
        solver.set_print_level(self.print_level)
        # solver.set_convergence_tolerance(1e-10)
        solver.set_linear_solver("ma57")

        my_ocp.ocp.add_plot_penalty(CostType.ALL)
        if self.show_optim:
            my_ocp.ocp.print(to_console=True, to_graph=False)

        print(f"##########################################################")
        print(
            f"Solving ... \n"
            f"filename: {filename} \n"
            f"i_rand={i_rand},\n"
            f"dynamics_type={dynamics_type},\n"
            f"ode_solver={str_ode_solver},\n"
            f"n_shooting={n_shooting},\n"
            f"n_threads={n_threads}\n"
            f"{dynamics_function}\n"
            f"twists = {twists}\n"
        )
        print(f"##########################################################")

        # --- time to solve --- #
        tic = time()
        sol = my_ocp.ocp.solve(solver)
        toc = time() - tic

        sol.print_cost()

        print(f"#################################################### done ")
        print(
            f"Solved in {toc} sec \n"
            f"i_rand={i_rand},\n"
            f"dynamics_type={dynamics_type},\n"
            f"ode_solver={str_ode_solver},\n"
            f"n_shooting={n_shooting},\n"
            f"n_threads={n_threads}\n"
            f"{dynamics_function}\n"
            f"twists = {twists}\n"
        )
        print(f"##################################################### done ")

        qddot = self.recompute_qddot(biorbd_model_path, sol)

        sol_integrated = sol.integrate(
            shooting_type=Shooting.SINGLE,
            keep_intermediate_points=False,
            merge_phases=True,
            integrator=SolutionIntegrator.SCIPY_DOP853,
        )

        sol_merged = sol.merge_phases()

        f = open(f"{outpath}.pckl", "wb")
        data = {
            "model_path": biorbd_model_path,
            "phase_time": my_ocp.phase_time,
            "irand": i_rand,
            "computation_time": toc,
            "cost": sol.cost,
            "detailed_cost": sol.detailed_cost,
            "iterations": sol.iterations,
            "status": sol.status,
            "states": sol.states,
            "controls": sol_merged.controls,
            "parameters": sol.parameters,
            "time": sol_integrated.time,
            "dynamics_type": dynamics_type,
            "ode_solver": ode_solver,
            "ode_solver_str": ode_solver.__str__().replace("\n", "_").replace(" ", "_"),
            "defects_type": ode_solver.defects_type,
            "tau": sol_merged.controls["tau"] if dynamics_function == DynamicsFcn.TORQUE_DRIVEN else None,
            "qddot_joints": sol_merged.controls["qddot_joints"] if dynamics_function == DynamicsFcn.JOINTS_ACCELERATION_DRIVEN else None,
            "q": sol_merged.states_no_intermediate["q"],
            "qdot": sol_merged.states_no_intermediate["qdot"],
            "qddot": qddot,
            "q_integrated": sol_integrated.states["q"],
            "qdot_integrated": sol_integrated.states["qdot"],
            "n_shooting": n_shooting,
            "n_theads": n_threads,
            "twists": twists,
            "dynamics_function":dynamics_function,
        }

        pickle.dump(data, f)
        f.close()
        try:
            my_ocp.ocp.save(sol, f"{outpath}.bo")
        except:
            print("could not save the .bo file")

    @staticmethod
    def recompute_qddot(biorbd_model_path, sol):
        biorbd_model = biorbd.Model(biorbd_model_path)
        qddot = list()
        if len(sol.phase_time) > 2:
            for p, (states, controls) in enumerate(zip(sol.states, sol.controls)):
                qddot.append(np.zeros((int(states["all"].shape[0] // 2), states["all"].shape[1])))
                for i, (x, u) in enumerate(zip(states["all"].T, controls["all"].T)):
                    states_dot = torque_driven_dynamics(
                        model=biorbd_model, states=x, controls=u, params=None, fext=None
                    )
                    qddot[p][:, i] = states_dot[biorbd_model.nbQ() :]
        else:
            p = 0
            states = sol.states
            controls = sol.controls
            qddot.append(np.zeros((int(states["all"].shape[0] // 2), states["all"].shape[1])))
            for i, (x, u) in enumerate(zip(states["all"].T, controls["all"].T)):
                states_dot = torque_driven_dynamics(model=biorbd_model, states=x, controls=u, params=None, fext=None)
                qddot[p][:, i] = states_dot[biorbd_model.nbQ() :]

        # merge qddot elements in one numpy array deleting the last node of each phase
        # and keeping the first node of each phase
        # qddot[p][:, -1] is not kept when merging phases except for the last phase
        qddot_list = [qddot_p[:, :-1] for qddot_p in qddot]
        qddot_list.append(np.expand_dims(qddot[-1][:, -1], axis=1))
        qddot = np.hstack(qddot_list)

        return qddot
