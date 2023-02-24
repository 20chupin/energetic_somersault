"""
This script is used to run the optimization for a lot of different parameters
"""
import os

from multiprocessing import cpu_count
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

from bioptim import OdeSolver, RigidBodyDynamics, DefectType, DynamicsFcn

from utils import generate_calls, run_pool
from somersault import MillerOcpOnePhase, Models
from run_ocp import RunOCP


def main(
    model: Models = None,
    iterations=10000,
    print_level=5,
    ignore_already_run=False,
    show_optim=False,
    seed_start=0,
    calls=1,
    twists=None,
):
    n_shooting = [125]
    run_ocp = RunOCP(
        ocp_class=MillerOcpOnePhase,
        show_optim=show_optim,
        iteration=iterations,
        print_level=print_level,
        ignore_already_run=ignore_already_run,
    )
    running_function = run_ocp.main

    ode_list = [
        OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
        OdeSolver.RK4(n_integration_steps=5),
        OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
        # OdeSolver.RK8(n_integration_steps=2),
        # # OdeSolver.CVODES(),
        # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
        # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
    ]
    dynamics_list = [
        DynamicsFcn.TORQUE_DRIVEN,
        DynamicsFcn.JOINTS_ACCELERATION_DRIVEN,
    ]
    twist_list = [
        # 2 * np.pi,
        # 4 * np.pi,
        # 6 * np.pi,
        twists,
    ]

    out_path = mkdir_result_folder(model)

    # --- Generate the parameters --- #
    n_thread = 8
    param = dict(
        model_str=[
            model.value,
        ],
        ode_solver=ode_list,
        n_shooting=n_shooting,
        n_thread=[n_thread],
        dynamic_type=[
            RigidBodyDynamics.ODE,
        ],
        dynamics=dynamics_list,
        twists=twist_list,
        out_path=[out_path.absolute().__str__()],
    )
    calls = int(calls)

    my_calls = generate_calls(
        call_number=calls,
        parameters=param,
        seed_start=seed_start,
    )

    cpu_number = cpu_count()
    my_pool_number = int(cpu_number / n_thread)

    # running_function(my_calls[0])
    # running_function(my_calls[1])
    columns = list(param.keys())
    columns.append("random")
    df = pd.DataFrame(my_calls, columns=columns)

    for ode_solver in ode_list:
        sub_df = df[df["ode_solver"] == ode_solver]
        my_calls = sub_df.to_numpy().tolist()
        run_pool(
            running_function=running_function,
            calls=my_calls,
            pool_nb=my_pool_number,
        )


def mkdir_result_folder(model):
    # --- Generate the output path --- #
    Date = date.today().strftime("%d-%m-%y")
    out_path = Path(
        Path(__file__).parent.__str__()
        + f"/../../msd-somersaults-results/"
        + f"ACROBAT_21-11-22"
          # f"{model.name}_{Date}"
    )
    try:
        os.mkdir(out_path)
    except:
        print(f"{out_path}" + Date + " is already created ")

    return out_path


if __name__ == "__main__":
    iteration = 3000
    main(
        model=Models.ACROBAT,
        iterations=iteration,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=10,
        calls=20,
        twists=2 * np.pi,
    )
    main(
        model=Models.ACROBAT,
        iterations=iteration,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=10,
        calls=20,
        twists=4 * np.pi,
    )
    main(
        model=Models.ACROBAT,
        iterations=iteration,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=10,
        calls=20,
        twists=6 * np.pi,
    )
