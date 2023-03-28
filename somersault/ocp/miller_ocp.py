from enum import Enum

import biorbd_casadi as biorbd
import biorbd as brd
import numpy as np
from scipy import interpolate
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    ControlType,
    Bounds,
    InterpolationType,
    PhaseTransitionList,
    BiMappingList,
    MultinodeConstraintList,
    RigidBodyDynamics,
    BiorbdModel
)


class MillerDynamics(Enum):
    """
    Selection of dynamics to perform the miller ocp
    """

    EXPLICIT = "explicit"
    ROOT_EXPLICIT = "root_explicit"
    IMPLICIT = "implicit"
    ROOT_IMPLICIT = "root_implicit"
    IMPLICIT_TAU_DRIVEN_QDDDOT = "implicit_qdddot"
    ROOT_IMPLICIT_QDDDOT = "root_implicit_qdddot"


class MillerOcp:
    """
    Class to generate the OCP for the miller acrobatic task for a 15-dof human model.

    Methods
    ----------
    _set_dynamics
        Set the dynamics of the OCP
    _set_objective
        Set the objective of the OCP
    _set_constraints
        Set the constraints of the OCP
    _set_bounds
        method to set the bounds of the OCP
    _set_initial_guess
        method to set the initial guess of the OCP
    _set_mapping
        method to set the mapping between variables of the model
    _print_bounds
        method to print the bounds of the states into the console
    """

    def __init__(
        self,
        biorbd_model_path: str = None,
        n_threads: int = 8,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        dynamics_function: DynamicsFcn = DynamicsFcn.TORQUE_DRIVEN,
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9168200/
        # https://www-ncbi-nlm-nih-gov.portail.psl.eu/pmc/articles/PMC4424452/
        vertical_velocity_0_min: float = 2.0,
        vertical_velocity_0_max: float = 3.0,
        somersaults: float = 2 * 2 * np.pi,
        twists: float = 2 * 2 * np.pi,
        jump_height: float = 1.0,
        use_sx: bool = False,
        extra_obj: bool = False,
        initial_x: InitialGuessList = None,
        initial_u: InitialGuessList = None,
        seed: int = None,
    ):
        """
        Parameters
        ----------
        biorbd_model_path : str
            path to the biorbd model
        n_shooting : tuple
            number of shooting points for each phase
        phase_durations : tuple
            duration of each phase
        n_threads : int
            number of threads to use for the solver
        ode_solver : OdeSolver
            type of ordinary differential equation solver to use
        rigidbody_dynamics : RigidBodyDynamics
            type of dynamics to use
        vertical_velocity_0 : float
            initial vertical velocity of the model to execute the Miller task
        somersaults : float
            number of somersaults to execute
        twists : float
            number of twists to execute
        use_sx : bool
            use SX for the dynamics
        extra_obj : bool
            use extra objective to the extra controls of implicit dynamics (algebraic states)
        initial_x : InitialGuessList
            initial guess for the states
        initial_u : InitialGuessList
            initial guess for the controls
        """
        self.biorbd_model_path = biorbd_model_path
        self.extra_obj = extra_obj
        self.n_phases = 2

        self.somersaults = somersaults
        self.twists = twists
        self.jump_height = jump_height

        self.x = None
        self.u = None

        self.vertical_velocity_0_min = vertical_velocity_0_min
        self.vertical_velocity_0_max = vertical_velocity_0_max

        vertical_velocity_0 = (vertical_velocity_0_min + vertical_velocity_0_max) / 2
        parable_duration = (vertical_velocity_0 + np.sqrt(vertical_velocity_0 ** 2 + 2 * 9.81 * jump_height)) / 9.81

        self.n_shooting = (int(100 * parable_duration), 14)

        self.phase_durations = (parable_duration, 0.193125)
        self.phase_time = self.phase_durations

        self.duration = np.sum(self.phase_durations)
        self.phase_proportions = (
            self.phase_durations[0] / self.duration,
            self.phase_durations[1] / self.duration,
        )

        self.velocity_x = 0
        self.velocity_y = 0
        self.vertical_velocity_0 = vertical_velocity_0
        self.somersault_rate_0 = self.somersaults / self.duration

        self.n_threads = n_threads
        self.ode_solver = ode_solver

        if biorbd_model_path is not None:
            self.biorbd_model = (
                BiorbdModel(biorbd_model_path),
                BiorbdModel(biorbd_model_path),
            )
            self.rigidbody_dynamics = rigidbody_dynamics
            self.dynamics_function = dynamics_function

            self.n_q = self.biorbd_model[0].nb_q
            self.n_qdot = self.biorbd_model[0].nb_qdot
            self.nb_root = self.biorbd_model[0].nb_root

            if (
                self.dynamics_function == DynamicsFcn.TORQUE_DRIVEN
                or self.dynamics_function == DynamicsFcn.JOINTS_ACCELERATION_DRIVEN
            ):
                self.n_qddot = self.biorbd_model[0].nb_qddot - self.biorbd_model[0].nb_root

            self.n_tau = self.biorbd_model[0].nb_tau - self.biorbd_model[0].nb_root

            self.tau_min, self.tau_init, self.tau_max = -100, 0, 100
            self.tau_hips_min, self.tau_hips_init, self.tau_hips_max = (
                -300,
                0,
                300,
            )  # hips and torso

            self.high_torque_idx = [
                6 - self.nb_root,
                7 - self.nb_root,
                8 - self.nb_root,
                13 - self.nb_root,
                14 - self.nb_root,
            ]
            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000

            self.velocity_max = 100  # qdot
            self.velocity_max_phase_transition = 10  # qdot hips, thorax in phase 2

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.phase_transitions = PhaseTransitionList()
            self.multinode_constraints = MultinodeConstraintList()
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()
            self.initial_states = []
            self.x_init = InitialGuessList() if initial_x is None else initial_x
            self.u_init = InitialGuessList() if initial_u is None else initial_u
            self.mapping = BiMappingList()

            self._set_boundary_conditions()

            if initial_x is None:
                self._set_initial_guesses()  # noise is into the initial guess
            if initial_u is None:
                self._set_initial_controls()  # noise is into the initial guess

            self._set_initial_momentum()
            self._set_dynamics()
            self._set_objective_functions()

            self._set_mapping()

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                self.phase_durations,
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                phase_transitions=self.phase_transitions,
                multinode_constraints=self.multinode_constraints,
                n_threads=n_threads,
                variable_mappings=self.mapping,
                control_type=ControlType.CONSTANT,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem
        """
        for phase in range(len(self.n_shooting)):
            if self.dynamics_function == DynamicsFcn.TORQUE_DRIVEN:
                self.dynamics.add(
                    self.dynamics_function,
                    with_contact=False,
                    rigidbody_dynamics=RigidBodyDynamics.ODE,
                )
            elif self.dynamics_function == DynamicsFcn.JOINTS_ACCELERATION_DRIVEN:
                self.dynamics.add(
                    self.dynamics_function,
                    rigidbody_dynamics=RigidBodyDynamics.ODE,
                )
            else:
                raise ValueError("This dynamics has not been implemented")

    def _set_objective_functions(self):
        """
        Set the multi-objective functions for each phase with specific weights
        """

        # --- Objective function --- #
        w_qdot = 1
        w_penalty = 10
        w_penalty_foot = 10
        w_penalty_hips = 100
        w_penalty_core = 10
        w_track_final = 0.1
        w_angular_momentum_x = 100000
        w_angular_momentum_yz = 1000
        for i in range(len(self.n_shooting)):
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                derivative=True,
                key="qdot",
                index=(6, 7, 8, 9, 10, 11, 12, 13, 14),
                weight=w_qdot,
                phase=i,
            )  # Regularization
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=0,
                marker_index=6,
                weight=w_penalty,
                phase=i,
                node=Node.ALL_SHOOTING,
            )  # Right hand trajectory
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_MARKERS,
                derivative=True,
                reference_jcs=0,
                marker_index=11,
                weight=w_penalty,
                phase=i,
                node=Node.ALL_SHOOTING,
            )  # Left hand trajectory
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_MARKERS,  # Lagrange
                node=Node.ALL_SHOOTING,
                derivative=True,
                reference_jcs=0,
                marker_index=16,
                weight=w_penalty_foot,
                phase=i,
                quadratic=False,
            )  # feet trajectory
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                index=(6, 7, 8, 13, 14),
                key="q",
                weight=w_penalty_core,
                phase=i,
            )  # core DoFs

            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                index=14,
                key="q",
                weight=w_penalty_hips,
                phase=i,
            )  # core DoFs

            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_ANGULAR_MOMENTUM,
                node=Node.START,
                phase=0,
                weight=w_angular_momentum_x,
                quadratic=False,
                index=0,
            )
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_ANGULAR_MOMENTUM,
                node=Node.START,
                phase=0,
                weight=w_angular_momentum_yz,
                quadratic=True,
                index=[1, 2],
            )

        # Help to stay upright at the landing.
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            index=(0, 1, 2),
            target=[0, 0, 0],
            key="q",
            weight=w_track_final,
            phase=1,
            node=Node.END,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            index=3,
            target=self.somersaults - self.thorax_hips_xyz - self.slack_final_somersault / 2,
            key="q",
            weight=w_track_final,
            phase=1,
            node=Node.END,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            index=4,
            target=0,
            key="q",
            weight=w_track_final,
            phase=1,
            node=Node.END,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_STATE,
            index=5,
            target=self.twists,
            key="q",
            weight=w_track_final,
            phase=1,
            node=Node.END,
        )

        slack_duration = 0.25
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=self.phase_durations[0] - slack_duration,
            max_bound=self.phase_durations[0] + slack_duration,
            phase=0,
            weight=1e-6,
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            min_bound=self.phase_durations[1] - slack_duration,
            max_bound=self.phase_durations[1] + slack_duration,
            phase=1,
            weight=1e-6,
        )

    def _set_initial_momentum(self):
        """
        Set initial angular momentum and linear momentum.
        """
        q_init = self.x_bounds[0].min[: self.n_q, 0]
        qdot_init = self.x_bounds[0].min[self.n_q :, 0]

        m = brd.Model(self.biorbd_model_path)
        self.sigma0 = m.angularMomentum(q_init, qdot_init, True).to_array()
        self.p0 = m.mass() * m.CoMdot(q_init, qdot_init, True).to_array()

    def _set_initial_guesses(self):
        """
        Set the initial guess for the optimal control problem (states and controls)
        """
        # --- Initial guess --- #
        total_n_shooting = np.sum(self.n_shooting) + len(self.n_shooting)
        # Initialize state vector
        self.x = np.zeros((self.n_q + self.n_qdot, total_n_shooting))

        # time vector
        data_point = np.linspace(0, self.duration, total_n_shooting)
        # parabolic trajectory on Z
        self.x[2, :] = self.jump_height + self.vertical_velocity_0 * data_point + -9.81 / 2 * data_point**2
        # Somersaults
        self.x[3, :] = np.hstack(
            (
                np.linspace(
                    0,
                    self.phase_proportions[0] * self.somersaults,
                    self.n_shooting[0] + 1,
                ),
                np.linspace(
                    self.phase_proportions[0] * self.somersaults,
                    self.somersaults,
                    self.n_shooting[1] + 1,
                ),
            )
        )
        # Twists
        self.x[5, :] = np.hstack(
            (
                np.linspace(0, self.twists, self.n_shooting[0] + 1),
                self.twists * np.ones(self.n_shooting[1] + 1),
            )
        )

        # Handle second DoF of arms with Noise.
        self.x[10, :] = np.ones((1, total_n_shooting)) * - np.pi / 2
        self.x[12, :] = np.ones((1, total_n_shooting)) * np.pi / 2

        # velocity on Y
        self.x[self.n_q + 0, :] = self.velocity_x
        self.x[self.n_q + 1, :] = self.velocity_y
        self.x[self.n_q + 2, :] = self.vertical_velocity_0 - 9.81 * data_point
        # Somersaults rate
        self.x[self.n_q + 3, :] = self.somersault_rate_0
        # Twists rate
        self.x[self.n_q + 5, :] = self.twists / self.duration

        self._set_initial_states(self.x)

    def _set_initial_states(self, X0: np.array = None):
        """
        Set the initial states of the optimal control problem.
        """
        if X0 is None:
            X0 = np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1))
            X0[2, :] = self.jump_height
            self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)
        else:
            mesh_point_init = 0
            for i in range(self.n_phases):
                self.x_init.add(
                    X0[:, mesh_point_init: mesh_point_init + self.n_shooting[i] + 1],
                    interpolation=InterpolationType.EACH_FRAME,
                )
                mesh_point_init += self.n_shooting[i]

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None and self.u is None:
            for phase in range(len(self.n_shooting)):
                n_shooting = self.n_shooting[phase]
                tau_J = np.zeros((self.n_tau, n_shooting))

                tau_max = self.tau_max * np.ones(self.n_tau)
                tau_max[self.high_torque_idx] = self.tau_hips_max

                qddot_J = np.zeros((self.n_tau, n_shooting))

                if self.dynamics_function == DynamicsFcn.TORQUE_DRIVEN:
                    self.u_init.add(tau_J, interpolation=InterpolationType.EACH_FRAME)
                elif self.dynamics_function == DynamicsFcn.JOINTS_ACCELERATION_DRIVEN:
                    self.u_init.add(qddot_J, interpolation=InterpolationType.EACH_FRAME)
                else:
                    raise ValueError("This dynamics has not been implemented")
        elif self.u is not None:
            for phase in range(len(self.n_shooting)):
                self.u_init.add(self.u[phase][:, :-1], interpolation=InterpolationType.EACH_FRAME)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)

                shooting = 0
                for i in range(len(self.n_shooting)):
                    self.u_init.add(
                        U0[:, shooting: shooting + self.n_shooting[i]],
                        interpolation=InterpolationType.EACH_FRAME,
                    )
                    shooting += self.n_shooting[i]

    def _set_boundary_conditions(self):
        """
        Set the boundary conditions for controls and states for each phase.
        """
        self.x_bounds = BoundsList()

        tilt_bound = np.pi / 4
        tilt_final_bound = np.pi / 12  # 15 degrees

        initial_arm_elevation = 2.8
        arm_rotation_z_upp = np.pi / 2
        arm_rotation_z_low = 1
        arm_elevation_y_low = 0.01
        arm_elevation_y_upp = np.pi - 0.01
        arm_end_y_low = 0.0
        arm_end_y_high = 0.48
        arm_end_z_low = 0.9
        arm_end_z_high = np.pi
        thorax_hips_xyz = np.pi / 6
        self.thorax_hips_xyz = thorax_hips_xyz
        arm_rotation_y_final = 2.4
        hips_x_low = -15 * np.pi / 180
        hips_x_high = np.pi / 2
        hips_y = 20 * np.pi / 180

        slack_initial_vertical_velocity = 2
        slack_initial_somersault_rate = 3
        slack_initial_translation_velocities = 1

        # end phase 0
        slack_somersault = 30 * 3.14 / 180
        slack_twist = 30 * 3.14 / 180

        slack_final_somersault = np.pi / 24  # 7.5 degrees
        self.slack_final_somersault = slack_final_somersault
        slack_final_twist = np.pi / 24  # 7.5 degrees
        slack_final_dofs = np.pi / 24  # 7.5 degrees

        x_min = np.zeros((2, self.n_q + self.n_qdot, 3))
        x_max = np.zeros((2, self.n_q + self.n_qdot, 3))

        x_min[0, : self.n_q, 0] = [
            0,
            0,
            self.jump_height,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -initial_arm_elevation,
            0,
            initial_arm_elevation,
            0,
            0,
        ]
        x_min[0, self.n_q :, 0] = [
            self.velocity_x - slack_initial_translation_velocities,
            self.velocity_y - slack_initial_translation_velocities,
            self.vertical_velocity_0_min,
            self.somersault_rate_0 - slack_initial_somersault_rate,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        x_max[0, : self.n_q, 0] = [
            0,
            0,
            self.jump_height,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -initial_arm_elevation,
            0,
            initial_arm_elevation,
            0,
            0,
        ]
        x_max[0, self.n_q :, 0] = [
            self.velocity_x + slack_initial_translation_velocities,
            self.velocity_y + slack_initial_translation_velocities,
            self.vertical_velocity_0_max,
            self.somersault_rate_0 + slack_initial_somersault_rate,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        x_min[0, : self.n_q, 1] = [
            -3,
            -3,
            -0.001,
            -0.001,
            -tilt_bound,
            -0.001,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
            -thorax_hips_xyz,
            -arm_rotation_z_low,
            -arm_elevation_y_upp,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            hips_x_low,
            -hips_y,
        ]
        x_min[0, self.n_q :, 1] = -self.velocity_max

        x_max[0, : self.n_q, 1] = [
            3,
            3,
            self.jump_height + 10,
            self.somersaults + slack_somersault,
            tilt_bound,
            self.twists + slack_twist,
            thorax_hips_xyz,
            thorax_hips_xyz,
            thorax_hips_xyz,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            arm_elevation_y_upp,
            hips_x_high,
            hips_y,
        ]
        x_max[0, self.n_q :, 1] = +self.velocity_max

        x_min[0, : self.n_q, 2] = [
            -3,
            -3,
            -0.001,
            self.phase_proportions[0] * self.somersaults - slack_final_somersault,
            -tilt_final_bound,
            self.twists - slack_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_rotation_z_low,
            -0.2,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            thorax_hips_xyz - slack_final_dofs,
            -slack_final_dofs,
        ]  # x_min[0, :self.n_q, 1]
        x_min[0, self.n_q :, 2] = -self.velocity_max

        x_max[0, : self.n_q, 2] = [
            3,
            3,
            10,
            self.phase_proportions[0] * self.somersaults + slack_final_somersault,
            tilt_final_bound,
            self.twists + slack_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            0.2,
            thorax_hips_xyz,
            slack_final_dofs,
        ]  # x_max[0, :self.n_q, 1]
        x_max[0, self.n_q :, 2] = +self.velocity_max

        x_min[1, : self.n_q, 0] = x_min[0, : self.n_q, 2]
        x_min[1, self.n_q :, 0] = x_min[0, self.n_q :, 2]

        x_max[1, : self.n_q, 0] = x_max[0, : self.n_q, 2]
        x_max[1, self.n_q :, 0] = x_max[0, self.n_q :, 2]

        x_min[1, : self.n_q, 1] = [
            -3,
            -3,
            -0.001,
            self.phase_proportions[0] * self.somersaults - slack_final_somersault,
            -tilt_bound,
            self.twists - slack_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_rotation_z_low,
            -arm_elevation_y_upp,
            -arm_rotation_z_upp,
            arm_elevation_y_low,
            -slack_final_dofs,
            -slack_final_dofs,
        ]  # x_min[0, :self.n_q, 1]
        x_min[1, self.n_q :, 1] = [
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
        ]

        x_max[1, : self.n_q, 1] = [
            3,
            3,
            10,
            self.somersaults + slack_somersault,
            tilt_bound,
            self.twists + slack_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            arm_rotation_z_upp,
            -arm_elevation_y_low,
            arm_rotation_z_low,
            arm_elevation_y_upp,
            thorax_hips_xyz,
            slack_final_dofs,
        ]  # x_max[0, :self.n_q, 1]
        x_max[1, self.n_q :, 1] = [
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
        ]

        x_min[1, : self.n_q, 2] = [
            -0.15,
            -0.25,
            -0.1,
            self.somersaults - thorax_hips_xyz - slack_final_somersault,
            -tilt_final_bound,
            self.twists - slack_final_twist,
            -slack_final_dofs,
            -slack_final_dofs,
            -slack_final_dofs,
            -arm_end_z_high,
            arm_end_y_low,
            arm_end_z_low,
            -arm_end_y_high,
            thorax_hips_xyz - slack_final_dofs,
            -slack_final_dofs,
        ]
        x_min[1, self.n_q :, 2] = [
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max_phase_transition,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max,
            -self.velocity_max_phase_transition,
        ]

        x_max[1, : self.n_q, 2] = [
            0.15,
            0.25,
            0.1,
            self.somersaults - thorax_hips_xyz,
            tilt_final_bound,
            self.twists + slack_final_twist,
            slack_final_dofs,
            slack_final_dofs,
            slack_final_dofs,
            -arm_end_z_low,
            arm_end_y_high,
            arm_end_z_high,
            arm_end_y_low,
            thorax_hips_xyz,
            slack_final_dofs,
        ]
        x_max[1, self.n_q :, 2] = [
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max_phase_transition,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max,
            self.velocity_max_phase_transition,
        ]

        for phase in range(len(self.n_shooting)):
            self.x_bounds.add(
                bounds=Bounds(
                    x_min[phase, :, :],
                    x_max[phase, :, :],
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )
            )

            if self.dynamics_function == DynamicsFcn.TORQUE_DRIVEN:
                self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)
                self.u_bounds[0].min[self.high_torque_idx, :] = self.tau_hips_min
                self.u_bounds[0].max[self.high_torque_idx, :] = self.tau_hips_max
            elif self.dynamics_function == DynamicsFcn.JOINTS_ACCELERATION_DRIVEN:
                self.u_bounds.add([self.qddot_min] * self.n_qddot, [self.qddot_max] * self.n_qddot)
            else:
                raise ValueError("This dynamics has not been implemented")

    def _set_mapping(self):
        """
        Set the mapping between the states and controls of the model
        """
        if self.dynamics_function == DynamicsFcn.TORQUE_DRIVEN:
            # self.mapping.add(
            #     "tau",
            #     to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            #     to_first=[6, 7, 8, 9, 10, 11, 12, 13, 14],
            # )
            bimap = SelectionMapping(
                nb_elements=bio_model.nb_dof,
                independent_indices=(6, 7, 8, 9, 10, 11, 12, 13, 14),
                dependencies=None,
            )
        elif self.dynamics_function == DynamicsFcn.JOINTS_ACCELERATION_DRIVEN:
            print("no bimapping")
        else:
            raise ValueError("This dynamics has not been implemented")
