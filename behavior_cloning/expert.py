"""Expert policy: the base and arm control laws over one episode."""

import dataclasses
import typing

import numpy as np
import numpy.typing as npt
import simulation_and_control

from behavior_cloning import control, env, robot


def _read_only(values: npt.ArrayLike) -> robot.FloatArray:
    """Return a read-only float64 copy of values."""
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


@dataclasses.dataclass(frozen=True)
class PlannerParams:
    """Parameters for choosing where the base parks.

    The base parks facing the target, at a horizontal standoff distance
    interpolated from the target height: larger for low targets, and
    smaller for targets high above the shoulder (about 0.49 m up).

    Attributes:
        standoff_heights: Target heights of the interpolation table,
            in increasing order (m).
        standoff_distances: Standoff distance at each height (m).
    """

    standoff_heights: tuple[float, ...] = (0.1, 0.5, 1.1, 1.25)
    standoff_distances: tuple[float, ...] = (0.80, 1.00, 0.75, 0.35)


def standoff_distance(
    target_heights: npt.NDArray[np.float64], params: PlannerParams
) -> npt.NDArray[np.float64]:
    """Return how far from each target the base parks, horizontally.

    Args:
        target_heights: Target heights (m).
        params: Planner parameters.

    Returns:
        The standoff distance for each height (m).
    """
    return np.interp(
        target_heights, params.standoff_heights, params.standoff_distances
    )


@dataclasses.dataclass(slots=True)
class ExpertAction:
    """Commands and internal quantities of the expert at one step.

    Attributes:
        wheel_cmds: Wheel velocity commands (rad/s).
        torque_arm: Arm joint torques (N*m).
        v_cmd: Commanded base forward speed (m/s).
        w_cmd: Commanded base turning rate (rad/s).
        q_des_arm: Arm joint reference (rad).
        qd_des_arm: Arm joint velocity reference (rad/s).
        gravity_arm: Gravity compensation in torque_arm before
            clipping (N*m).
        phase: Base navigation phase after this step.
        base_done: Whether base navigation has finished.
    """

    wheel_cmds: robot.FloatArray
    torque_arm: robot.FloatArray
    v_cmd: float
    w_cmd: float
    q_des_arm: robot.FloatArray
    qd_des_arm: robot.FloatArray
    gravity_arm: robot.FloatArray
    phase: control.NavPhase
    base_done: bool


class ExpertPolicy:
    """Expert reaching controller with its memory held between steps.

    The base drives to the standoff distance from the target, facing it,
    then the arm reaches for it. The navigation phase and the arm joint
    reference persist from one call of act to the next; reset starts a
    new episode.

    Forward kinematics for each observation's q_pin must be up to date
    in the Pinocchio data before act is called.
    """

    def __init__(
        self,
        dyn_model: simulation_and_control.PinWrapper,
        ee_frame_id: int,
        shoulder_frame_id: int,
        lower_bound: robot.FloatArray,
        upper_bound: robot.FloatArray,
        dt: float,
        base_params: control.BaseControllerParams | None = None,
        arm_params: control.ArmControllerParams | None = None,
        planner_params: PlannerParams | None = None,
    ) -> None:
        """Initialize the expert.

        Args:
            dyn_model: Pinocchio model and data of the whole robot.
            ee_frame_id: Pinocchio frame ID of the end effector.
            shoulder_frame_id: Pinocchio frame ID of the shoulder.
            lower_bound: Lower joint reference bound from
                reference_bounds (rad).
            upper_bound: Upper joint reference bound from
                reference_bounds (rad).
            dt: Control time step (s).
            base_params: Base controller parameters; defaults if None.
            arm_params: Arm controller parameters; defaults if None.
            planner_params: Planner parameters; defaults if None.
        """
        self._dyn_model = dyn_model
        self._ee_frame_id = ee_frame_id
        self._shoulder_frame_id = shoulder_frame_id
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._dt = dt
        self._base_params = base_params or control.BaseControllerParams()
        self._arm_params = arm_params or control.ArmControllerParams()
        self._planner_params = planner_params or PlannerParams()
        self._target_pos = np.zeros(3)
        self._target_x = 0.0
        self._target_y = 0.0
        self._standoff = 0.0
        self._nav_state = control.BaseNavState()
        self._q_hold_arm = robot.ARM_ZEROS
        self._q_des_arm = self._q_hold_arm.copy()

    @property
    def base_params(self) -> control.BaseControllerParams:
        """Base controller parameters."""
        return self._base_params

    @property
    def arm_params(self) -> control.ArmControllerParams:
        """Arm controller parameters."""
        return self._arm_params

    @classmethod
    def from_environment(
        cls,
        environment: env.Environment,
        base_params: control.BaseControllerParams | None = None,
        arm_params: control.ArmControllerParams | None = None,
        planner_params: PlannerParams | None = None,
    ) -> typing.Self:
        """Create an expert for the robot in an environment.

        Args:
            environment: Simulation whose robot the expert controls.
            base_params: Base controller parameters; defaults if None.
            arm_params: Arm controller parameters; defaults if None.
            planner_params: Planner parameters; defaults if None.

        Returns:
            The expert.
        """
        arm_params = arm_params or control.ArmControllerParams()
        lower_bound, upper_bound = control.reference_bounds(
            environment.arm_lower,
            environment.arm_upper,
            arm_params.limit_margin,
        )
        return cls(
            environment.dyn_model,
            environment.ee_frame_id,
            environment.shoulder_frame_id,
            lower_bound,
            upper_bound,
            environment.dt,
            base_params,
            arm_params,
            planner_params,
        )

    def reset(self, target_pos: npt.ArrayLike, obs: env.Observation) -> None:
        """Start a new episode.

        Args:
            target_pos: Target end-effector position, world frame (m).
            obs: First observation of the episode. The arm holds its
                joint angles while the base moves.

        Raises:
            ValueError: If target_pos does not hold three coordinates.
        """
        self._target_pos = _read_only(target_pos)
        if self._target_pos.shape != (3,):
            raise ValueError(
                "Expected target_pos of shape (3,), got "
                f"{self._target_pos.shape}"
            )
        self._q_hold_arm = _read_only(obs.q[robot.ARM_MOTORS])
        self._target_x = float(self._target_pos[0])
        self._target_y = float(self._target_pos[1])
        self._standoff = float(
            standoff_distance(self._target_pos[2:], self._planner_params)[0]
        )
        self._nav_state = control.BaseNavState()
        self._q_des_arm = self._q_hold_arm.copy()

    def act(self, obs: env.Observation) -> ExpertAction:
        """Compute the expert's commands for one step.

        Args:
            obs: Current robot state.

        Returns:
            The commands and internal quantities for this step.
        """
        nav_state = self._nav_state
        was_done = nav_state.done
        v_cmd, w_cmd = control.base_controller(
            nav_state,
            obs.base_pos,
            obs.base_bearing,
            self._target_x,
            self._target_y,
            self._standoff,
            obs.time,
            self._base_params,
        )
        wheel_cmds = (
            robot.WHEEL_ZEROS
            if was_done
            else control.wheel_velocities(v_cmd, w_cmd, self._base_params)
        )
        torque_arm, self._q_des_arm, qd_des_arm, gravity_arm = (
            control.arm_controller(
                self._dyn_model,
                obs.q_pin,
                obs.q[robot.ARM_MOTORS],
                obs.qd[robot.ARM_MOTORS],
                self._q_des_arm,
                self._q_hold_arm,
                self._target_pos,
                obs.ee_pos,
                self._ee_frame_id,
                self._shoulder_frame_id,
                self._lower_bound,
                self._upper_bound,
                self._dt,
                self._arm_params,
                reach_active=nav_state.done,
            )
        )
        return ExpertAction(
            wheel_cmds,
            torque_arm,
            v_cmd,
            w_cmd,
            self._q_des_arm,
            qd_des_arm,
            gravity_arm,
            nav_state.phase,
            nav_state.done,
        )
