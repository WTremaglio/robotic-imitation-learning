"""Control laws of the expert: arm reaching and base navigation."""

import dataclasses
import math
import typing

import numpy as np
import pinocchio as pin
import simulation_and_control

from behavior_cloning import robot

NavPhase = typing.Literal["align", "drive", "orient"]

# Read-only identity matrix, shared across steps.
_EYE3 = np.eye(3)
_EYE3.flags.writeable = False


def clip(
    values: robot.FloatArray,
    lower: robot.FloatArray | float,
    upper: robot.FloatArray | float,
) -> robot.FloatArray:
    """Clip values to [lower, upper] elementwise.

    Gives the same result as np.clip, at about half the cost on small
    arrays.

    Args:
        values: Array to clip.
        lower: Lower bound, scalar or broadcastable array.
        upper: Upper bound, scalar or broadcastable array.

    Returns:
        The clipped array.
    """
    return np.minimum(np.maximum(values, lower), upper)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi) (rad)."""
    return (angle + math.pi) % math.tau - math.pi


def _damped_least_squares(
    jacobian: robot.FloatArray,
    damping_term: robot.FloatArray,
    velocity: robot.FloatArray,
) -> robot.FloatArray:
    """Return the damped least-squares joint velocities for a task.

    Args:
        jacobian: Task Jacobian of the joints being solved for.
        damping_term: Squared damping times the identity matrix.
        velocity: Desired task velocity.

    Returns:
        Joint velocities J^T (J J^T + damping_term)^-1 velocity.
    """
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping_term, velocity
    )


def reference_bounds(
    arm_lower: robot.FloatArray,
    arm_upper: robot.FloatArray,
    limit_margin: float,
) -> tuple[robot.FloatArray, robot.FloatArray]:
    """Return the bounds that the arm joint reference is kept within.

    Each limited joint's range is shrunk by limit_margin at both ends.
    Joints whose upper limit is not above the lower one are treated as
    unlimited and get infinite bounds. Computed once per expert for
    arm_controller.

    Args:
        arm_lower: Lower arm joint limits (rad).
        arm_upper: Upper arm joint limits (rad).
        limit_margin: Distance kept from each limit (rad).

    Returns:
        A tuple (lower_bound, upper_bound) of read-only arrays (rad).
    """
    has_limits = arm_lower < arm_upper
    lower_bound = np.where(has_limits, arm_lower + limit_margin, -np.inf)
    upper_bound = np.where(has_limits, arm_upper - limit_margin, np.inf)
    lower_bound.flags.writeable = False
    upper_bound.flags.writeable = False
    return lower_bound, upper_bound


@dataclasses.dataclass(frozen=True)
class ArmControllerParams:
    """Tuning parameters of the arm reaching controller.

    Attributes:
        reach_limit: Maximum end-effector distance from the shoulder
            that the arm is asked to reach (m). Targets further away are
            projected onto this sphere, so the arm settles at the
            closest pose instead of fully stretching into a singular
            pose.
        kp: Joint PD position gain.
        kd: Joint PD velocity gain.
        max_joint_vel: Joint speed limit (rad/s).
        max_q_lead: Maximum distance of the joint reference from the
            measured angle (rad).
        max_torque: Joint torque limit (N*m).
        cartesian_kp: Gain of the Cartesian velocity command.
        max_cartesian_speed: End-effector speed limit (m/s).
        min_ik_damping: IK damping far from singular poses.
        max_ik_damping: IK damping at a singular pose.
        singularity_threshold: Smallest singular value below which
            the damping increases.
        limit_margin: Joints this close to a limit are treated as
            saturated (rad); applied by reference_bounds.
    """

    reach_limit: float = 0.85
    kp: float = 100.0
    kd: float = 30.0
    max_joint_vel: float = 1.0
    max_q_lead: float = 0.15
    max_torque: float = 30.0
    cartesian_kp: float = 1.5
    max_cartesian_speed: float = 0.25
    min_ik_damping: float = 0.01
    max_ik_damping: float = 0.10
    singularity_threshold: float = 0.15
    limit_margin: float = 0.05


def arm_controller(
    dyn_model: simulation_and_control.PinWrapper,
    q_pin: robot.FloatArray,
    q_arm: robot.FloatArray,
    qd_arm: robot.FloatArray,
    q_des_arm: robot.FloatArray,
    q_hold_arm: robot.FloatArray,
    target_pos: robot.FloatArray,
    ee_pos: robot.FloatArray,
    ee_frame_id: int,
    shoulder_frame_id: int,
    lower_bound: robot.FloatArray,
    upper_bound: robot.FloatArray,
    dt: float,
    params: ArmControllerParams,
    reach_active: bool,
) -> tuple[
    robot.FloatArray, robot.FloatArray, robot.FloatArray, robot.FloatArray
]:
    """Compute arm joint torques that reach for a target.

    Differential IK (singularity-damped and joint-limit aware) turns a
    Cartesian velocity command into a joint reference, which a
    gravity-compensated joint PD law tracks. Forward kinematics for
    q_pin must already be up to date in dyn_model.pin_data.

    Args:
        dyn_model: Pinocchio model and data of the whole robot.
        q_pin: Full configuration in Pinocchio joint order.
        q_arm: Measured arm joint angles (rad).
        qd_arm: Measured arm joint velocities (rad/s).
        q_des_arm: Joint reference from the previous step (rad).
        q_hold_arm: Joint angles to hold while reach_active is False.
            Returned without copying; must not be modified in place.
        target_pos: Target end-effector position, world frame (m).
        ee_pos: Current end-effector position, world frame (m).
        ee_frame_id: Pinocchio frame ID of the end effector.
        shoulder_frame_id: Pinocchio frame ID of the shoulder.
        lower_bound: Lower joint reference bound from
            reference_bounds (rad).
        upper_bound: Upper joint reference bound from
            reference_bounds (rad).
        dt: Control time step (s).
        params: Controller tuning parameters.
        reach_active: Whether to reach; if False, hold q_hold_arm.

    Returns:
        A tuple (torque_arm, q_des_arm, qd_des_arm, gravity_arm) of
        joint torques (N*m), the updated joint reference (rad), the
        joint velocity reference (rad/s) and the gravity compensation
        included in the torques before clipping (N*m).
    """
    model = dyn_model.pin_model
    data = dyn_model.pin_data
    if not reach_active:
        q_des_arm = q_hold_arm
        qd_des_arm = robot.ARM_ZEROS
    else:
        # End-effector Jacobian in the world-aligned frame; column-major
        # like the masked copies taken in the re-solve below.
        frame_jacobian = pin.computeFrameJacobian(
            model, data, q_pin, ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        arm_jacobian = frame_jacobian[:3, robot.ARM_DOFS].copy(order="F")

        # Increase damping near singular poses to keep joint velocities
        # bounded: linear from max_ik_damping at a singular value of 0
        # to min_ik_damping at singularity_threshold and above.
        jacobian_sq = arm_jacobian @ arm_jacobian.T
        sigma_min = math.sqrt(max(np.linalg.eigvalsh(jacobian_sq)[0], 0.0))
        if sigma_min >= params.singularity_threshold:
            damping = params.min_ik_damping
        else:
            damping_slope = (
                params.min_ik_damping - params.max_ik_damping
            ) / params.singularity_threshold
            damping = damping_slope * sigma_min + params.max_ik_damping
        damping_term = damping**2 * _EYE3

        # Proportional Cartesian velocity command toward the
        # (reach-limited) target.
        shoulder_pos = data.oMf[shoulder_frame_id].translation  # pyright: ignore[reportOptionalSubscript]
        offset = target_pos - shoulder_pos
        offset_norm = math.sqrt(offset.dot(offset))
        if offset_norm > params.reach_limit:
            offset *= params.reach_limit / offset_norm
        ik_error = shoulder_pos + offset - ee_pos
        v_cartesian = params.cartesian_kp * ik_error
        cartesian_speed = math.sqrt(v_cartesian.dot(v_cartesian))
        if cartesian_speed > params.max_cartesian_speed:
            v_cartesian *= params.max_cartesian_speed / cartesian_speed

        qd_des_arm = arm_jacobian.T @ np.linalg.solve(
            jacobian_sq + damping_term, v_cartesian
        )

        # Saturation-aware solve: joints at a limit that the motion
        # would push further into are frozen, and their share of the
        # motion is redistributed to the remaining joints.
        at_lower = q_arm <= lower_bound
        at_upper = q_arm >= upper_bound
        if (at_lower | at_upper).any():
            free = np.ones(robot.NUM_ARM_JOINTS, dtype=bool)
            for _ in range(robot.NUM_ARM_JOINTS):
                newly_blocked = (
                    (at_lower & (qd_des_arm < 0))
                    | (at_upper & (qd_des_arm > 0))
                ) & free
                if not newly_blocked.any():
                    break
                free &= ~newly_blocked
                qd_des_arm = np.zeros(robot.NUM_ARM_JOINTS)
                qd_des_arm[free] = _damped_least_squares(
                    arm_jacobian[:, free], damping_term, v_cartesian
                )

        max_joint_speed = np.abs(qd_des_arm).max()
        if max_joint_speed > params.max_joint_vel:
            qd_des_arm *= params.max_joint_vel / max_joint_speed
        q_des_arm = q_des_arm + qd_des_arm * dt
        # Anti-windup: for unreachable targets the reference must not
        # run away from the measured state.
        q_des_arm = clip(
            q_des_arm, q_arm - params.max_q_lead, q_arm + params.max_q_lead
        )
        # Keep the reference inside the limits so PD overshoot does not
        # hit the hard stop.
        q_des_arm = clip(q_des_arm, lower_bound, upper_bound)
    gravity_arm = pin.computeGeneralizedGravity(model, data, q_pin)[
        robot.ARM_DOFS
    ]
    torque_arm = joint_torque(
        params, q_des_arm, qd_des_arm, q_arm, qd_arm, gravity_arm
    )
    return torque_arm, q_des_arm, qd_des_arm, gravity_arm


def joint_torque(
    params: ArmControllerParams,
    q_des_arm: robot.FloatArray,
    qd_des_arm: robot.FloatArray,
    q_arm: robot.FloatArray,
    qd_arm: robot.FloatArray,
    gravity_arm: robot.FloatArray,
) -> robot.FloatArray:
    """Return the gravity-compensated joint PD torques of the arm.

    This is the torque law of arm_controller, shared by every policy.

    Args:
        params: Arm controller parameters, for the gains and limit.
        q_des_arm: Arm joint reference (rad).
        qd_des_arm: Arm joint velocity reference (rad/s).
        q_arm: Measured arm joint angles (rad).
        qd_arm: Measured arm joint velocities (rad/s).
        gravity_arm: Gravity compensation torques (N*m).

    Returns:
        Arm joint torques (N*m).
    """
    torque_arm = params.kp * (q_des_arm - q_arm) + params.kd * (
        qd_des_arm - qd_arm
    )
    return clip(
        torque_arm + gravity_arm, -params.max_torque, params.max_torque
    )


@dataclasses.dataclass(frozen=True)
class BaseControllerParams:
    """Tuning parameters of the mobile-base navigation controller.

    Attributes:
        nav_timeout: Maximum time for the base to park before the arm
            starts reaching (s).
        kp_dist: Gain from standoff distance error to forward speed.
        kp_heading: Gain from heading error to turning rate.
        max_lin_vel: Linear speed limit (m/s).
        min_lin_vel: Smallest driving speed outside pos_tol, which
            avoids stalling against wheel friction (m/s).
        max_ang_vel: Turning speed limit (rad/s).
        drive_angle: Heading error at which the forward speed reaches
            zero (rad); below it the speed shrinks in proportion, so the
            base turns and drives at once.
        pos_tol: Standoff distance error that counts as arrived (m).
        ang_tol: Heading error that counts as facing the target (rad).
        wheel_radius: Wheel radius (m).
        wheel_base_width: Effective skid-steer track width (m),
            calibrated in simulation to a yaw rate of about 0.34 rad/s
            per rad/s of wheel speed.
    """

    nav_timeout: float = 25.0
    kp_dist: float = 1.0
    kp_heading: float = 3.0
    max_lin_vel: float = 0.25
    min_lin_vel: float = 0.05
    max_ang_vel: float = 1.0
    drive_angle: float = 0.5
    pos_tol: float = 0.03
    ang_tol: float = 0.05
    wheel_radius: float = 0.11
    wheel_base_width: float = 0.66


@dataclasses.dataclass
class BaseNavState:
    """Running state of the base navigation controller.

    The phase only describes the current step: "align" while turning
    on the spot, "drive" while moving, and "orient" while turning within
    pos_tol of the standoff distance. Only done is remembered.

    Attributes:
        phase: Phase of the latest step.
        done: Whether navigation has finished.
    """

    phase: NavPhase = "align"
    done: bool = False


def base_controller(
    state: BaseNavState,
    base_pos: robot.FloatArray,
    base_bearing: float,
    target_x: float,
    target_y: float,
    standoff: float,
    current_time: float,
    params: BaseControllerParams,
) -> tuple[float, float]:
    """Drive the base to the standoff distance, facing the target.

    Every command follows from the current pose alone: the base turns
    toward the target and moves along its heading to close the gap to
    the standoff distance, backing up if it is too close. Once within
    pos_tol and ang_tol it parks for good.

    Args:
        state: Navigation state, updated in place.
        base_pos: Measured base position, world frame (m).
        base_bearing: Measured base bearing (rad).
        target_x: Target x, world frame (m).
        target_y: Target y, world frame (m).
        standoff: Horizontal distance to park from the target (m).
        current_time: Simulation time (s).
        params: Controller tuning parameters.

    Returns:
        A tuple (v_cmd, w_cmd) of the commanded forward speed (m/s) and
        turning rate (rad/s).
    """
    if state.done:
        return 0.0, 0.0

    dx = target_x - base_pos[0]
    dy = target_y - base_pos[1]
    dist_error = math.hypot(dx, dy) - standoff
    heading_err = _wrap_angle(math.atan2(dy, dx) - base_bearing)
    w_cmd = min(
        max(params.kp_heading * heading_err, -params.max_ang_vel),
        params.max_ang_vel,
    )
    if abs(dist_error) < params.pos_tol:
        state.phase = "orient"
        v_cmd = 0.0
        if abs(heading_err) < params.ang_tol:
            state.done = True
            w_cmd = 0.0
    else:
        speed = min(
            max(params.kp_dist * abs(dist_error), params.min_lin_vel),
            params.max_lin_vel,
        )
        ramp = max(0.0, 1.0 - abs(heading_err) / params.drive_angle)
        v_cmd = math.copysign(speed * ramp, dist_error)
        state.phase = "drive" if ramp > 0.0 else "align"

    if current_time > params.nav_timeout:
        state.done = True
    return v_cmd, w_cmd


def wheel_velocities(
    v_cmd: float, w_cmd: float, params: BaseControllerParams
) -> robot.FloatArray:
    """Convert a forward speed and turning rate into wheel speeds.

    Args:
        v_cmd: Forward speed (m/s).
        w_cmd: Turning rate (rad/s).
        params: Base controller parameters, for the wheel geometry.

    Returns:
        Wheel velocity commands in the order front right, front left,
        back left, back right (rad/s).
    """
    turn = w_cmd * params.wheel_base_width / 2.0
    lw = (v_cmd - turn) / params.wheel_radius
    rw = (v_cmd + turn) / params.wheel_radius
    return np.array((rw, lw, lw, rw))
