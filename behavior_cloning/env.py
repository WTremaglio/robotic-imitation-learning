"""Simulation environment of the robot."""

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import pybullet as p
import simulation_and_control
from simulation_and_control import pb

from behavior_cloning import robot

_CONFIG_FILE = "robotnik_arm.json"
_EE_FRAME = "robot_j2s7s300_end_effector"
_SHOULDER_FRAME = "robot_j2s7s300_link_2"

_FINGER_JOINT_NAMES = frozenset(
    {
        "robot_j2s7s300_joint_finger_1",
        "robot_j2s7s300_joint_finger_tip_1",
        "robot_j2s7s300_joint_finger_2",
        "robot_j2s7s300_joint_finger_tip_2",
        "robot_j2s7s300_joint_finger_3",
        "robot_j2s7s300_joint_finger_tip_3",
    }
)
_FINGER_HOLD_FORCE = 5.0  # N

# Contact and drive settings for realistic skid-steer behavior.
_FLOOR_FRICTION = 1.0
_WHEEL_CONTACT_STIFFNESS = 30000.0
_WHEEL_CONTACT_DAMPING = 1000.0
_WHEEL_ANISOTROPIC_FRICTION = (1.0, 0.01, 1.0)
_WHEEL_KD = 30.0  # N*m*s/rad
_TARGET_MARKER_RADIUS = 0.03  # m


def quaternion2bearing(
    q_w: float, q_x: float, q_y: float, q_z: float
) -> float:
    """Return the bearing (yaw) of a quaternion orientation.

    The bearing is the angle of the body's forward axis seen from above.
    Both atan2 arguments scale equally with the quaternion's norm, so
    the input need not be normalized.

    Args:
        q_w: Scalar part of the quaternion.
        q_x: First vector component.
        q_y: Second vector component.
        q_z: Third vector component.

    Returns:
        Bearing angle in radians, in [-pi, pi].
    """
    return math.atan2(
        2.0 * (q_w * q_z + q_x * q_y),
        q_w * q_w + q_x * q_x - q_y * q_y - q_z * q_z,
    )


@dataclasses.dataclass(slots=True)
class Observation:
    """Robot state at one control step.

    Attributes:
        time: Simulation time (s).
        base_pos: Base position, world frame (m).
        base_bearing: Base bearing (rad).
        q: Motor angles in simulator motor order (rad).
        qd: Motor velocities in simulator motor order (rad/s).
        q_pin: Full configuration in Pinocchio joint order.
        ee_pos: End-effector position, world frame (m).
    """

    time: float
    base_pos: robot.FloatArray
    base_bearing: float
    q: robot.FloatArray
    qd: robot.FloatArray
    q_pin: robot.FloatArray
    ee_pos: robot.FloatArray


class Environment:
    """Simulation of the robot, built once and reset for each episode.

    Wheels track velocity commands through a velocity-feedback torque,
    arm joints take torque commands and the fingers are position-held.
    """

    def __init__(
        self,
        config_dir: str,
        use_gui: bool = False,
        fixed_finger_angles: npt.ArrayLike | None = None,
    ) -> None:
        """Build the simulation.

        Args:
            config_dir: Folder holding the configs and models folders.
            use_gui: Whether to open the PyBullet GUI.
            fixed_finger_angles: Six finger angles (rad), or None to use
                the initial configuration.

        Raises:
            ValueError: If the robot lacks the six finger joints, or
                fixed_finger_angles does not hold six angles.
        """
        sim = pb.SimInterface(
            _CONFIG_FILE, conf_file_path_ext=config_dir, use_gui=use_gui
        )
        sim.SetFloorFriction(_FLOOR_FRICTION)
        client = sim.pybullet_client
        bot = sim.bot[0]
        body = bot.bot_pybullet
        joint_names = [
            client.getJointInfo(body, j)[1].decode()
            for j in range(client.getNumJoints(body))
        ]
        for j, name in enumerate(joint_names):
            if "wheel" in name:
                client.changeDynamics(
                    body,
                    j,
                    contactStiffness=_WHEEL_CONTACT_STIFFNESS,
                    contactDamping=_WHEEL_CONTACT_DAMPING,
                    anisotropicFriction=_WHEEL_ANISOTROPIC_FRICTION,
                )

        finger_joint_ids = [
            j
            for j, name in enumerate(joint_names)
            if name in _FINGER_JOINT_NAMES
        ]

        if len(finger_joint_ids) != len(_FINGER_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(_FINGER_JOINT_NAMES)} finger joints, found "
                f"{len(finger_joint_ids)}"
            )

        if fixed_finger_angles is None:
            finger_angles = np.array(sim.GetInitMotorAngles())[
                robot.FINGER_MOTORS
            ]
        else:
            finger_angles = np.array(fixed_finger_angles, dtype=np.float64)
            if finger_angles.shape != (len(finger_joint_ids),):
                raise ValueError(
                    f"Expected {len(finger_joint_ids)} finger angles, got "
                    f"shape {finger_angles.shape}"
                )

        for jid, angle in zip(
            finger_joint_ids, finger_angles.tolist(), strict=True
        ):
            client.resetJointState(
                body, jid, targetValue=angle, targetVelocity=0.0
            )
        finger_angles.flags.writeable = False

        self._marker = client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=client.createVisualShape(
                p.GEOM_SPHERE,
                radius=_TARGET_MARKER_RADIUS,
                rgbaColor=[1.0, 0.0, 0.0, 0.85],
            ),
        )

        ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
        self.dyn_model = simulation_and_control.PinWrapper(
            _CONFIG_FILE,
            "pybullet",
            ext_names,
            ["pybullet"],
            False,
            0,
            config_dir,
        )
        self.ee_frame_id = self.dyn_model.pin_model.getFrameId(_EE_FRAME)
        self.shoulder_frame_id = self.dyn_model.pin_model.getFrameId(
            _SHOULDER_FRAME
        )
        lower_limits, upper_limits = sim.GetBotJointsLimit()
        self.arm_lower = np.array(lower_limits[robot.ARM_MOTORS])
        self.arm_upper = np.array(upper_limits[robot.ARM_MOTORS])
        self.finger_angles = finger_angles
        self.dt: float = sim.GetTimeStep()

        self._client = client
        self._body = body
        self._joint_ids = bot.active_joint_ids
        self._torque_ids = bot.active_joint_ids[robot.TORQUE_MOTORS]
        self._finger_ids = bot.active_joint_ids[robot.FINGER_MOTORS]
        self._finger_targets = finger_angles.tolist()
        self._finger_forces = [_FINGER_HOLD_FORCE] * len(finger_joint_ids)
        # Bound once: each client attribute access builds a new wrapper.
        self._get_joint_states = client.getJointStates
        self._get_base_pose = client.getBasePositionAndOrientation
        self._set_motors = client.setJointMotorControlArray
        self._step_simulation = client.stepSimulation
        self._initial_state = client.saveState()
        self._stepped = False
        self._time = 0.0
        self._wheel_speeds = [0.0] * robot.NUM_WHEELS

    def reset(
        self,
        target_pos: npt.ArrayLike,
        base_xy: tuple[float, float] | None = None,
        base_bearing: float | None = None,
    ) -> Observation:
        """Start a new episode.

        Args:
            target_pos: Target end-effector position, world frame (m).
            base_xy: Start position of the base (m), or None to keep the
                initial one.
            base_bearing: Start bearing of the base (rad), or None to
                keep the initial one.

        Returns:
            The first observation of the episode.
        """
        client = self._client
        if self._stepped:
            client.restoreState(self._initial_state)
            self._stepped = False
        if base_xy is not None or base_bearing is not None:
            pos, orn = client.getBasePositionAndOrientation(self._body)
            if base_xy is not None:
                pos = (base_xy[0], base_xy[1], pos[2])
            if base_bearing is not None:
                orn = client.getQuaternionFromEuler((0.0, 0.0, base_bearing))
            client.resetBasePositionAndOrientation(self._body, pos, orn)
        client.resetBasePositionAndOrientation(
            self._marker, np.asarray(target_pos).tolist(), (0, 0, 0, 1)
        )
        self._time = 0.0
        return self._observe()

    def step(
        self, wheel_cmds: robot.FloatArray, torque_arm: robot.FloatArray
    ) -> Observation:
        """Apply one step of commands and advance the simulation.

        Args:
            wheel_cmds: Wheel velocity commands (rad/s).
            torque_arm: Arm joint torques (N*m).

        Returns:
            The observation after the step.
        """
        # + 0.0 maps -0.0 to 0.0, matching the course's servo model.
        forces = [
            _WHEEL_KD * (command - speed) + 0.0
            for command, speed in zip(
                wheel_cmds.tolist(), self._wheel_speeds, strict=True
            )
        ]
        forces += [torque + 0.0 for torque in torque_arm.tolist()]
        self._set_motors(
            bodyIndex=self._body,
            jointIndices=self._torque_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=forces,
        )
        self._set_motors(
            bodyIndex=self._body,
            jointIndices=self._finger_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self._finger_targets,
            forces=self._finger_forces,
        )
        self._step_simulation()
        self._stepped = True
        self._time += self.dt
        return self._observe()

    def push(self, velocity_xy: tuple[float, float]) -> None:
        """Set the base's horizontal velocity, as a push would.

        Args:
            velocity_xy: New base velocity, world frame (m/s).
        """
        _, angular = self._client.getBaseVelocity(self._body)
        self._client.resetBaseVelocity(
            self._body, (*velocity_xy, 0.0), angular
        )

    def close(self) -> None:
        """Disconnect from the simulation."""
        self._client.disconnect()

    def key_pressed(self, key: int) -> bool:
        """Return whether a key was just pressed in the GUI window.

        Args:
            key: Key code, e.g. ord("q").

        Returns:
            Whether the key was pressed since the last check.
        """
        return bool(
            self._client.getKeyboardEvents().get(key, 0) & p.KEY_WAS_TRIGGERED
        )

    def _observe(self) -> Observation:
        """Read the robot state and update the end-effector kinematics.

        Returns:
            The current observation.
        """
        states = self._get_joint_states(self._body, self._joint_ids)
        q = np.array([state[0] for state in states])
        qd = np.array([state[1] for state in states])
        self._wheel_speeds = qd[robot.WHEEL_MOTORS].tolist()
        pos, orn = self._get_base_pose(self._body)
        model = self.dyn_model.pin_model
        data = self.dyn_model.pin_data
        q_pin = self.dyn_model.ReoderJoints2PinVec(
            np.concatenate((pos, orn, q)), "pos"
        )
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        return Observation(
            self._time,
            np.array(pos),
            quaternion2bearing(orn[3], orn[0], orn[1], orn[2]),
            q,
            qd,
            q_pin,
            data.oMf[self.ee_frame_id].translation.copy(),  # pyright: ignore[reportOptionalSubscript]
        )
