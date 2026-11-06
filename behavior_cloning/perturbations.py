"""Adversities for robustness evaluations and push training.

A perturbed policy sees noisy measurements and sends noisy commands,
while the simulation and the stop checks keep using the true state. A
push sets the base sliding sideways once, partway through the drive;
training episodes draw their pushes at random.
"""

import collections.abc
import dataclasses
import math

import numpy as np

from behavior_cloning import control, env, episode, robot

# DART noise of the training data, the unit of command noise.
_DART_V = 0.0156  # m/s
_DART_W = 0.137  # rad/s
_DART_Q = 0.004  # rad
# Second seed word, so perturbations draw from their own stream.
_PERTURB_STREAM = 2
# Third seed word, so training pushes draw from their own stream.
_PUSH_STREAM = 3

type StepCallback[ActionT: episode.Action] = collections.abc.Callable[
    [env.Observation, ActionT, env.Observation], bool
]


@dataclasses.dataclass(frozen=True)
class Perturbation:
    """Adversities applied during an evaluation episode.

    Attributes:
        obs_noise: Standard deviation of the noise on every measured
            position (m) and angle (rad) the policy sees, drawn afresh
            at each step.
        cmd_noise: Noise on the executed commands, as a multiple of the
            training data's DART noise, held for one policy period.
        push_speed: Sideways speed the base is set to once (m/s).
        push_time: Time of the push (s).
    """

    obs_noise: float = 0.0
    cmd_noise: float = 0.0
    push_speed: float = 0.0
    push_time: float = 2.0

    @property
    def perturbs_policy(self) -> bool:
        """Whether the policy's measurements or commands are noisy."""
        return self.obs_noise > 0.0 or self.cmd_noise > 0.0


class PerturbedPolicy[ActionT: episode.Action]:
    """Policy wrapper with noisy measurements and noisy commands."""

    def __init__(
        self,
        policy: episode.Policy[ActionT],
        perturbation: Perturbation,
        seed: int,
    ) -> None:
        """Wrap a policy for one episode.

        Args:
            policy: Policy to perturb.
            perturbation: Noise levels; push_callback applies the push.
            seed: Episode seed; the same seed gives the same noise.
        """
        self._policy = policy
        self._obs_noise = perturbation.obs_noise
        scale = perturbation.cmd_noise
        self._cmd_noise = scale > 0.0
        self._cmd_std = (scale * _DART_V, scale * _DART_W, scale * _DART_Q)
        self._base_params = control.BaseControllerParams()
        self._arm_params = control.ArmControllerParams()
        self._rng = np.random.default_rng((seed, _PERTURB_STREAM))
        self._countdown = 1
        self._wheel_noise = robot.WHEEL_ZEROS
        self._torque_noise = robot.ARM_ZEROS

    def reset(
        self, target_pos: robot.FloatArray, obs: env.Observation
    ) -> None:
        """Start a new episode from a noisy first measurement.

        Args:
            target_pos: Target end-effector position, world frame (m).
            obs: First observation of the episode.
        """
        self._countdown = 1
        self._policy.reset(target_pos, self._measure(obs))

    def act(self, obs: env.Observation) -> ActionT:
        """Act on a noisy measurement, then add noise to the commands.

        Command noise is drawn every policy period and held between,
        as in DART. It acts on the forward speed, turning rate, and arm
        joint references, through the same wheel and PD laws the
        policies use, and leaves a parked base's wheels alone.

        Args:
            obs: Current robot state.

        Returns:
            The policy's action, with its executed commands perturbed.
        """
        action = self._policy.act(self._measure(obs))
        if not self._cmd_noise:
            return action
        countdown = self._countdown - 1
        if countdown:
            self._countdown = countdown
        else:
            self._countdown = robot.POLICY_PERIOD
            v_std, w_std, q_std = self._cmd_std
            normal = self._rng.normal
            self._wheel_noise = control.wheel_velocities(
                normal(0.0, v_std), normal(0.0, w_std), self._base_params
            )
            zeros = robot.ARM_ZEROS
            self._torque_noise = control.joint_torque(
                self._arm_params,
                normal(0.0, q_std, robot.NUM_ARM_JOINTS),
                zeros,
                zeros,
                zeros,
                zeros,
            )
        if not action.base_done:
            action.wheel_cmds = action.wheel_cmds + self._wheel_noise
        action.torque_arm = action.torque_arm + self._torque_noise
        return action

    def _measure(self, obs: env.Observation) -> env.Observation:
        """Return the observation as a noisy sensor would report it."""
        std = self._obs_noise
        if not std:
            return obs
        normal = self._rng.normal
        q = obs.q.copy()
        q[robot.ARM_MOTORS] += normal(0.0, std, robot.NUM_ARM_JOINTS)
        return dataclasses.replace(
            obs,
            base_pos=obs.base_pos + normal(0.0, std, 3),
            base_bearing=obs.base_bearing + normal(0.0, std),
            q=q,
            ee_pos=obs.ee_pos + normal(0.0, std, 3),
        )


def sample_push(
    seed: int,
    speed: tuple[float, float],
    push_time: tuple[float, float],
) -> tuple[float, float]:
    """Draw the push of one training episode.

    Args:
        seed: Episode seed; the same seed gives the same push.
        speed: Range of the push speed (m/s).
        push_time: Range of the push time (s).

    Returns:
        A tuple (speed, time) of the sideways speed, positive to the
        base's left and equally likely either way (m/s), and the time
        of the push (s).
    """
    rng = np.random.default_rng((seed, _PUSH_STREAM))
    magnitude = rng.uniform(*speed)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return float(sign * magnitude), float(rng.uniform(*push_time))


def push_callback[ActionT: episode.Action](
    environment: env.Environment,
    speed: float,
    push_time: float,
    then: StepCallback[ActionT],
) -> StepCallback[ActionT]:
    """Return a step callback that pushes the base sideways once.

    The push sets the base sliding perpendicular to its heading at the
    first step that ends at or after the push time. Every step is then
    passed on.

    Args:
        environment: Simulation the episode runs in.
        speed: Sideways speed, positive to the base's left (m/s).
        push_time: Time of the push (s).
        then: Step callback to pass every step on to.

    Returns:
        The step callback; it ends the episode when then does.
    """
    pending = True

    def on_step(
        obs: env.Observation, action: ActionT, next_obs: env.Observation
    ) -> bool:
        nonlocal pending
        if pending and next_obs.time >= push_time:
            pending = False
            bearing = next_obs.base_bearing
            environment.push(
                (-speed * math.sin(bearing), speed * math.cos(bearing))
            )
        return then(obs, action, next_obs)

    return on_step
