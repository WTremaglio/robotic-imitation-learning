"""Sampling, recording, and merging of expert episodes."""

import collections.abc
import dataclasses
import math
import os

import numpy as np
import numpy.typing as npt

from behavior_cloning import (
    control,
    env,
    episode,
    expert,
    perturbations,
    robot,
)

NAV_PHASES: tuple[control.NavPhase, ...] = ("align", "drive", "orient")
_PHASE_CODES = {phase: code for code, phase in enumerate(NAV_PHASES)}

# Recorded fields and their widths: robot state, the expert's
# noise-free labels, then the executed commands.
_FIELDS = {
    "time": 1,
    "base_pos": 3,
    "base_bearing": 1,
    "q_arm": robot.NUM_ARM_JOINTS,
    "qd_arm": robot.NUM_ARM_JOINTS,
    "wheel_speeds": robot.NUM_WHEELS,
    "ee_pos": 3,
    "v_cmd": 1,
    "w_cmd": 1,
    "q_des_arm": robot.NUM_ARM_JOINTS,
    "qd_des_arm": robot.NUM_ARM_JOINTS,
    "gravity_arm": robot.NUM_ARM_JOINTS,
    "wheel_cmds": robot.NUM_WHEELS,
    "torque_arm": robot.NUM_ARM_JOINTS,
}

# Names of the per-row and per-episode arrays of a recorded episode.
ROW_FIELDS = (*_FIELDS, "phase", "base_done")
EPISODE_FIELDS = (
    "seed",
    "target_pos",
    "base_start",
    "noise",
    "steps",
    "reached",
    "final_error",
    "stop_message",
)

# Second seed word of the start conditions, so they come from a
# different random stream than the noise.
_SPEC_STREAM = 1


@dataclasses.dataclass(frozen=True)
class NoiseLevels:
    """Standard deviations of the DART noise on the executed commands.

    Attributes:
        v_cmd: Noise on the base forward speed (m/s).
        w_cmd: Noise on the base turning rate (rad/s).
        q_des_arm: Noise on each arm joint reference (rad).
    """

    v_cmd: float = 0.0
    w_cmd: float = 0.0
    q_des_arm: float = 0.0

    @property
    def active(self) -> bool:
        """Whether any noise is applied."""
        return self.v_cmd > 0.0 or self.w_cmd > 0.0 or self.q_des_arm > 0.0


class DartExpert:
    """Expert whose executed commands are perturbed by DART noise.

    A noise sample is drawn every period steps and held in between, so
    the robot drifts away from the expert's path. Only the executed
    commands (wheel_cmds and torque_arm) of each action change; its
    labels stay the expert's noise-free ones. The wheels are left
    unperturbed once the base is parked, since the expert no longer
    corrects its position then.
    """

    def __init__(
        self,
        expert_policy: expert.ExpertPolicy,
        noise: NoiseLevels,
        period: int,
        rng: np.random.Generator,
    ) -> None:
        """Initialize the noisy expert.

        Args:
            expert_policy: Expert to perturb.
            noise: Noise standard deviations.
            period: Steps between noise samples.
            rng: Random number generator for the noise.
        """
        self._expert = expert_policy
        self._base_params = expert_policy.base_params
        self._arm_params = expert_policy.arm_params
        self._noise = noise
        self._active = noise.active
        self._period = period
        self._rng = rng
        self._countdown = 1
        self._v_noise = 0.0
        self._w_noise = 0.0
        self._q_noise = robot.ARM_ZEROS

    def reset(
        self, target_pos: robot.FloatArray, obs: env.Observation
    ) -> None:
        """Start a new episode.

        Args:
            target_pos: Target end-effector position, world frame (m).
            obs: First observation of the episode.
        """
        self._expert.reset(target_pos, obs)
        self._countdown = 1

    def act(self, obs: env.Observation) -> expert.ExpertAction:
        """Compute the noisy commands for one step.

        Args:
            obs: Current robot state.

        Returns:
            The expert's action, with its executed commands perturbed.
        """
        action = self._expert.act(obs)
        if not self._active:
            return action
        countdown = self._countdown - 1
        if countdown:
            self._countdown = countdown
        else:
            self._countdown = self._period
            noise = self._noise
            rng = self._rng
            self._v_noise = rng.normal(0.0, noise.v_cmd)
            self._w_noise = rng.normal(0.0, noise.w_cmd)
            self._q_noise = rng.normal(
                0.0, noise.q_des_arm, robot.NUM_ARM_JOINTS
            )
        if not action.base_done:
            action.wheel_cmds = control.wheel_velocities(
                action.v_cmd + self._v_noise,
                action.w_cmd + self._w_noise,
                self._base_params,
            )
        action.torque_arm = control.joint_torque(
            self._arm_params,
            action.q_des_arm + self._q_noise,
            action.qd_des_arm,
            obs.q[robot.ARM_MOTORS],
            obs.qd[robot.ARM_MOTORS],
            action.gravity_arm,
        )
        return action


class DaggerExpert:
    """Expert that labels the states a learned policy drives into.

    Each action keeps the expert's labels, but its executed commands
    (wheel_cmds and torque_arm) are the learner's.
    """

    def __init__(
        self,
        expert_policy: episode.Policy[expert.ExpertAction],
        learner: episode.Policy[episode.Action],
    ) -> None:
        """Pair an expert with the policy that drives.

        Args:
            expert_policy: Expert whose labels are recorded.
            learner: Policy whose commands move the robot.
        """
        self._expert = expert_policy
        self._learner = learner

    def reset(
        self, target_pos: robot.FloatArray, obs: env.Observation
    ) -> None:
        """Start a new episode for both policies.

        Args:
            target_pos: Target end-effector position, world frame (m).
            obs: First observation of the episode.
        """
        self._expert.reset(target_pos, obs)
        self._learner.reset(target_pos, obs)

    def act(self, obs: env.Observation) -> expert.ExpertAction:
        """Label the step with the expert and drive with the learner.

        Args:
            obs: Current robot state.

        Returns:
            The expert's action, with the learner's executed commands.
        """
        # The expert goes first, while the forward kinematics in the
        # Pinocchio data are still exactly the ones the step set up.
        action = self._expert.act(obs)
        learned_action = self._learner.act(obs)
        action.wheel_cmds = learned_action.wheel_cmds
        action.torque_arm = learned_action.torque_arm
        return action


class EpisodeRecorder:
    """Records every period-th step of an episode into arrays."""

    def __init__(self, max_steps: int, period: int) -> None:
        """Allocate storage for one episode.

        Args:
            max_steps: Most steps the episode can take.
            period: Steps between recorded rows.
        """
        self._period = period
        rows = -(-max_steps // period)
        self._data = {
            name: np.empty((rows, width), dtype=np.float32)
            for name, width in _FIELDS.items()
        }
        data = self._data
        self._time = data["time"]
        self._base_pos = data["base_pos"]
        self._base_bearing = data["base_bearing"]
        self._q_arm = data["q_arm"]
        self._qd_arm = data["qd_arm"]
        self._wheel_speeds = data["wheel_speeds"]
        self._ee_pos = data["ee_pos"]
        self._v_cmd = data["v_cmd"]
        self._w_cmd = data["w_cmd"]
        self._q_des_arm = data["q_des_arm"]
        self._qd_des_arm = data["qd_des_arm"]
        self._gravity_arm = data["gravity_arm"]
        self._wheel_cmds = data["wheel_cmds"]
        self._torque_arm = data["torque_arm"]
        self._phase = np.empty(rows, dtype=np.int8)
        self._base_done = np.empty(rows, dtype=np.bool_)
        self._countdown = 1
        self._rows = 0

    def __call__(
        self,
        obs: env.Observation,
        action: expert.ExpertAction,
        next_obs: env.Observation,
    ) -> bool:
        """Record the step if it starts a period.

        Args:
            obs: Observation before the step.
            action: Expert action, with executed commands.
            next_obs: Observation after the step (unused).

        Returns:
            False, so recording never ends the episode.
        """
        del next_obs
        countdown = self._countdown - 1
        if countdown:
            self._countdown = countdown
            return False
        self._countdown = self._period
        row = self._rows
        self._rows = row + 1
        self._time[row] = obs.time
        self._base_pos[row] = obs.base_pos
        self._base_bearing[row] = obs.base_bearing
        self._q_arm[row] = obs.q[robot.ARM_MOTORS]
        self._qd_arm[row] = obs.qd[robot.ARM_MOTORS]
        self._wheel_speeds[row] = obs.qd[robot.WHEEL_MOTORS]
        self._ee_pos[row] = obs.ee_pos
        self._v_cmd[row] = action.v_cmd
        self._w_cmd[row] = action.w_cmd
        self._q_des_arm[row] = action.q_des_arm
        self._qd_des_arm[row] = action.qd_des_arm
        self._gravity_arm[row] = action.gravity_arm
        self._wheel_cmds[row] = action.wheel_cmds
        self._torque_arm[row] = action.torque_arm
        self._phase[row] = _PHASE_CODES[action.phase]
        self._base_done[row] = action.base_done
        return False

    def arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        """Return the recorded rows, one array per field.

        Returns:
            Arrays of shape (rows, width), plus the phase codes and
            base-done flags of shape (rows,).
        """
        rows = self._rows
        arrays: dict[str, npt.NDArray[np.generic]] = {
            name: values[:rows] for name, values in self._data.items()
        }
        arrays["phase"] = self._phase[:rows]
        arrays["base_done"] = self._base_done[:rows]
        return arrays


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """Start conditions of one recorded episode.

    Attributes:
        seed: Seed of the episode's start conditions and noise.
        target_pos: Target end-effector position, world frame (m).
        base_xy: Start position of the base (m).
        base_bearing: Start bearing of the base (rad).
        noise: DART noise standard deviations.
    """

    seed: int
    target_pos: tuple[float, float, float]
    base_xy: tuple[float, float]
    base_bearing: float
    noise: NoiseLevels


@dataclasses.dataclass(frozen=True)
class TaskRanges:
    """Ranges the start conditions of an episode are sampled from.

    The defaults are the training ranges.

    Attributes:
        base_radius: Largest distance of the base start from the world
            origin (m).
        target_radius: Range of the horizontal distance of the target
            from the world origin (m).
        target_height: Range of the target height (m).
    """

    base_radius: float = 0.5
    target_radius: tuple[float, float] = (0.6, 1.6)
    target_height: tuple[float, float] = (0.3, 1.0)


def sample_spec(
    seed: int, ranges: TaskRanges, noise: NoiseLevels
) -> EpisodeSpec:
    """Sample the start conditions of one episode.

    The base starts uniformly within a disc around the world origin,
    with any bearing. The target lies in any direction, with its
    horizontal distance and height uniform within their ranges.

    Args:
        seed: Seed of the episode; the same seed and ranges give the
            same start conditions.
        ranges: Ranges to sample from.
        noise: DART noise standard deviations.

    Returns:
        The start conditions.
    """
    rng = np.random.default_rng((seed, _SPEC_STREAM))
    base_angle, base_bearing, target_angle = rng.uniform(-math.pi, math.pi, 3)
    base_range = ranges.base_radius * math.sqrt(rng.random())
    target_range = rng.uniform(*ranges.target_radius)
    return EpisodeSpec(
        seed,
        (
            target_range * math.cos(target_angle),
            target_range * math.sin(target_angle),
            float(rng.uniform(*ranges.target_height)),
        ),
        (base_range * math.cos(base_angle), base_range * math.sin(base_angle)),
        float(base_bearing),
        noise,
    )


def record_episode(
    environment: env.Environment,
    expert_policy: expert.ExpertPolicy,
    spec: EpisodeSpec,
    criteria: episode.StopCriteria,
    period: int,
    learner: episode.Policy[episode.Action] | None = None,
    push: tuple[float, float] | None = None,
) -> dict[str, npt.NDArray[np.generic]]:
    """Run and record one expert episode.

    Args:
        environment: Simulation to run in.
        expert_policy: Expert to run and label with.
        spec: Start conditions.
        criteria: Stop thresholds; max_steps must be positive.
        period: Steps between recorded rows and noise samples.
        learner: Policy that drives while the expert labels (DAgger), or
            None for the expert to drive itself.
        push: Sideways speed, positive to the left (m/s), and time (s)
            of one push of the base, or None for no push.

    Returns:
        The recorded arrays and the episode's settings and outcome.

    Raises:
        ValueError: If criteria.max_steps is not positive.
    """
    if criteria.max_steps <= 0:
        raise ValueError("Recording needs a positive step cap")
    policy: episode.Policy[expert.ExpertAction] = (
        DartExpert(
            expert_policy, spec.noise, period, np.random.default_rng(spec.seed)
        )
        if learner is None
        else DaggerExpert(expert_policy, learner)
    )
    recorder = EpisodeRecorder(criteria.max_steps, period)
    result = episode.run_episode(
        environment,
        policy,
        spec.target_pos,
        criteria,
        spec.base_xy,
        spec.base_bearing,
        on_step=(
            recorder
            if push is None
            else perturbations.push_callback(environment, *push, recorder)
        ),
    )
    final_error = np.linalg.norm(
        np.asarray(spec.target_pos) - result.final_obs.ee_pos
    )
    arrays = recorder.arrays()
    arrays.update(
        seed=np.array(spec.seed),
        target_pos=np.array(spec.target_pos),
        base_start=np.array((*spec.base_xy, spec.base_bearing)),
        noise=np.array(
            (spec.noise.v_cmd, spec.noise.w_cmd, spec.noise.q_des_arm)
        ),
        steps=np.array(result.steps),
        reached=np.array(final_error < criteria.reach_tolerance),
        final_error=np.array(final_error),
        stop_message=np.array(result.stop_message or ""),
    )
    return arrays


def save_arrays(
    path: str | os.PathLike[str],
    arrays: dict[str, npt.NDArray[np.generic]],
) -> None:
    """Save named arrays as an uncompressed .npz file.

    Args:
        path: File to write.
        arrays: Arrays to save, such as a recorded episode.
    """
    np.savez(path, allow_pickle=False, **arrays)


def load_arrays(
    path: str | os.PathLike[str],
) -> dict[str, npt.NDArray[np.generic]]:
    """Load all named arrays of an .npz file.

    Args:
        path: File to read.

    Returns:
        The arrays, by name.
    """
    with np.load(path) as saved:
        return {name: saved[name] for name in saved.files}


def merge_episodes(
    episodes: collections.abc.Sequence[
        collections.abc.Mapping[str, npt.NDArray[np.generic]]
    ],
) -> dict[str, npt.NDArray[np.generic]]:
    """Merge recorded episodes into one dataset.

    Row fields are joined end to end, so the rows of episode k are
    episode_starts[k] to episode_starts[k + 1]. Episode values are
    stacked, one entry per episode.

    Args:
        episodes: Recorded episodes, in dataset order.

    Returns:
        The merged arrays, plus episode_starts.

    Raises:
        ValueError: If episodes is empty.
    """
    if not episodes:
        raise ValueError("No episodes to merge")
    per_row: dict[str, list[npt.NDArray[np.generic]]] = {
        name: [] for name in ROW_FIELDS
    }
    per_episode: dict[str, list[npt.NDArray[np.generic]]] = {
        name: [] for name in EPISODE_FIELDS
    }
    for recorded in episodes:
        for name, values in per_row.items():
            values.append(recorded[name])
        for name, values in per_episode.items():
            values.append(recorded[name])
    merged = {name: np.concatenate(values) for name, values in per_row.items()}
    merged.update(
        (name, np.stack(values)) for name, values in per_episode.items()
    )
    episode_starts = np.zeros(len(episodes) + 1, dtype=np.int64)
    np.cumsum([len(time) for time in per_row["time"]], out=episode_starts[1:])
    merged["episode_starts"] = episode_starts
    return merged
