"""Stop conditions and the episode loop."""

import collections.abc
import dataclasses
import math
import typing

import numpy as np
import numpy.typing as npt

from behavior_cloning import env, robot


@dataclasses.dataclass(frozen=True)
class StopCriteria:
    """Thresholds for the episode stop conditions.

    Attributes:
        reach_tolerance: End-effector error that counts as reached (m).
        max_steps: Step cap; 0 disables it.
        min_time: Time before reached or stalled can trigger (s).
        stall_window: Time without meaningful improvement of the
            end-effector error that counts as a stall (s).
        stall_improvement: Smallest error decrease that counts as an
            improvement (m).
    """

    reach_tolerance: float = 0.01
    max_steps: int = 40000
    min_time: float = 8.0
    stall_window: float = 3.0
    stall_improvement: float = 0.002


@dataclasses.dataclass
class StallState:
    """Running state of the stall check.

    Attributes:
        best_ee_error: Smallest end-effector error so far (m).
        best_ee_error_time: Time at which best_ee_error was reached (s).
    """

    best_ee_error: float = math.inf
    best_ee_error_time: float = 0.0


class Action(typing.Protocol):
    """Commands a policy returns for one step."""

    wheel_cmds: robot.FloatArray
    torque_arm: robot.FloatArray
    base_done: bool


class Policy[ActionT: Action](typing.Protocol):
    """Controller that the episode loop can run."""

    def reset(
        self, target_pos: robot.FloatArray, obs: env.Observation
    ) -> None:
        """Start a new episode."""
        ...

    def act(self, obs: env.Observation) -> ActionT:
        """Compute the commands for one step."""
        ...


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one episode.

    Attributes:
        stop_message: Why the episode stopped, or None if a callback
            ended it.
        final_obs: Observation at which the episode stopped.
        steps: Number of simulation steps taken.
    """

    stop_message: str | None
    final_obs: env.Observation
    steps: int


def check_stop(
    criteria: StopCriteria,
    state: StallState,
    step: int,
    current_time: float,
    ee_dist_error: float,
    base_done: bool,
) -> str | None:
    """Evaluate the stop conditions for the current step.

    Stall tracking runs only once the base is done, while the arm is
    reaching. Updates state.

    Args:
        criteria: Stop thresholds.
        state: Running state of the stall check.
        step: Index of the current step.
        current_time: Simulation time (s).
        ee_dist_error: Distance between end effector and target (m).
        base_done: Whether the base has parked.

    Returns:
        A message saying why the simulation should stop, or None to
        continue.
    """
    if criteria.max_steps and step >= criteria.max_steps:
        return (
            f"Reached step cap {criteria.max_steps} at t={current_time:.2f} s"
        )
    if (
        base_done
        and ee_dist_error < criteria.reach_tolerance
        and current_time > criteria.min_time
    ):
        return (
            "Target reached within "
            f"{criteria.reach_tolerance * 1000:.0f} mm tolerance at "
            f"t={current_time:.2f} s"
        )
    if base_done:
        if ee_dist_error < state.best_ee_error - criteria.stall_improvement:
            state.best_ee_error = ee_dist_error
            state.best_ee_error_time = current_time
        elif (
            current_time - state.best_ee_error_time > criteria.stall_window
            and current_time > criteria.min_time
        ):
            return (
                "Target not reachable: closest pose reached at "
                f"t={current_time:.2f} s (best EE error "
                f"{state.best_ee_error:.4f} m, current "
                f"{ee_dist_error:.4f} m)"
            )
    return None


def run_episode[ActionT: Action](
    environment: env.Environment,
    policy: Policy[ActionT],
    target_pos: npt.ArrayLike,
    criteria: StopCriteria,
    base_xy: tuple[float, float] | None = None,
    base_bearing: float | None = None,
    on_step: collections.abc.Callable[
        [env.Observation, ActionT, env.Observation], bool
    ]
    | None = None,
) -> EpisodeResult:
    """Run one episode of a policy until a stop condition holds.

    Args:
        environment: Simulation to run in.
        policy: Controller to run.
        target_pos: Target end-effector position, world frame (m).
        criteria: Stop thresholds.
        base_xy: Start position of the base (m), or None to keep the
            initial one.
        base_bearing: Start bearing of the base (rad), or None to keep
            the initial one.
        on_step: Called after each step that does not stop the episode,
            with the observation, the action and the next observation;
            returning True ends the episode.

    Returns:
        The outcome of the episode.
    """
    target = np.array(target_pos, dtype=np.float64)
    target.flags.writeable = False
    obs = environment.reset(target, base_xy, base_bearing)
    policy.reset(target, obs)
    state = StallState()
    steps = 0
    while True:
        action = policy.act(obs)
        next_obs = environment.step(action.wheel_cmds, action.torque_arm)
        if action.base_done:
            ee_error = target - obs.ee_pos
            ee_dist_error = math.sqrt(ee_error.dot(ee_error))
        else:
            ee_dist_error = math.inf
        stop_message = check_stop(
            criteria, state, steps, obs.time, ee_dist_error, action.base_done
        )
        steps += 1
        if stop_message:
            return EpisodeResult(stop_message, obs, steps)
        if on_step is not None and on_step(obs, action, next_obs):
            return EpisodeResult(None, next_obs, steps)
        obs = next_obs
