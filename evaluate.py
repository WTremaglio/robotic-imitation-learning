"""Evaluate the expert or a trained policy on held-out episodes.

Runs one episode per seed in parallel worker processes, each with its
own simulation. Test seeds start at 1000, clear of every seed used to
record training data, and a seed always gives the same start
conditions, so every policy is scored on the same episodes.

Writes evaluations/NAME.json: the settings, each episode's outcome,
and the summary.

Usage:
    python evaluate.py POLICY [--name NAME] [--episodes N] [--seed S]
        [--workers W] [--tolerance T] [--ranges {train,wide}]
        [--obs-noise S] [--cmd-noise K] [--push V]

POLICY is "expert", a checkpoint name under checkpoints/, or several
checkpoint names joined by commas, which run as an ensemble. A trained
policy's network decision time is measured before the episodes run.
"""

import argparse
import dataclasses
import json
import multiprocessing
import os
import time
import timeit

import numpy as np
import numpy.typing as npt
import torch

from behavior_cloning import (
    data,
    env,
    episode,
    expert,
    features,
    learned,
    paths,
    perturbations,
    robot,
)

_EXPERT = "expert"
# Final error within which an episode counts as close (m).
_CLOSE_ERROR = 0.02
# Task ranges by name: the training ranges, and wider ones for testing
# generalization.
_RANGES = {
    "train": data.TaskRanges(),
    "wide": data.TaskRanges(
        base_radius=1.0, target_radius=(0.4, 2.0), target_height=(0.15, 1.2)
    ),
}
# Decisions per timing run, and timing runs per measurement.
_TIMING_CALLS = 100
_TIMING_RUNS = 5


@dataclasses.dataclass(frozen=True)
class EvalSettings:
    """Settings that determine an evaluation.

    Attributes:
        policy: "expert" or a checkpoint name.
        seed: Seed of the first episode.
        episodes: Number of episodes.
        ranges: Ranges of the start conditions.
        criteria: Stop thresholds.
        finger_angles: Finger angles held in every episode (rad).
        perturbation: Adversities applied in every episode.
    """

    policy: str
    seed: int
    episodes: int
    ranges: data.TaskRanges
    criteria: episode.StopCriteria
    finger_angles: tuple[float, ...]
    perturbation: perturbations.Perturbation


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    """Result of one evaluation episode.

    Attributes:
        seed: Episode seed.
        reached: Whether the final error is within the reach
            tolerance.
        final_error: End-effector distance from the target at the end
            (m).
        time: Simulation time at the end (s).
        park_time: Time the policy first judged the base parked (s),
            or None if it never did.
        reach_time: Time the end effector first came within the reach
            tolerance (s), or None if it never did.
        stop_message: Why the episode stopped.
    """

    seed: int
    reached: bool
    final_error: float
    time: float
    park_time: float | None
    reach_time: float | None
    stop_message: str


class _Timer:
    """Records when the base first parks and the hand first gets close.

    Attributes:
        park_time: Time of the first parked step (s), or None.
        reach_time: Time the end effector first came within the
            tolerance (s), or None.
    """

    def __init__(self, target_pos: npt.ArrayLike, tolerance: float) -> None:
        """Start with neither moment seen.

        Args:
            target_pos: Target end-effector position, world frame (m).
            tolerance: Reach tolerance (m).
        """
        self._target = np.asarray(target_pos, dtype=np.float64)
        self._tolerance_sq = tolerance * tolerance
        self.park_time: float | None = None
        self.reach_time: float | None = None

    def __call__(
        self,
        obs: env.Observation,
        action: episode.Action,
        next_obs: env.Observation,
    ) -> bool:
        """Note the first parked step and the first close one.

        The hand is only checked once the base is parked, since it
        never reaches before then.

        Args:
            obs: Observation before the step.
            action: Policy action of the step.
            next_obs: Observation after the step (unused).

        Returns:
            False, so timing never ends the episode.
        """
        del next_obs
        if action.base_done and self.reach_time is None:
            if self.park_time is None:
                self.park_time = obs.time
            error = self._target - obs.ee_pos
            if error.dot(error) < self._tolerance_sq:
                self.reach_time = obs.time
        return False


class _Worker:
    """Simulation and policy of one worker process."""

    def __init__(self, settings: EvalSettings) -> None:
        """Build the simulation and the policy.

        Args:
            settings: Settings of the evaluation.
        """
        self.settings = settings
        self.environment = env.Environment(
            paths.REPO_DIR, fixed_finger_angles=settings.finger_angles
        )
        self.policy: episode.Policy[episode.Action]
        if settings.policy == _EXPERT:
            self.policy = expert.ExpertPolicy.from_environment(
                self.environment
            )
        else:
            torch.set_num_threads(1)
            self.policy = learned.load_policy(
                self.environment, settings.policy
            )


_worker: _Worker | None = None


def _init_worker(settings: EvalSettings) -> None:
    """Build the worker's simulation once, for all its episodes."""
    global _worker
    _worker = _Worker(settings)


def _run(seed: int) -> EpisodeOutcome:
    """Run one evaluation episode in a worker process.

    Args:
        seed: Episode seed.

    Returns:
        The episode's outcome.
    """
    assert _worker is not None
    settings = _worker.settings
    spec = data.sample_spec(seed, settings.ranges, data.NoiseLevels())
    timer = _Timer(spec.target_pos, settings.criteria.reach_tolerance)
    perturbation = settings.perturbation
    policy = (
        perturbations.PerturbedPolicy(_worker.policy, perturbation, seed)
        if perturbation.perturbs_policy
        else _worker.policy
    )
    result = episode.run_episode(
        _worker.environment,
        policy,
        spec.target_pos,
        settings.criteria,
        spec.base_xy,
        spec.base_bearing,
        on_step=(
            perturbations.push_callback(
                _worker.environment,
                perturbation.push_speed,
                perturbation.push_time,
                timer,
            )
            if perturbation.push_speed > 0.0
            else timer
        ),
    )
    error = float(
        np.linalg.norm(np.asarray(spec.target_pos) - result.final_obs.ee_pos)
    )
    reached = error < settings.criteria.reach_tolerance
    reach_time = timer.reach_time
    if reach_time is None and reached:
        reach_time = result.final_obs.time
    return EpisodeOutcome(
        seed,
        reached,
        error,
        result.final_obs.time,
        timer.park_time,
        reach_time,
        result.stop_message or "",
    )


def _summary(
    outcomes: list[EpisodeOutcome],
    min_time: float,
) -> dict[str, int | float | None]:
    """Return the headline numbers of an evaluation.

    Args:
        outcomes: Outcome of each episode.
        min_time: Time before which the task never counts a reach (s).

    Returns:
        Counts of reached, close, parked, and late episodes (reached
        only after min_time), the median final error (m), and the mean
        times to first come within tolerance and to park (s), which
        are None when no episode reached or parked.
    """
    errors = np.array([outcome.final_error for outcome in outcomes])
    reached = [outcome for outcome in outcomes if outcome.reached]
    reach_times = [
        outcome.reach_time
        for outcome in reached
        if outcome.reach_time is not None
    ]
    park_times = [
        outcome.park_time
        for outcome in outcomes
        if outcome.park_time is not None
    ]
    return {
        "episodes": len(outcomes),
        "reached": len(reached),
        "close": int((errors < _CLOSE_ERROR).sum()),
        "parked": len(park_times),
        "late": sum(
            1
            for outcome in reached
            if outcome.reach_time is not None and outcome.reach_time > min_time
        ),
        "median_error": float(np.median(errors)),
        "mean_reach_time": (
            float(np.mean(reach_times)) if reach_times else None
        ),
        "mean_park_time": float(np.mean(park_times)) if park_times else None,
    }


def _decision_time(policy: str) -> float | None:
    """Time one decision of a trained policy's model.

    Uses learned.decision_function, the path LearnedPolicy runs.

    Args:
        policy: "expert", a checkpoint name, or names joined by commas.

    Returns:
        The fastest mean time of one decision over the timing runs
        (s), or None for the expert.
    """
    if policy == _EXPERT:
        return None
    torch.set_num_threads(1)
    network = learned.load_network(policy)
    window = np.zeros(
        (network.config.history, features.STATE_WIDTH), dtype=np.float32
    )
    decide = learned.decision_function(network, window)
    runs = timeit.repeat(decide, number=_TIMING_CALLS, repeat=_TIMING_RUNS)
    return min(runs) / _TIMING_CALLS


def _report(summary: dict[str, int | float | None], seconds: float) -> None:
    """Print the summary of an evaluation.

    Args:
        summary: Summary from _summary.
        seconds: Evaluation time (s).
    """
    episodes = summary["episodes"]
    reach_time = summary["mean_reach_time"]
    park_time = summary["mean_park_time"]
    decision_time = summary["decision_time"]
    print("==================================================================")
    print(f"Reached                   : {summary['reached']}/{episodes}")
    print(f"Within 2 cm               : {summary['close']}/{episodes}")
    print(f"Parked                    : {summary['parked']}/{episodes}")
    print(f"Median final error (m)    : {summary['median_error']:.4f}")
    print(f"Reached after 8 s         : {summary['late']}/{episodes}")
    print(
        "Mean time to tolerance (s): "
        + (f"{reach_time:.2f}" if reach_time is not None else "-")
    )
    print(
        "Mean time to park (s)     : "
        + (f"{park_time:.2f}" if park_time is not None else "-")
    )
    print(
        "Decision time (us)        : "
        + ("-" if decision_time is None else f"{decision_time * 1e6:.1f}")
    )
    print(f"Evaluation time (s)       : {seconds:.1f}")
    print("==================================================================")


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on held-out episodes."
    )
    parser.add_argument(
        "policy", help='"expert", a checkpoint name, or names joined by commas'
    )
    parser.add_argument(
        "--name", help="results file name under evaluations/ (default: POLICY)"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="number of episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="seed of the first episode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="worker processes (default: one per CPU core)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="reach tolerance (m) (default: 0.01)",
    )
    parser.add_argument(
        "--ranges",
        choices=sorted(_RANGES),
        default="train",
        help="target and start ranges (default: train)",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.0,
        help="noise on measured positions (m) and angles (rad)",
    )
    parser.add_argument(
        "--cmd-noise",
        type=float,
        default=0.0,
        help="command noise as a multiple of the DART noise",
    )
    parser.add_argument(
        "--push",
        type=float,
        default=0.0,
        help="sideways push speed at 2 s (m/s)",
    )
    args = parser.parse_args()
    if min(args.obs_noise, args.cmd_noise, args.push) < 0.0:
        parser.error("noise and push must not be negative")
    if args.episodes < 1 or args.workers < 1:
        parser.error("episodes and workers must be positive")
    if args.seed < 0:
        parser.error("seed must not be negative")
    if args.policy != _EXPERT:
        for name in args.policy.split(","):
            if not learned.checkpoint_exists(name):
                parser.error(f"no checkpoint named {name}")
        if "," in args.policy and args.name is None:
            parser.error("name an ensemble's results with --name")
    if args.tolerance is not None and args.tolerance <= 0.0:
        parser.error("tolerance must be positive")
    return args


def main() -> None:
    """Run the episodes, save the outcomes, and report the summary."""
    args = _parse_args()
    settings = EvalSettings(
        policy=args.policy,
        seed=args.seed,
        episodes=args.episodes,
        ranges=_RANGES[args.ranges],
        criteria=(
            episode.StopCriteria()
            if args.tolerance is None
            else episode.StopCriteria(reach_tolerance=args.tolerance)
        ),
        finger_angles=robot.FINGER_HOLD_ANGLES,
        perturbation=perturbations.Perturbation(
            obs_noise=args.obs_noise,
            cmd_noise=args.cmd_noise,
            push_speed=args.push,
        ),
    )
    decision_time = _decision_time(args.policy)
    outcomes: list[EpisodeOutcome] = []
    start = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        min(args.workers, args.episodes),
        initializer=_init_worker,
        initargs=(settings,),
    ) as pool:
        seeds = range(args.seed, args.seed + args.episodes)
        for done, outcome in enumerate(
            pool.imap_unordered(_run, seeds), start=1
        ):
            outcomes.append(outcome)
            print(
                f"[{done}/{args.episodes}] seed {outcome.seed}: "
                f"{'reached' if outcome.reached else 'missed '} "
                f"error {outcome.final_error:.4f} m"
            )
    seconds = time.perf_counter() - start
    outcomes.sort(key=lambda outcome: outcome.seed)
    summary = _summary(outcomes, settings.criteria.min_time)
    summary["decision_time"] = decision_time

    path = os.path.join(
        paths.EVALUATION_DIR, f"{args.name or args.policy}.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": dataclasses.asdict(settings),
                "summary": summary,
                "seconds": seconds,
                "outcomes": [
                    dataclasses.asdict(outcome) for outcome in outcomes
                ],
            },
            file,
            indent=2,
        )
    _report(summary, seconds)


if __name__ == "__main__":
    main()
