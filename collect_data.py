"""Record expert demonstrations for behavior cloning.

Runs expert episodes in parallel worker processes, each with its own
simulation, and saves one file per episode. Episode n uses seed
seed + n, so the data does not depend on the number of workers or the
order episodes finish in. A rerun with the same settings skips the
episodes already saved. All episodes are merged into one dataset file.

Output, under data/NAME:
    settings.json: Settings that determine the data.
    episodes/NNNNN.npz: One recorded episode each.
    dataset.npz: All episodes merged.

Usage:
    python collect_data.py NAME [--episodes N] [--seed S] [--workers W]
        [--v-noise V] [--w-noise W] [--q-noise Q] [--policy P]
        [--push-speed MIN MAX] [--push-time MIN MAX]
"""

import argparse
import dataclasses
import json
import multiprocessing
import os
import time

import numpy as np
import numpy.typing as npt
import torch

from behavior_cloning import (
    control,
    data,
    env,
    episode,
    expert,
    learned,
    paths,
    perturbations,
    robot,
)


@dataclasses.dataclass(frozen=True)
class RunSettings:
    """Settings that determine the recorded data.

    The number of episodes is not one of them: a rerun with more
    episodes extends the data.

    Attributes:
        seed: Seed of the first episode.
        period: Steps per recorded row and noise sample.
        noise: DART noise standard deviations.
        base_params: Base controller parameters of the expert.
        arm_params: Arm controller parameters of the expert.
        planner_params: Planner parameters of the expert.
        ranges: Ranges of the start conditions.
        criteria: Stop thresholds.
        finger_angles: Finger angles held in every episode (rad).
        policy: Checkpoint(s) whose policy drives while the expert
            labels (DAgger); None for the expert driving itself.
        push_speed: Range of the speed of one sideways push per
            episode (m/s), or None for no pushes.
        push_time: Range of the push time (s).
    """

    seed: int
    period: int
    noise: data.NoiseLevels
    base_params: control.BaseControllerParams
    arm_params: control.ArmControllerParams
    planner_params: expert.PlannerParams
    ranges: data.TaskRanges
    criteria: episode.StopCriteria
    finger_angles: tuple[float, ...]
    policy: str | None
    push_speed: tuple[float, float] | None
    push_time: tuple[float, float]


class _Worker:
    """Simulation and expert of one worker process."""

    def __init__(self, settings: RunSettings, episodes_dir: str) -> None:
        """Build the simulation and the expert.

        Args:
            settings: Settings of the run.
            episodes_dir: Folder the episode files are saved in.
        """
        self.settings = settings
        self.episodes_dir = episodes_dir
        self.environment = env.Environment(
            paths.REPO_DIR, fixed_finger_angles=settings.finger_angles
        )
        self.expert_policy = expert.ExpertPolicy.from_environment(
            self.environment,
            settings.base_params,
            settings.arm_params,
            settings.planner_params,
        )
        self.learner: learned.LearnedPolicy | None = None
        if settings.policy is not None:
            torch.set_num_threads(1)
            self.learner = learned.load_policy(
                self.environment, settings.policy
            )


_worker: _Worker | None = None


def _init_worker(settings: RunSettings, episodes_dir: str) -> None:
    """Build the worker's simulation once, for all its episodes."""
    global _worker
    _worker = _Worker(settings, episodes_dir)


def _episode_path(episodes_dir: str, index: int) -> str:
    """Return the file of episode index."""
    return os.path.join(episodes_dir, f"{index:05d}.npz")


def _collect(index: int) -> tuple[int, dict[str, npt.NDArray[np.generic]]]:
    """Record and save one episode in a worker process.

    Args:
        index: Episode index.

    Returns:
        A tuple (index, arrays) of the episode index and its recorded
        arrays.
    """
    assert _worker is not None
    settings = _worker.settings
    seed = settings.seed + index
    spec = data.sample_spec(seed, settings.ranges, settings.noise)
    push = (
        None
        if settings.push_speed is None
        else perturbations.sample_push(
            seed, settings.push_speed, settings.push_time
        )
    )
    arrays = data.record_episode(
        _worker.environment,
        _worker.expert_policy,
        spec,
        settings.criteria,
        settings.period,
        _worker.learner,
        push,
    )
    path = _episode_path(_worker.episodes_dir, index)
    # Written under a temporary name, so a crash never leaves a
    # partial file under the final one.
    temp_path = path.removesuffix(".npz") + ".tmp.npz"
    data.save_arrays(temp_path, arrays)
    os.replace(temp_path, path)
    return index, arrays


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Record expert demonstrations."
    )
    parser.add_argument("name", help="dataset folder name under data/")
    parser.add_argument(
        "--episodes", type=int, default=300, help="number of episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the first episode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="worker processes (default: one per CPU core)",
    )
    parser.add_argument(
        "--v-noise", type=float, default=0.0, help="DART noise on v (m/s)"
    )
    parser.add_argument(
        "--w-noise", type=float, default=0.0, help="DART noise on w (rad/s)"
    )
    parser.add_argument(
        "--q-noise",
        type=float,
        default=0.0,
        help="DART noise on each arm joint reference (rad)",
    )
    parser.add_argument(
        "--policy",
        help="checkpoint(s) that drives while the expert labels (DAgger)",
    )
    parser.add_argument(
        "--push-speed",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="range of one sideways push per episode (m/s)",
    )
    parser.add_argument(
        "--push-time",
        type=float,
        nargs=2,
        default=(0.5, 2.5),
        metavar=("MIN", "MAX"),
        help="range of the push time (s) (default: 0.5 2.5)",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error("episodes and workers must be positive")
    if args.seed < 0:
        parser.error("seed must not be negative")
    if min(args.v_noise, args.w_noise, args.q_noise) < 0.0:
        parser.error("noise levels must not be negative")
    if args.policy is not None:
        if max(args.v_noise, args.w_noise, args.q_noise) > 0.0:
            parser.error("DAgger runs take no DART noise")
        for name in args.policy.split(","):
            if not learned.checkpoint_exists(name):
                parser.error(f"no checkpoint named {name}")
    if args.push_speed is not None and not (
        0.0 < args.push_speed[0] <= args.push_speed[1]
    ):
        parser.error("push speeds must be positive, with MIN <= MAX")
    if not 0.0 <= args.push_time[0] <= args.push_time[1]:
        parser.error("push times must not be negative, with MIN <= MAX")
    return args


def _check_settings(path: str, settings: RunSettings) -> None:
    """Save the settings, or check them against a previous run.

    Args:
        path: Settings file.
        settings: Settings of this run.

    Raises:
        SystemExit: If the folder holds data from other settings.
    """
    saved = dataclasses.asdict(settings)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            previous = json.load(file)
        # JSON stores tuples as lists.
        if previous != json.loads(json.dumps(saved)):
            raise SystemExit(f"{path} holds other settings; use another name")
        return
    with open(path, "w", encoding="utf-8") as file:
        json.dump(saved, file, indent=2)


def _report(dataset_path: str, seconds: float) -> None:
    """Print the expert's results, read back from the saved dataset.

    Args:
        dataset_path: Saved dataset file.
        seconds: Recording time (s).
    """
    with np.load(dataset_path) as dataset:
        reached = dataset["reached"]
        mean_error = dataset["final_error"].mean()
        mean_steps = dataset["steps"].mean()
        rows = dataset["time"].shape[0]

    print("==================================================================")
    print(f"Episodes reached          : {reached.sum()}/{reached.size}")
    print(f"Mean final error (m)      : {mean_error:.4f}")
    print(f"Mean steps per episode    : {mean_steps:.0f}")
    print(f"Rows                      : {rows}")
    print(
        "Dataset size (MB)         : "
        f"{os.path.getsize(dataset_path) / 1e6:.1f}"
    )
    print(f"Recording time (s)        : {seconds:.1f}")
    print("==================================================================")


def main() -> None:
    """Record and merge the episodes, then report the results."""
    args = _parse_args()
    settings = RunSettings(
        seed=args.seed,
        period=robot.POLICY_PERIOD,
        noise=data.NoiseLevels(args.v_noise, args.w_noise, args.q_noise),
        base_params=control.BaseControllerParams(),
        arm_params=control.ArmControllerParams(),
        planner_params=expert.PlannerParams(),
        ranges=data.TaskRanges(),
        criteria=episode.StopCriteria(),
        finger_angles=robot.FINGER_HOLD_ANGLES,
        policy=args.policy,
        push_speed=(
            None
            if args.push_speed is None
            else (args.push_speed[0], args.push_speed[1])
        ),
        push_time=(args.push_time[0], args.push_time[1]),
    )
    out_dir = os.path.join(paths.DATA_DIR, args.name)
    episodes_dir = os.path.join(out_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)
    _check_settings(os.path.join(out_dir, "settings.json"), settings)

    episode_paths = [
        _episode_path(episodes_dir, n) for n in range(args.episodes)
    ]
    pending = [
        n for n, path in enumerate(episode_paths) if not os.path.exists(path)
    ]
    print(
        f"{args.episodes - len(pending)} episodes already saved, "
        f"recording {len(pending)} with {args.workers} workers"
    )

    # Episodes recorded in this run, kept in memory so the merge does
    # not read their files back.
    recorded: dict[int, dict[str, npt.NDArray[np.generic]]] = {}

    start = time.perf_counter()
    if pending:
        # Spawn starts each worker fresh; PyBullet is not fork-safe.
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            min(args.workers, len(pending)),
            initializer=_init_worker,
            initargs=(settings, episodes_dir),
        ) as pool:
            for done, (index, arrays) in enumerate(
                pool.imap_unordered(_collect, pending), start=1
            ):
                recorded[index] = arrays
                print(
                    f"[{done}/{len(pending)}] episode {index}: "
                    f"{'reached ' if arrays['reached'] else 'missed  '}"
                    f"error {arrays['final_error']:.4f} m, "
                    f"{arrays['steps']} steps"
                )

    dataset = data.merge_episodes(
        [
            recorded[n] if n in recorded else data.load_arrays(path)
            for n, path in enumerate(episode_paths)
        ]
    )
    for name, values in dataset.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise SystemExit(f"Non-finite values in {name}")
    dataset_path = paths.dataset_file(args.name)
    temp_path = os.path.join(out_dir, "dataset.tmp.npz")
    data.save_arrays(temp_path, dataset)
    os.replace(temp_path, dataset_path)
    _report(dataset_path, time.perf_counter() - start)


if __name__ == "__main__":
    main()
