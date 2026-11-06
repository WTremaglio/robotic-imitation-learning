"""Watch the expert or a trained policy drive the robot in the GUI.

Usage:
    python demo.py [POLICY] [--seed S] [--episodes N]

POLICY is a named model, each an ensemble of three seeds:
    mlp: an MLP that sees the current step.
    mlp-push: the final MLP, also trained on 300 demonstrations with
        a sideways push.
    mlp-history: an MLP that sees the last 10 steps.
    mlp-baseline: an MLP trained on plain behavior cloning on 300 clean
        demonstrations, without DART noise or prioritized replay.
    lstm: an LSTM that sees the last 10 steps.
    xgb: a gradient-boosted tree policy that sees the current step.

Leave POLICY out to run the final model. POLICY can also be "expert",
a checkpoint name under checkpoints/, or several checkpoint names
joined by commas, which run as an ensemble.
"""

import argparse

import numpy as np

from behavior_cloning import data, env, episode, expert, learned, paths, robot

_EXPERT = "expert"
_ENSEMBLES = {
    "mlp": "v2-08-replay-gentle",
    "mlp-push": "v2-13-push",
    "mlp-history": "v2-09-mlp-h10",
    "mlp-baseline": "v2-11-mlp-baseline",
    "lstm": "v2-10-lstm-h10",
    "xgb": "v2-12-xgb",
}
_DEFAULT_POLICY = "mlp"
_SEEDS = 3


def _checkpoints(policy: str) -> str:
    """Return a named ensemble's checkpoints, or the policy as given."""
    if policy not in _ENSEMBLES:
        return policy
    name = _ENSEMBLES[policy]
    return ",".join(f"{name}/s{seed}" for seed in range(_SEEDS))


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch a policy drive the robot."
    )
    parser.add_argument(
        "policy",
        nargs="?",
        default=_DEFAULT_POLICY,
        type=_checkpoints,
        help=f"{', '.join(_ENSEMBLES)}, {_EXPERT}, or checkpoint "
        f"names joined by commas (default: {_DEFAULT_POLICY})",
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="seed of the first episode"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="number of episodes"
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.seed < 0:
        parser.error("seed must not be negative")
    if args.policy != _EXPERT:
        for name in args.policy.split(","):
            if not learned.checkpoint_exists(name):
                parser.error(f"no checkpoint named {name}")
    return args


def main() -> None:
    """Play the episodes in the GUI and print how each ended."""
    args = _parse_args()
    environment = env.Environment(
        paths.REPO_DIR,
        use_gui=True,
        fixed_finger_angles=robot.FINGER_HOLD_ANGLES,
    )
    policy: episode.Policy[episode.Action] = (
        expert.ExpertPolicy.from_environment(environment)
        if args.policy == _EXPERT
        else learned.load_policy(environment, args.policy)
    )
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            spec = data.sample_spec(
                seed, data.TaskRanges(), data.NoiseLevels()
            )
            result = episode.run_episode(
                environment,
                policy,
                spec.target_pos,
                episode.StopCriteria(),
                spec.base_xy,
                spec.base_bearing,
            )
            error = np.linalg.norm(
                np.asarray(spec.target_pos) - result.final_obs.ee_pos
            )
            print(
                f"seed {seed}: error {error * 1000:.1f} mm, "
                f"{result.stop_message}"
            )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
