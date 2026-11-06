"""Estimate DART noise levels from a trained policy's validation errors.

Following Laskey et al. (2017), the noise on each executed command is
set to the spread of the policy's error on its held-out validation
episodes, optionally scaled. Base noise uses only rows where the base
is still driving, since a parked base gets no noise.

Usage:
    python dart_noise.py CHECKPOINT [--scale K]
"""

import argparse
import json
import os

import torch

from behavior_cloning import features, model, paths


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate DART noise from a policy's errors."
    )
    parser.add_argument("checkpoint", help="trained policy under checkpoints/")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="multiplier on the noise"
    )
    args = parser.parse_args()
    if args.scale <= 0.0:
        parser.error("scale must be positive")
    if not os.path.isfile(paths.network_file(args.checkpoint)):
        parser.error(f"no network checkpoint named {args.checkpoint}")
    return args


def main() -> None:
    """Print the noise levels as collect_data.py options."""
    args = _parse_args()
    with open(paths.training_file(args.checkpoint), encoding="utf-8") as file:
        settings = json.load(file)["settings"]
    network = model.load_network(paths.network_file(args.checkpoint))

    states, labels, done, episode_starts = features.load_datasets(
        [paths.dataset_file(name) for name in settings["datasets"]]
    )
    _, val_episodes = features.split_episodes(
        len(episode_starts) - 1, settings["val_fraction"], settings["seed"]
    )
    rows = features.episode_rows(episode_starts, val_episodes)
    windows = features.window_indices(episode_starts, network.config.history)
    predictions = network.predict(
        torch.from_numpy(states)[torch.from_numpy(windows[rows])]
    )[0].numpy()
    errors = labels[rows] - predictions
    driving = ~done[rows]

    v_noise = args.scale * errors[driving, features.V_LABEL].std()
    w_noise = args.scale * errors[driving, features.W_LABEL].std()
    q_noise = args.scale * errors[:, features.Q_OFFSET_LABELS].std()
    print(f"Validation rows   : {len(rows)} ({driving.sum()} driving)")
    print(f"v noise (m/s)     : {v_noise:.4f}")
    print(f"w noise (rad/s)   : {w_noise:.4f}")
    print(f"q noise (rad)     : {q_noise:.4f}")
    print(
        "collect_data.py options: "
        f"--v-noise {v_noise:.4f} --w-noise {w_noise:.4f} "
        f"--q-noise {q_noise:.4f}"
    )


if __name__ == "__main__":
    main()
