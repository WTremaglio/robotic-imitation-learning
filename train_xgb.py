"""Train a gradient-boosted tree policy on recorded demonstrations.

Fits the three XGBoost models of trees.TreePolicy on the same inputs,
labels, and episode split as train.py: the done classifier on every
row, the base regressor on rows before parking, and the arm regressor
on rows after. Each stops early on its validation loss.

Writes checkpoints/NAME/: the models and trees.json, which holds the
settings, the rounds and leaves used, and the validation results.

Usage:
    python train_xgb.py NAME --data DATASET [DATASET ...] [--seed S]
        [--max-depth D] [--learning-rate R] [--rounds N]
        [--early-stopping N] [--subsample F] [--colsample F]
        [--threads T]
"""

import argparse
import os
import time

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from behavior_cloning import features, paths, trees

# Time between recorded rows: one row per 100 Hz policy step (s).
_ROW_SECONDS = 0.01
# Variance below which a label's R^2 is undefined.
_MIN_VARIANCE = 1e-12


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a gradient-boosted tree policy."
    )
    parser.add_argument(
        "name", help="checkpoint folder name under checkpoints/"
    )
    parser.add_argument(
        "--data", nargs="+", required=True, help="dataset names under data/"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--max-depth", type=int, default=8, help="maximum tree depth"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="shrinkage on each tree",
    )
    parser.add_argument(
        "--rounds", type=int, default=2000, help="maximum boosting rounds"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="rounds without improvement before stopping",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="share of rows per tree",
    )
    parser.add_argument(
        "--colsample",
        type=float,
        default=0.8,
        help="share of inputs per tree",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads (default: one per CPU core)",
    )
    args = parser.parse_args()
    counts = (args.max_depth, args.rounds, args.early_stopping, args.threads)
    if min(counts) < 1 or args.learning_rate <= 0.0:
        parser.error("counts and the learning rate must be positive")
    if not (0.0 < args.subsample <= 1.0 and 0.0 < args.colsample <= 1.0):
        parser.error("subsample and colsample must be in (0, 1]")
    if args.seed < 0:
        parser.error("seed must not be negative")
    if os.path.exists(paths.checkpoint_dir(args.name)):
        parser.error(f"checkpoint {args.name} already exists")
    return args


def _fit(
    params: dict[str, object],
    args: argparse.Namespace,
    train: xgb.DMatrix,
    val: xgb.DMatrix,
) -> tuple[xgb.Booster, int, float]:
    """Fit one booster, stopping early on the validation loss.

    Args:
        params: XGBoost parameters for this part.
        args: Command-line arguments.
        train: Training rows and labels.
        val: Validation rows and labels.

    Returns:
        The booster, the rounds it uses, and the training time (s).
    """
    start = time.perf_counter()
    booster = xgb.train(
        params,
        train,
        num_boost_round=args.rounds,
        evals=[(val, "val")],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=False,
    )
    return booster, booster.best_iteration + 1, time.perf_counter() - start


def _leaves(booster: xgb.Booster, rounds: int) -> int:
    """Return the number of leaves in the rounds a booster uses."""
    return sum(tree.count("leaf=") for tree in booster[:rounds].get_dump())


def _regression_metrics(
    labels: npt.NDArray[np.float32], predictions: npt.NDArray[np.float32]
) -> dict[str, list[float]]:
    """Return the RMSE and R^2 of each predicted label column."""
    mse = np.mean((predictions - labels) ** 2, axis=0)
    variance = labels.var(axis=0)
    r2 = np.where(
        variance > _MIN_VARIANCE,
        1.0 - mse / np.maximum(variance, _MIN_VARIANCE),
        np.nan,
    )
    return {"rmse": np.sqrt(mse).tolist(), "r2": r2.tolist()}


def _park_lags(
    done_probs: npt.NDArray[np.float32],
    done: npt.NDArray[np.bool_],
    episode_starts: npt.NDArray[np.int64],
    episodes: npt.NDArray[np.int64],
) -> list[float]:
    """Return how late the classifier first judges each episode parked.

    Args:
        done_probs: Predicted base-done probability of each row.
        done: Expert base-done flag of each row.
        episode_starts: First row of each episode, then the row count.
        episodes: Episodes the rows come from, in order.

    Returns:
        Seconds from the expert parking to the first probability above
        one half, for each episode in which the expert parks: negative
        if early, and inf if the classifier never fires.
    """
    lags = []
    row = 0
    for episode in episodes:
        end = row + episode_starts[episode + 1] - episode_starts[episode]
        parked = np.flatnonzero(done[row:end])
        fired = np.flatnonzero(done_probs[row:end] > 0.5)
        row = end
        if parked.size:
            lags.append(
                (fired[0] - parked[0]) * _ROW_SECONDS
                if fired.size
                else float("inf")
            )
    return lags


def main() -> None:
    """Train the three boosters, report them, and save the policy."""
    args = _parse_args()
    states, labels, done, episode_starts = features.load_datasets(
        [paths.dataset_file(name) for name in args.data]
    )
    train_episodes, val_episodes = features.split_episodes(
        len(episode_starts) - 1, features.VAL_FRACTION, args.seed
    )
    train_rows = features.episode_rows(episode_starts, train_episodes)
    val_rows = features.episode_rows(episode_starts, val_episodes)
    goal = features.DONE_INPUTS
    common = {
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": args.threads,
    }
    boosters: dict[str, xgb.Booster] = {}
    rounds: dict[str, int] = {}
    seconds: dict[str, float] = {}

    val_goal = xgb.DMatrix(
        states[val_rows][:, goal], label=done[val_rows].astype(np.float32)
    )
    boosters["done"], rounds["done"], seconds["done"] = _fit(
        {**common, "objective": "binary:logistic", "eval_metric": "logloss"},
        args,
        xgb.DMatrix(
            states[train_rows][:, goal],
            label=done[train_rows].astype(np.float32),
        ),
        val_goal,
    )
    done_probs = boosters["done"].predict(
        val_goal, iteration_range=(0, rounds["done"])
    )
    lags = np.array(
        _park_lags(done_probs, done[val_rows], episode_starts, val_episodes)
    )
    fired = lags[np.isfinite(lags)]
    done_metrics = {
        "accuracy": float(np.mean((done_probs > 0.5) == done[val_rows])),
        "fired": int(fired.size),
        "parked_episodes": int(lags.size),
        "median_lag": float(np.median(fired)) if fired.size else None,
    }
    metrics: dict[str, object] = {"done": done_metrics}
    print(
        f"done : accuracy {done_metrics['accuracy']:.4f}, fired in "
        f"{fired.size}/{lags.size} parking episodes, median lag "
        f"{done_metrics['median_lag']} s"
    )

    regression = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    for part, columns, parked in (
        ("base", trees.BASE_LABELS, False),
        ("arm", trees.ARM_LABELS, True),
    ):
        part_train = train_rows[done[train_rows] == parked]
        part_val = val_rows[done[val_rows] == parked]
        val = xgb.DMatrix(states[part_val], label=labels[part_val, columns])
        boosters[part], rounds[part], seconds[part] = _fit(
            {**common, **regression},
            args,
            xgb.DMatrix(states[part_train], label=labels[part_train, columns]),
            val,
        )
        predictions = boosters[part].predict(
            val, iteration_range=(0, rounds[part])
        )
        part_metrics = _regression_metrics(
            labels[part_val, columns], predictions.reshape(len(part_val), -1)
        )
        metrics[part] = part_metrics
        print(
            f"{part:<5}: mean validation R^2 "
            f"{np.nanmean(part_metrics['r2']):.3f}"
        )

    leaves = {part: _leaves(boosters[part], rounds[part]) for part in rounds}
    for part in trees.PARTS:
        print(
            f"{part:<5}: {rounds[part]} rounds, {leaves[part]:,} leaves, "
            f"{seconds[part]:.0f} s"
        )
    trees.save_policy(
        paths.checkpoint_dir(args.name),
        boosters,
        rounds,
        trees.TreeConfig(max_depth=args.max_depth),
        {
            "settings": {
                "datasets": args.data,
                **{
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"name", "data"}
                },
            },
            "leaves": leaves,
            "train_seconds": seconds,
            "val_metrics": metrics,
        },
    )
    print(f"Saved checkpoints/{args.name}")


if __name__ == "__main__":
    main()
