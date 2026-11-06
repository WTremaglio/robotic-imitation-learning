"""Network inputs and labels built from recorded episodes."""

import collections.abc
import json
import os

import numpy as np
import numpy.typing as npt

from behavior_cloning import expert

Float32Array = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]

# Inputs per step, and the goal inputs that the base-done head reads.
STATE_WIDTH = 18
DONE_INPUTS = slice(13, 18)
DONE_WIDTH = DONE_INPUTS.stop - DONE_INPUTS.start

# Labels per step, and their columns.
LABEL_WIDTH = 16
V_LABEL = 0
W_LABEL = 1
Q_OFFSET_LABELS = slice(2, 9)
QD_DES_LABELS = slice(9, 16)

# Share of episodes held out for validation.
VAL_FRACTION = 0.2

_PLANNER = expert.PlannerParams()


def _to_base_frame(
    vectors: npt.NDArray[np.floating],
    cos: npt.NDArray[np.floating],
    sin: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Rotate world-frame vectors about z into the base frame."""
    x = vectors[..., 0]
    y = vectors[..., 1]
    return np.concatenate(
        (
            (cos * x + sin * y)[..., np.newaxis],
            (cos * y - sin * x)[..., np.newaxis],
            vectors[..., 2:],
        ),
        axis=-1,
    )


def state_features(
    base_pos: npt.ArrayLike,
    base_bearing: npt.ArrayLike,
    q_arm: npt.ArrayLike,
    ee_pos: npt.ArrayLike,
    target_pos: npt.ArrayLike,
    standoff: npt.ArrayLike,
) -> Float32Array:
    """Return the network inputs of one or more steps.

    Inputs may hold one step or many along matching leading axes. The
    robot's own speeds are left out. The target is never directly above
    the base, so its direction is always defined.

    Args:
        base_pos: Base position, world frame (m).
        base_bearing: Base bearing (rad), without a trailing axis.
        q_arm: Arm joint angles (rad).
        ee_pos: End-effector position, world frame (m).
        target_pos: Target end-effector position, world frame (m).
        standoff: Horizontal distance the base parks from the target
            (m), without a trailing axis.

    Returns:
        Inputs of width STATE_WIDTH: target and end-effector error in
        the base frame, arm joint angles, the standoff point in the
        base frame, the distance still to cover (negative when too
        close), and cos and sin of the heading error to the target.
    """
    base_pos = np.asarray(base_pos)
    target_pos = np.asarray(target_pos)
    bearing = np.asarray(base_bearing)
    cos = np.cos(bearing)
    sin = np.sin(bearing)
    target = _to_base_frame(target_pos - base_pos, cos, sin)
    flat = target[..., :2]
    distance = np.linalg.norm(flat, axis=-1, keepdims=True)
    direction = flat / distance
    standoff = np.asarray(standoff)[..., np.newaxis]
    return np.concatenate(
        (
            target,
            _to_base_frame(target_pos - np.asarray(ee_pos), cos, sin),
            q_arm,
            flat - standoff * direction,
            distance - standoff,
            direction,
        ),
        axis=-1,
        dtype=np.float32,
    )


def action_labels(
    v_cmd: npt.ArrayLike,
    w_cmd: npt.ArrayLike,
    q_des_arm: npt.ArrayLike,
    q_arm: npt.ArrayLike,
    qd_des_arm: npt.ArrayLike,
) -> Float32Array:
    """Return the network labels of recorded steps.

    Args:
        v_cmd: Base forward speeds (m/s), shape (steps, 1).
        w_cmd: Base turning rates (rad/s), shape (steps, 1).
        q_des_arm: Arm joint references (rad).
        q_arm: Measured arm joint angles (rad).
        qd_des_arm: Arm joint velocity references (rad/s).

    Returns:
        Labels of width LABEL_WIDTH.
    """
    return np.concatenate(
        (
            v_cmd,
            w_cmd,
            np.subtract(q_des_arm, q_arm),
            qd_des_arm,
        ),
        axis=-1,
        dtype=np.float32,
    )


def dataset_arrays(
    dataset: collections.abc.Mapping[str, npt.NDArray[np.generic]],
    episodes: npt.NDArray[np.bool_] | None = None,
) -> tuple[Float32Array, Float32Array, npt.NDArray[np.bool_], IndexArray]:
    """Return the inputs, labels, and base-done flags of every row.

    Args:
        dataset: Merged dataset from data.merge_episodes.
        episodes: Mask of the episodes to keep, or None for all.

    Returns:
        A tuple (states, labels, base_done, lengths) with one entry per
        kept row, and the length of each kept episode.
    """
    lengths = np.diff(dataset["episode_starts"])
    target_pos = dataset["target_pos"]
    rows: slice | npt.NDArray[np.bool_] = slice(None)
    if episodes is not None:
        rows = np.repeat(episodes, lengths)
        lengths = lengths[episodes]
        target_pos = target_pos[episodes]
    q_arm = dataset["q_arm"][rows]
    standoffs = expert.standoff_distance(
        np.asarray(target_pos[:, 2], dtype=np.float64), _PLANNER
    )
    states = state_features(
        dataset["base_pos"][rows],
        dataset["base_bearing"][rows, 0],
        q_arm,
        dataset["ee_pos"][rows],
        np.repeat(target_pos, lengths, axis=0),
        np.repeat(standoffs, lengths),
    )
    labels = action_labels(
        dataset["v_cmd"][rows],
        dataset["w_cmd"][rows],
        dataset["q_des_arm"][rows],
        q_arm,
        dataset["qd_des_arm"][rows],
    )
    done = dataset["base_done"][rows].astype(np.bool_)
    return states, labels, done, lengths


def load_datasets(
    paths: collections.abc.Sequence[str | os.PathLike[str]],
) -> tuple[Float32Array, Float32Array, npt.NDArray[np.bool_], IndexArray]:
    """Load datasets and join them into one, e.g. expert and DART data.

    Episodes the expert failed while driving are left out. DAgger
    datasets keep every episode, since the expert's labels stay valid
    wherever the learner drove.

    Args:
        paths: Dataset files written by collect_data.py.

    Returns:
        A tuple (states, labels, base_done, episode_starts) covering the
        kept episodes of every dataset, in order.
    """
    states: list[Float32Array] = []
    labels: list[Float32Array] = []
    done: list[npt.NDArray[np.bool_]] = []
    episode_lengths: list[IndexArray] = []
    for path in paths:
        settings_path = os.path.join(os.path.dirname(path), "settings.json")
        with open(settings_path, encoding="utf-8") as file:
            learner_drove = json.load(file).get("policy") is not None
        with np.load(path) as dataset:
            reached = dataset["reached"]
            keep = None if learner_drove or reached.all() else reached
            dataset_states, dataset_labels, dataset_done, lengths = (
                dataset_arrays(dataset, keep)
            )
        states.append(dataset_states)
        labels.append(dataset_labels)
        done.append(dataset_done)
        episode_lengths.append(lengths)
    all_lengths = np.concatenate(episode_lengths)
    episode_starts = np.zeros(len(all_lengths) + 1, dtype=np.int64)
    np.cumsum(all_lengths, out=episode_starts[1:])
    return (
        np.concatenate(states),
        np.concatenate(labels),
        np.concatenate(done),
        episode_starts,
    )


def window_indices(episode_starts: IndexArray, history: int) -> IndexArray:
    """Return, for each row, the rows of its history window.

    Each window lists the row and the history - 1 rows before it,
    oldest first. Windows never reach into the previous episode; at
    the start of an episode they repeat its first row instead.

    Args:
        episode_starts: First row of each episode, plus the total row
            count, as from data.merge_episodes.
        history: Steps per window.

    Returns:
        Row indices of shape (rows, history).
    """
    starts = np.asarray(episode_starts, dtype=np.int64)
    first_row = np.repeat(starts[:-1], np.diff(starts))
    rows = np.arange(starts[-1], dtype=np.int64)
    offsets = np.arange(history - 1, -1, -1, dtype=np.int64)
    return np.maximum(rows[:, np.newaxis] - offsets, first_row[:, np.newaxis])


def split_episodes(
    num_episodes: int, val_fraction: float, seed: int
) -> tuple[IndexArray, IndexArray]:
    """Split episodes at random into training and validation sets.

    Args:
        num_episodes: Number of episodes.
        val_fraction: Share of episodes for validation.
        seed: Seed of the split.

    Returns:
        A tuple (train, val) of sorted episode indices.
    """
    order = np.random.default_rng(seed).permutation(num_episodes)
    num_val = max(1, round(num_episodes * val_fraction))
    return np.sort(order[num_val:]), np.sort(order[:num_val])


def episode_rows(
    episode_starts: IndexArray, episodes: IndexArray
) -> IndexArray:
    """Return the rows of the given episodes, in episode order.

    Args:
        episode_starts: First row of each episode, plus the total row
            count, as from data.merge_episodes.
        episodes: Episode indices.

    Returns:
        Row indices.
    """
    starts = np.asarray(episode_starts, dtype=np.int64)
    first = starts[episodes]
    lengths = starts[episodes + 1] - first
    # Each episode's rows continue from its first row, wherever its
    # block starts in the output.
    block_starts = np.cumsum(lengths) - lengths
    return np.repeat(first - block_starts, lengths) + np.arange(lengths.sum())
