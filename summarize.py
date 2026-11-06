"""Summarize trained checkpoints and their evaluations in one table.

Each row is one name: the checkpoint's training record, joined to the
evaluation of the same name. Evaluations named NAME-confirm and
NAME-final fill extra columns of NAME's row. A checkpoint folder of
seed members (s0, s1, ...) is one ensemble row.

Usage:
    python summarize.py [FOLDER] [--csv FILE]

FOLDER is a subfolder of checkpoints/ and evaluations/, such as v1; the
top level is used when it is left out.
"""

import argparse
import csv
import json
import os
import typing

from behavior_cloning import model, paths, trees

# Evaluation name suffixes that fill columns of their own.
_SUFFIXES = ("confirm", "final")
# Done-head width of networks trained before it was a setting.
_OLD_DONE_HIDDEN = 64
_COLUMNS = (
    "name",
    "data",
    "network",
    "weights",
    "best epoch",
    "val loss",
    "train s",
    "reached",
    "2 cm",
    "median mm",
    "late",
    "park s",
    "tol s",
    "decision us",
    "confirm reached",
    "confirm median mm",
    "final reached",
    "final median mm",
)

Row = dict[str, str]


def _read_json(path: str) -> dict[str, typing.Any]:
    """Return the contents of a JSON file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _listing(root: str) -> list[str]:
    """Return a folder's entries in order, or none if it is missing."""
    return sorted(os.listdir(root)) if os.path.isdir(root) else []


def _seconds(value: float | None) -> str:
    """Format a time in seconds, or a dash when there is none."""
    return "-" if value is None else f"{value:.2f}"


def _microseconds(value: float | None) -> str:
    """Format a time in seconds as microseconds, or a dash if none."""
    return "-" if value is None else f"{value * 1e6:.1f}"


def _millimeters(value: float | None) -> str:
    """Format a distance in meters as millimeters, or a dash if none."""
    return "-" if value is None else f"{value * 1000:.1f}"


def _checkpoint(name: str) -> tuple[dict[str, typing.Any], int] | None:
    """Return a saved model's training record and size, or None.

    A network's size is its learned weights, and a tree policy's is
    the leaves in the rounds it uses.
    """
    tree_record = os.path.join(paths.checkpoint_dir(name), trees.RECORD_FILE)
    if os.path.isfile(tree_record):
        record = _read_json(tree_record)
        return record, sum(record["leaves"].values())
    record_path = paths.training_file(name)
    if not os.path.isfile(record_path):
        return None
    weights = model.count_weights(paths.network_file(name))
    return _read_json(record_path), weights


def _network(record: dict[str, typing.Any]) -> str:
    """Describe the model shape of a training record."""
    if "rounds" in record:
        config = record["config"]
        return f"xgb depth {config['max_depth']} h{config['history']}"
    network = record["settings"]["network"]
    done_hidden = network.get("done_hidden", _OLD_DONE_HIDDEN)
    return (
        f"{network['kind']} {network['layers']}x{network['hidden']}"
        f" h{network['history']} d{done_hidden}"
    )


def _train_seconds(record: dict[str, typing.Any]) -> float:
    """Return the training time, to the best epoch for networks (s)."""
    if "train_seconds" in record:
        return sum(record["train_seconds"].values())
    return sum(record["epoch_seconds"][: record["best_epoch"]])


def _training_columns(folder: str) -> dict[str, Row]:
    """Return the training columns of each checkpoint in a folder.

    A checkpoint folder holding seed members instead of a model is
    one ensemble, whose sizes and training times are summed.

    Args:
        folder: Subfolder of checkpoints/, or "" for the top level.

    Returns:
        The columns of each checkpoint, by name.
    """
    rows: dict[str, Row] = {}
    for name in _listing(paths.checkpoint_dir(folder)):
        qualified = os.path.join(folder, name)
        checkpoint = _checkpoint(qualified)
        if checkpoint is not None:
            record, size = checkpoint
            row = {
                "data": "+".join(record["settings"]["datasets"]),
                "network": _network(record),
                "weights": f"{size:,}",
                "train s": f"{_train_seconds(record):.0f}",
            }
            if "best_epoch" in record:
                epochs = len(record["epoch_seconds"])
                row["best epoch"] = f"{record['best_epoch']}/{epochs}"
                row["val loss"] = f"{record['best_val_loss']:.4f}"
            rows[name] = row
            continue
        members = []
        for member in _listing(paths.checkpoint_dir(qualified)):
            if member.startswith("s") and member[1:].isdigit():
                checkpoint = _checkpoint(os.path.join(qualified, member))
                if checkpoint is not None:
                    members.append(checkpoint)
        if members:
            first, _ = members[0]
            size = sum(count for _, count in members)
            train_seconds = sum(
                _train_seconds(record) for record, _ in members
            )
            rows[name] = {
                "data": "+".join(first["settings"]["datasets"]),
                "network": f"{len(members)} x {_network(first)}",
                "weights": f"{size:,}",
                "train s": f"{train_seconds:.0f}",
            }
    return rows


def _evaluation_columns(folder: str) -> dict[str, Row]:
    """Return the evaluation columns of each name in a folder.

    Args:
        folder: Subfolder of evaluations/, or "" for the top level.

    Returns:
        The columns of each name, by name, with confirm and final
        results folded into the row of the name they extend.
    """
    root = os.path.join(paths.EVALUATION_DIR, folder)
    rows: dict[str, Row] = {}
    for file_name in _listing(root):
        name = file_name.removesuffix(".json")
        if name == file_name:
            continue
        summary = _read_json(os.path.join(root, file_name))["summary"]
        episodes = summary["episodes"]
        reached = f"{summary['reached']}/{episodes}"
        median = _millimeters(summary["median_error"])
        for suffix in _SUFFIXES:
            if name.endswith(f"-{suffix}"):
                row = rows.setdefault(name.removesuffix(f"-{suffix}"), {})
                row[f"{suffix} reached"] = reached
                row[f"{suffix} median mm"] = median
                break
        else:
            rows.setdefault(name, {}).update(
                {
                    "reached": reached,
                    "2 cm": f"{summary['close']}/{episodes}",
                    "median mm": median,
                    "late": str(summary.get("late", "-")),
                    "park s": _seconds(summary["mean_park_time"]),
                    "tol s": _seconds(summary["mean_reach_time"]),
                    "decision us": _microseconds(summary.get("decision_time")),
                }
            )
    return rows


def _table(rows: list[Row], columns: list[str]) -> str:
    """Return rows as text with each column padded to one width."""
    widths = [
        max([len(column)] + [len(row.get(column, "-")) for row in rows])
        for column in columns
    ]
    lines = [
        "  ".join(
            (row.get(column, "-") if row else column).ljust(width)
            for column, width in zip(columns, widths, strict=True)
        )
        for row in [{}, *rows]
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize checkpoints and evaluations."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="",
        help="subfolder such as v1 (default: the top level)",
    )
    parser.add_argument("--csv", help="also save the table to this file")
    return parser.parse_args()


def main() -> None:
    """Print the table, and save it as CSV if asked."""
    args = _parse_args()
    training = _training_columns(args.folder)
    evaluations = _evaluation_columns(args.folder)
    rows = [
        {"name": name, **training.get(name, {}), **evaluations.get(name, {})}
        for name in sorted(training.keys() | evaluations.keys())
    ]
    if not rows:
        raise SystemExit("Nothing to summarize")
    columns = [c for c in _COLUMNS if any(c in row for row in rows)]
    print(_table(rows, columns))
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, columns, restval="-")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
