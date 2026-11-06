"""Draw the report's figures from saved evaluations and training runs.

Writes figures/NAME.pdf and figures/NAME.png for each figure:
    progression: median error and 5 mm success along the main line.
    data: success against the amount and kind of training data.
    loss: validation loss against success over the sweeps.
    frameworks: the compared models, with the same data and training.

Usage:
    python plot_results.py
"""

import json
import math
import os
import typing

import matplotlib
import numpy as np
from matplotlib import figure

from behavior_cloning import model, paths, trees

_FIGURE_DIR = os.path.join(paths.REPO_DIR, "figures")
_WIDTH = 3.5  # in, one IEEE column
_HEIGHT = 2.4  # in
_TALL_HEIGHT = 1.6 * _HEIGHT  # in, two stacked panels
_SETS = (("", "Dev"), ("-confirm", "Confirm"))
_MAIN_LINE = (
    ("v2-01-bc", "Clean"),
    ("v2-02-bc-dart", "+DART"),
    ("v2-03-bc-dart-small", "2x64"),
    ("v2-04-bc-dart-d16", "Done 16"),
    ("v2-05-bc-dart-600", "600 ep."),
    ("v2-06-ensemble3", "Ensemble"),
    ("v2-08-replay-gentle", "Replay"),
)
_DATA_SWEEP = (
    ("300 clean", ("data-c300-s0", "data-c300-s1", "data-c300-s2")),
    (
        "100 + 200 DART",
        ("data-c100-d200-s0", "data-c100-d200-s1", "data-c100-d200-s2"),
    ),
    (
        "300 + 300 DART",
        ("data-c300-d300-s0", "data-c300-d300-s1", "data-c300-d300-s2"),
    ),
    ("300 + 600 DART", ("done16", "done16-s1", "done16-s2")),
)
_FRAMEWORKS = (
    ("v2-08-replay-gentle", "MLP h1"),
    ("v2-09-mlp-h10", "MLP h10"),
    ("v2-10-lstm-h10", "LSTM h10"),
    ("v2-12-xgb", "XGB h1"),
)


def _load(path: str) -> dict[str, typing.Any] | None:
    """Return a JSON file's contents, or None if it is missing."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _summary(name: str) -> dict[str, typing.Any] | None:
    """Return an evaluation's summary, or None if it was not run."""
    record = _load(os.path.join(paths.EVALUATION_DIR, f"{name}.json"))
    return None if record is None else record["summary"]


def _reached(name: str) -> float:
    """Return an evaluation's reached share (%), or NaN if not run."""
    summary = _summary(name)
    if summary is None:
        return math.nan
    return 100.0 * summary["reached"] / summary["episodes"]


def _median_mm(name: str) -> float:
    """Return an evaluation's median final error (mm), or NaN."""
    summary = _summary(name)
    return math.nan if summary is None else summary["median_error"] * 1e3


def _size(checkpoint: str) -> str:
    """Describe a checkpoint's size: weights, or leaves for trees."""
    record = _load(
        os.path.join(paths.checkpoint_dir(checkpoint), trees.RECORD_FILE)
    )
    if record is not None:
        return f"{sum(record['leaves'].values()):,}\nleaves"
    weights = model.count_weights(paths.network_file(checkpoint))
    return f"{weights:,}\nweights"


def _progression() -> figure.Figure:
    """Median error and 5 mm success at each main-line step."""
    fig = figure.Figure(figsize=(_WIDTH, _TALL_HEIGHT))
    error_axes, tolerance_axes = fig.subplots(2, 1, sharex=True)
    x = np.arange(len(_MAIN_LINE))
    for offset, (suffix, label) in zip((-0.2, 0.2), _SETS, strict=True):
        error_axes.bar(
            x + offset,
            [_median_mm(name + suffix) for name, _ in _MAIN_LINE],
            0.4,
            label=label,
        )
        tolerance_axes.bar(
            x + offset,
            [_reached(f"{name}-tol5{suffix}") for name, _ in _MAIN_LINE],
            0.4,
            label=label,
        )
    error_axes.axhline(
        _median_mm("v2-00-expert"),
        color="k",
        linestyle="--",
        linewidth=0.8,
        label="Expert",
    )
    error_axes.set_ylabel("Median error (mm)")
    error_axes.legend()
    tolerance_axes.axhline(
        _reached("v2-00-expert-tol5"),
        color="k",
        linestyle="--",
        linewidth=0.8,
    )
    tolerance_axes.set_ylabel("Reached at 5 mm (%)")
    tolerance_axes.set_ylim(0, 105)
    tolerance_axes.set_xticks(
        x, [label for _, label in _MAIN_LINE], rotation=30, ha="right"
    )
    return fig


def _data() -> figure.Figure:
    """Success per data recipe: seed mean as bars, seeds as dots."""
    fig = figure.Figure(figsize=(_WIDTH, _HEIGHT))
    axes = fig.subplots()
    for x, (_, names) in enumerate(_DATA_SWEEP):
        values = [_reached(f"sweep/{name}") for name in names]
        axes.bar(x, float(np.nanmean(values)), 0.6, color="C0")
        axes.scatter([x] * len(values), values, color="k", s=8, zorder=3)
    axes.set_xticks(
        range(len(_DATA_SWEEP)),
        [label for label, _ in _DATA_SWEEP],
        rotation=20,
        ha="right",
    )
    axes.set_ylabel("Reached at 1 cm (%)")
    axes.set_ylim(0, 105)
    return fig


def _loss() -> figure.Figure:
    """Validation loss against success for every sweep network."""
    losses: list[float] = []
    rates: list[float] = []
    for name in sorted(os.listdir(paths.checkpoint_dir("sweep"))):
        record = _load(paths.training_file(f"sweep/{name}"))
        rate = _reached(f"sweep/{name}")
        if record is None or math.isnan(rate):
            continue
        losses.append(record["best_val_loss"])
        rates.append(rate)
    fig = figure.Figure(figsize=(_WIDTH, _HEIGHT))
    axes = fig.subplots()
    axes.scatter(losses, rates, s=10)
    correlation = np.corrcoef(losses, rates)[0, 1]
    axes.set_title(f"Pearson r = {correlation:.2f}")
    axes.set_xlabel("Best validation loss")
    axes.set_ylabel("Reached at 1 cm (%)")
    return fig


def _frameworks() -> figure.Figure:
    """5 mm success and median error of each compared model."""
    fig = figure.Figure(figsize=(_WIDTH, _TALL_HEIGHT))
    tolerance_axes, error_axes = fig.subplots(2, 1, sharex=True)
    x = np.arange(len(_FRAMEWORKS))
    for offset, (suffix, label) in zip((-0.2, 0.2), _SETS, strict=True):
        tolerance_axes.bar(
            x + offset,
            [_reached(f"{name}-tol5{suffix}") for name, _ in _FRAMEWORKS],
            0.4,
            label=label,
        )
        error_axes.bar(
            x + offset,
            [_median_mm(name + suffix) for name, _ in _FRAMEWORKS],
            0.4,
            label=label,
        )
    tolerance_axes.set_ylabel("Reached at 5 mm (%)")
    tolerance_axes.set_ylim(0, 105)
    error_axes.set_ylabel("Median error (mm)")
    error_axes.set_ylim(0, 4)
    error_axes.legend(loc="upper right", ncols=2)
    error_axes.set_xticks(
        x,
        [f"{label}\n{_size(f'{name}/s0')}" for name, label in _FRAMEWORKS],
    )
    return fig


def _save(fig: figure.Figure, name: str) -> None:
    """Save a figure as PDF and PNG under figures/."""
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURE_DIR, f"{name}.{extension}"), dpi=300)


def main() -> None:
    """Draw every figure."""
    matplotlib.rcParams.update({"font.size": 8})
    os.makedirs(_FIGURE_DIR, exist_ok=True)
    for name, draw in (
        ("progression", _progression),
        ("data", _data),
        ("loss", _loss),
        ("frameworks", _frameworks),
    ):
        _save(draw(), name)
        print(f"Wrote figures/{name}.pdf and figures/{name}.png")


if __name__ == "__main__":
    main()
