"""Train the reaching policy on recorded expert episodes.

Holds out a fifth of the episodes for validation, stops once the
validation loss stalls for --patience epochs, and keeps the best
epoch's network.

Writes to checkpoints/NAME:
    network.pt: Best network.
    training.json: Settings, per-epoch losses, and epoch times.
    losses.png: Loss curves.

Usage:
    python train.py NAME --data DATASET [DATASET ...]
        [--kind {mlp,lstm}] [--history H] [--hidden W] [--layers L]
        [--done-hidden D] [--epochs E] [--batch-size B]
        [--learning-rate R] [--patience P] [--seed S] [--threads T]
        [--replay-alpha A] [--replay-beta B]
"""

import argparse
import copy
import dataclasses
import json
import math
import os
import time

import torch
from matplotlib import figure
from torch import nn

from behavior_cloning import features, model, paths, replay

_MAX_GRAD_NORM = 1.0
_EVAL_BATCH_SIZE = 4096


@dataclasses.dataclass(frozen=True)
class TrainSettings:
    """Settings that determine a trained network.

    Attributes:
        datasets: Dataset names under data/.
        network: Network shape.
        epochs: Maximum number of epochs.
        batch_size: Samples per training step.
        learning_rate: Adam step size.
        patience: Epochs without improvement before stopping.
        val_fraction: Share of episodes held out for validation.
        seed: Seed of the split, initial weights, and batch order.
        threads: CPU threads used by PyTorch.
        replay_alpha: Priority exponent of prioritized replay.
        replay_beta: Importance-weight exponent of prioritized replay
            at the first epoch; it rises linearly to 1 at the last.
    """

    datasets: tuple[str, ...]
    network: model.NetworkConfig
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    val_fraction: float
    seed: int
    threads: int
    replay_alpha: float
    replay_beta: float


@dataclasses.dataclass(frozen=True)
class _Samples:
    """Inputs, labels, base-done flags, and history windows of all rows.

    Attributes:
        states: Standardized network inputs.
        labels: Standardized expert labels.
        done: Base-done flags of each row, as 0.0 or 1.0.
        windows: Rows of each row's history window.
    """

    states: torch.Tensor
    labels: torch.Tensor
    done: torch.Tensor
    windows: torch.Tensor

    def batch(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the input windows, labels, and done flags of rows."""
        window_rows = self.windows[rows]
        return (
            self.states.index_select(0, window_rows.flatten()).view(
                *window_rows.shape, -1
            ),
            self.labels.index_select(0, rows),
            self.done.index_select(0, rows),
        )


def _row_losses(
    network: model.PolicyNetwork,
    windows: torch.Tensor,
    labels: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """Return each row's action MSE plus its base-done BCE.

    Args:
        network: Policy being trained.
        windows: Standardized history windows.
        labels: Standardized expert labels.
        done: Base-done flags (0.0 or 1.0).

    Returns:
        Losses of shape (batch,), whose mean is the batch loss.
    """
    actions, done_logits = network(windows)
    action_losses = (actions - labels).square().mean(dim=1)
    done_losses = nn.functional.binary_cross_entropy_with_logits(
        done_logits, done, reduction="none"
    )
    return action_losses + done_losses


def _train_epoch(
    network: model.PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    samples: _Samples,
    rows: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    sampler: replay.Replay | None,
    beta: float,
) -> float:
    """Train for one pass over rows in random order.

    Args:
        network: Network to train.
        optimizer: Optimizer of the network's parameters.
        samples: All rows.
        rows: Training rows.
        batch_size: Samples per step.
        generator: Random generator of the batch order.
        sampler: Prioritized replay, or None to use every row once
            per epoch.
        beta: Importance-weight exponent, unused without replay.

    Returns:
        The mean training loss of the pass.
    """
    network.train()
    parameters = list(network.parameters())
    total = torch.zeros((), dtype=torch.float64)
    positions = (
        torch.randperm(len(rows), generator=generator)
        if sampler is None
        else sampler.sample(generator, beta)
    )
    for batch_positions in positions.split(batch_size):
        losses = _row_losses(network, *samples.batch(rows[batch_positions]))
        mean_loss = losses.mean()
        if sampler is None:
            loss = mean_loss
        else:
            loss = sampler.weighted_mean(losses, batch_positions)
            sampler.update(batch_positions, losses)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.get_total_norm(
            [p.grad for p in parameters if p.grad is not None]
        )
        if grad_norm > _MAX_GRAD_NORM:
            nn.utils.clip_grads_with_norm_(
                parameters, _MAX_GRAD_NORM, grad_norm
            )
        optimizer.step()
        total += mean_loss.detach() * len(batch_positions)
    return total.item() / len(rows)


def _evaluate(
    network: model.PolicyNetwork, samples: _Samples, rows: torch.Tensor
) -> float:
    """Return the mean loss over rows, without training.

    Args:
        network: Network to score.
        samples: All rows.
        rows: Rows to score.

    Returns:
        The mean loss.
    """
    network.eval()
    total = torch.zeros((), dtype=torch.float64)
    with torch.inference_mode():
        for batch_rows in rows.split(_EVAL_BATCH_SIZE):
            losses = _row_losses(network, *samples.batch(batch_rows))
            total += losses.sum()
    return total.item() / len(rows)


def _plot_losses(
    train_losses: list[float], val_losses: list[float], path: str
) -> None:
    """Plot the training and validation loss curves.

    Args:
        train_losses: Training loss of each epoch.
        val_losses: Validation loss of each epoch.
        path: Image file to write.
    """
    epochs = range(1, len(train_losses) + 1)
    loss_figure = figure.Figure(figsize=(6, 4))
    axes = loss_figure.subplots()
    axes.plot(epochs, train_losses, label="Training")
    axes.plot(epochs, val_losses, label="Validation")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    axes.grid(True, alpha=0.3)
    axes.legend()
    loss_figure.tight_layout()
    loss_figure.savefig(path, dpi=150)


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a policy network.")
    parser.add_argument(
        "name", help="checkpoint folder name under checkpoints/"
    )
    parser.add_argument(
        "--data", nargs="+", required=True, help="dataset names under data/"
    )
    parser.add_argument(
        "--kind",
        choices=("mlp", "lstm"),
        default="mlp",
        help="network architecture",
    )
    parser.add_argument(
        "--history", type=int, default=1, help="history window length (steps)"
    )
    parser.add_argument(
        "--hidden", type=int, default=64, help="hidden layer width"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="hidden (MLP) or stacked (LSTM) layers",
    )
    parser.add_argument(
        "--done-hidden",
        type=int,
        default=16,
        help="hidden layer width of the base-done head",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="maximum epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="rows per gradient step"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam step size"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="epochs without improvement before stopping",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--threads", type=int, default=4, help="CPU threads for PyTorch"
    )
    parser.add_argument(
        "--replay-alpha",
        type=float,
        default=0.0,
        help="prioritized replay exponent, 0 for none (default: 0)",
    )
    parser.add_argument(
        "--replay-beta",
        type=float,
        default=0.4,
        help="importance-weight exponent of replay at the first epoch",
    )
    args = parser.parse_args()
    positive = (
        args.history,
        args.hidden,
        args.layers,
        args.done_hidden,
        args.epochs,
        args.batch_size,
        args.patience,
        args.threads,
    )
    if min(positive) < 1 or args.learning_rate <= 0.0:
        parser.error("sizes, counts, and the learning rate must be positive")
    if args.seed < 0:
        parser.error("seed must not be negative")
    if args.replay_alpha < 0.0 or not 0.0 <= args.replay_beta <= 1.0:
        parser.error("replay alpha must not be negative, and beta in [0, 1]")
    if os.path.exists(paths.checkpoint_dir(args.name)):
        parser.error(f"checkpoint {args.name} already exists")
    return args


def main() -> None:
    """Train a network, save the best one, and plot its losses."""
    args = _parse_args()
    settings = TrainSettings(
        datasets=tuple(args.data),
        network=model.NetworkConfig(
            kind=args.kind,
            history=args.history,
            hidden=args.hidden,
            layers=args.layers,
            done_hidden=args.done_hidden,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        val_fraction=features.VAL_FRACTION,
        seed=args.seed,
        threads=args.threads,
        replay_alpha=args.replay_alpha,
        replay_beta=args.replay_beta,
    )
    out_dir = paths.checkpoint_dir(args.name)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(settings.threads)
    generator = torch.Generator().manual_seed(settings.seed)

    states, labels, done, episode_starts = features.load_datasets(
        [paths.dataset_file(name) for name in settings.datasets]
    )
    train_episodes, val_episodes = features.split_episodes(
        len(episode_starts) - 1, settings.val_fraction, settings.seed
    )
    train_rows = torch.from_numpy(
        features.episode_rows(episode_starts, train_episodes)
    )
    val_rows = torch.from_numpy(
        features.episode_rows(episode_starts, val_episodes)
    )
    states_tensor = torch.from_numpy(states)
    labels_tensor = torch.from_numpy(labels)
    network = model.PolicyNetwork(settings.network)
    network.state_scaler.fit(states_tensor[train_rows])
    network.label_scaler.fit(labels_tensor[train_rows])
    samples = _Samples(
        network.state_scaler(states_tensor),
        network.label_scaler(labels_tensor),
        torch.from_numpy(done).float(),
        torch.from_numpy(
            features.window_indices(episode_starts, settings.network.history)
        ),
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=settings.learning_rate, fused=True
    )
    sampler = (
        replay.Replay(len(train_rows), settings.replay_alpha)
        if settings.replay_alpha > 0.0
        else None
    )
    print(
        f"{len(train_episodes)} training and {len(val_episodes)} "
        f"validation episodes, {len(train_rows)} training rows"
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    epoch_seconds: list[float] = []
    best_state = copy.deepcopy(network.state_dict())
    best_loss = math.inf
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        start = time.perf_counter()
        train_losses.append(
            _train_epoch(
                network,
                optimizer,
                samples,
                train_rows,
                settings.batch_size,
                generator,
                sampler,
                replay.beta_schedule(
                    settings.replay_beta, epoch, settings.epochs
                ),
            )
        )
        val_losses.append(_evaluate(network, samples, val_rows))
        epoch_seconds.append(time.perf_counter() - start)
        print(
            f"epoch {epoch:3d}: training {train_losses[-1]:.5f}, "
            f"validation {val_losses[-1]:.5f} ({epoch_seconds[-1]:.1f} s)"
        )
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
        elif epoch - best_epoch >= settings.patience:
            break

    network.load_state_dict(best_state)
    os.makedirs(out_dir)
    model.save_network(network.eval(), paths.network_file(args.name))
    with open(paths.training_file(args.name), "w", encoding="utf-8") as file:
        json.dump(
            {
                "settings": dataclasses.asdict(settings),
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "epoch_seconds": epoch_seconds,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            file,
            indent=2,
        )
    _plot_losses(train_losses, val_losses, os.path.join(out_dir, "losses.png"))
    print(f"Best validation loss {best_loss:.5f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
