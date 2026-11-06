"""Policy networks that map a window of recent states to actions."""

import collections.abc
import dataclasses
import math
import os
import typing

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from behavior_cloning import features

NetworkKind = typing.Literal["mlp", "lstm"]

# Lower bound on the scale, for values that never vary.
_MIN_STD = 1e-6
# Saved buffers that are scaling statistics, not learned weights.
_SCALERS = ("state_scaler.", "label_scaler.")


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """Shape of a policy network.

    Attributes:
        kind: "mlp" reads the whole window at once; "lstm" reads it
            one step at a time.
        history: Steps per input window.
        hidden: Width of each hidden layer.
        layers: Number of hidden layers, or of stacked LSTM layers.
        done_hidden: Width of the hidden layers in the base-done head.
    """

    kind: NetworkKind = "mlp"
    history: int = 1
    hidden: int = 64
    layers: int = 2
    done_hidden: int = 16


class Scaler(nn.Module):
    """Scales values to zero mean and unit spread, and back.

    Its statistics are buffers, saved with the network.
    """

    def __init__(self, width: int) -> None:
        """Create a scaler that leaves values unchanged until fitted.

        Args:
            width: Number of values per sample.
        """
        super().__init__()
        self.mean = nn.Buffer(torch.zeros(width))
        self.std = nn.Buffer(torch.ones(width))

    def fit(self, values: torch.Tensor) -> None:
        """Set the statistics from samples.

        Args:
            values: Samples of shape (count, width).
        """
        self.mean.copy_(values.mean(dim=0))
        self.std.copy_(values.std(dim=0).clamp_min(_MIN_STD))

    @typing.override
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return the scaled values."""
        return (values - self.mean) / self.std

    def inverse(self, scaled: torch.Tensor) -> torch.Tensor:
        """Return the values that scale to scaled."""
        return scaled * self.std + self.mean


class _LstmBody(nn.Module):
    """Stacked LSTM that reads a window and keeps its last output."""

    def __init__(self, config: NetworkConfig) -> None:
        """Create the LSTM layers.

        Args:
            config: Network shape.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            features.STATE_WIDTH,
            config.hidden,
            num_layers=config.layers,
            batch_first=True,
        )

    @typing.override
    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Return the output after the last step of each window."""
        outputs, _ = self.lstm(windows)
        return outputs[:, -1]


def _mlp_body(config: NetworkConfig) -> nn.Sequential:
    """Return hidden layers that read a flattened window."""
    layers: list[nn.Module] = [nn.Flatten()]
    width = config.history * features.STATE_WIDTH
    for _ in range(config.layers):
        layers += [nn.Linear(width, config.hidden), nn.ReLU()]
        width = config.hidden
    return nn.Sequential(*layers)


def _done_head(config: NetworkConfig) -> nn.Sequential:
    """Return the base-done head, which reads the goal columns only."""
    return nn.Sequential(
        nn.Linear(features.DONE_WIDTH, config.done_hidden),
        nn.ReLU(),
        nn.Linear(config.done_hidden, config.done_hidden),
        nn.ReLU(),
        nn.Linear(config.done_hidden, 1),
    )


class PolicyNetwork(nn.Module):
    """Policy mapping a history window to the next action."""

    def __init__(self, config: NetworkConfig) -> None:
        """Create an untrained network.

        Args:
            config: Network shape.
        """
        super().__init__()
        self.config = config
        self.state_scaler = Scaler(features.STATE_WIDTH)
        self.label_scaler = Scaler(features.LABEL_WIDTH)
        self.body = (
            _mlp_body(config) if config.kind == "mlp" else _LstmBody(config)
        )
        self.head = nn.Linear(config.hidden, features.LABEL_WIDTH)
        self.done_head = _done_head(config)

    @typing.override
    def forward(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict standardized actions and base-done logits.

        Args:
            windows: Standardized states, shape
                (batch, history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_logits) of shapes
            (batch, LABEL_WIDTH) and (batch,).
        """
        actions = self.head(self.body(windows))
        done_logits = self.done_head(windows[:, -1, features.DONE_INPUTS])
        return actions, done_logits.squeeze(1)

    @torch.inference_mode()
    def predict(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict physical actions and base-done probabilities.

        Args:
            windows: States in physical units, shape
                (batch, history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_probs) of shapes
            (batch, LABEL_WIDTH) and (batch,).
        """
        actions, done_logits = self(self.state_scaler(windows))
        return self.label_scaler.inverse(actions), torch.sigmoid(done_logits)


def save_network(network: PolicyNetwork, path: str | os.PathLike[str]) -> None:
    """Save a network with its shape and scaling.

    Args:
        network: Network to save.
        path: File to write.
    """
    torch.save(
        {
            "config": dataclasses.asdict(network.config),
            "state": network.state_dict(),
        },
        path,
    )


def load_network(path: str | os.PathLike[str]) -> PolicyNetwork:
    """Load a saved network, ready to run.

    Args:
        path: File written by save_network.

    Returns:
        The network, in evaluation mode.
    """
    checkpoint = torch.load(path, weights_only=True)
    # Early checkpoints record done_hidden only in their weight shapes.
    config = {
        "done_hidden": checkpoint["state"]["done_head.0.weight"].shape[0],
        **checkpoint["config"],
    }
    network = PolicyNetwork(NetworkConfig(**config))
    network.load_state_dict(checkpoint["state"])
    return network.eval()


def count_weights(path: str | os.PathLike[str]) -> int:
    """Return the number of learned weights in a saved network."""
    state = torch.load(path, weights_only=True)["state"]
    return sum(
        tensor.numel()
        for key, tensor in state.items()
        if not key.startswith(_SCALERS)
    )


class Ensemble(nn.Module):
    """Policy networks whose predictions are averaged.

    Attributes:
        members: The averaged networks, registered as submodules.
        config: Shape of the first member; all share its history.
    """

    def __init__(
        self, networks: collections.abc.Sequence[PolicyNetwork]
    ) -> None:
        """Group trained networks into one policy.

        Args:
            networks: Networks with the same history length.

        Raises:
            ValueError: If there are no networks, or their histories
                differ.
        """
        super().__init__()
        if not networks:
            raise ValueError("An ensemble needs at least one network")
        if len({network.config.history for network in networks}) > 1:
            raise ValueError("Ensemble members must share a history")
        self.members = nn.ModuleList(networks)
        self._networks = tuple(networks)
        self.config = networks[0].config

    @property
    def networks(self) -> tuple[PolicyNetwork, ...]:
        """The averaged networks, in order."""
        return self._networks

    @torch.inference_mode()
    def predict(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average the members' physical actions and done probabilities.

        Args:
            windows: States in physical units, shape
                (batch, history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_probs) of shapes (batch, LABEL_WIDTH)
            and (batch,).
        """
        predictions = [network.predict(windows) for network in self._networks]
        return (
            torch.stack([actions for actions, _ in predictions]).mean(dim=0),
            torch.stack([done for _, done in predictions]).mean(dim=0),
        )


def _linears(
    state: dict[str, npt.NDArray[np.float32]], prefix: str
) -> list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
    """Return the weights and biases of a Sequential's linear layers.

    Args:
        state: Network state as NumPy arrays.
        prefix: Name of the Sequential, such as "body".

    Returns:
        A (weight, bias) pair per linear layer, in order.
    """
    indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith(f"{prefix}.") and key.endswith(".weight")
    )
    return [
        (state[f"{prefix}.{i}.weight"], state[f"{prefix}.{i}.bias"])
        for i in indices
    ]


def _sigmoid(logit: float) -> float:
    """Return the logistic sigmoid of logit without overflow."""
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1.0 + exp)


def _fold_scale(
    weight: npt.NDArray[np.float32], std: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Return a layer's weights for inputs not yet divided by std."""
    return (weight.astype(np.float64) / std).astype(np.float32)


class FrozenMlp:
    """A trained MLP's weights in NumPy, for one decision at a time.

    The input spreads are folded into the first layer of the body and
    of the done head, and the label scaling into the action head, so a
    decision only subtracts the input means before the linear layers.
    """

    def __init__(self, network: PolicyNetwork) -> None:
        """Copy the weights and scaling of a trained MLP.

        Args:
            network: Trained network of kind "mlp".

        Raises:
            ValueError: If the network is not an MLP.
        """
        if network.config.kind != "mlp":
            raise ValueError("Only MLP networks can be frozen")
        state = {
            key: value.detach().numpy().astype(np.float32)
            for key, value in network.state_dict().items()
        }
        state_std = state["state_scaler.std"]
        label_mean = state["label_scaler.mean"].astype(np.float64)
        label_std = state["label_scaler.std"].astype(np.float64)
        history = network.config.history
        self._state_mean = np.tile(state["state_scaler.mean"], history)
        (weight, bias), *body = _linears(state, "body")
        self._body = [
            (_fold_scale(weight, np.tile(state_std, history)), bias),
            *body,
        ]
        self._head_weight = (
            state["head.weight"] * label_std[:, np.newaxis]
        ).astype(np.float32)
        self._head_bias = (state["head.bias"] * label_std + label_mean).astype(
            np.float32
        )
        (weight, bias), *done_hidden, (done_weight, done_bias) = _linears(
            state, "done_head"
        )
        self._done_hidden = [
            (_fold_scale(weight, state_std[features.DONE_INPUTS]), bias),
            *done_hidden,
        ]
        self._done_weight = done_weight[0]
        self._done_bias = float(done_bias[0])

    def predict(
        self, window: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Predict physical actions and the base-done probability.

        Args:
            window: States in physical units, shape
                (history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_prob) of the actions, shape
            (LABEL_WIDTH,), and the base-done probability.
        """
        x = window.reshape(-1) - self._state_mean
        done = x[-features.STATE_WIDTH :][features.DONE_INPUTS]
        for weight, bias in self._body:
            x = np.maximum(weight @ x + bias, 0.0)
        actions = self._head_weight @ x + self._head_bias
        for weight, bias in self._done_hidden:
            done = np.maximum(weight @ done + bias, 0.0)
        logit = float(self._done_weight @ done) + self._done_bias
        return actions.astype(np.float64), _sigmoid(logit)


class FrozenEnsemble:
    """NumPy copies of an ensemble's MLPs, with predictions averaged."""

    def __init__(self, ensemble: Ensemble) -> None:
        """Copy every member's weights and scaling.

        Args:
            ensemble: Ensemble whose members are all MLPs.

        Raises:
            ValueError: If a member is not an MLP.
        """
        self._members = tuple(
            FrozenMlp(network) for network in ensemble.networks
        )

    def predict(
        self, window: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Average the members' physical actions and done probabilities.

        Args:
            window: States in physical units, shape
                (history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_prob) of the averaged actions, shape
            (LABEL_WIDTH,), and the averaged base-done probability.
        """
        predictions = [member.predict(window) for member in self._members]
        actions = np.stack([actions for actions, _ in predictions])
        done_prob = math.fsum(done for _, done in predictions)
        return actions.mean(axis=0), done_prob / len(predictions)
