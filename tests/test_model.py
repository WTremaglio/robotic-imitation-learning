"""Tests of the policy networks."""

import os
import tempfile
import unittest

import numpy as np
import torch

from behavior_cloning import features, model

_KINDS: tuple[model.NetworkKind, ...] = ("mlp", "lstm")
_BATCH = 4


def _windows(history: int) -> torch.Tensor:
    """Return random input windows."""
    return torch.randn(_BATCH, history, features.STATE_WIDTH)


class ScalerTest(unittest.TestCase):
    """The scaler standardizes values and undoes it exactly."""

    def test_fit_standardizes(self) -> None:
        """Fitted values have zero mean and unit spread."""
        values = torch.randn(1000, 3) * 5.0 + 2.0
        scaler = model.Scaler(3)
        scaler.fit(values)
        scaled = scaler(values)
        torch.testing.assert_close(scaled.mean(dim=0), torch.zeros(3))
        torch.testing.assert_close(scaled.std(dim=0), torch.ones(3))

    def test_inverse_undoes_scaling(self) -> None:
        """Scaling and then inverting returns the original values."""
        values = torch.randn(10, 3) * 5.0 + 2.0
        scaler = model.Scaler(3)
        scaler.fit(values)
        torch.testing.assert_close(scaler.inverse(scaler(values)), values)


class PolicyNetworkTest(unittest.TestCase):
    """Both network kinds give outputs of the documented shape."""

    def test_output_shapes(self) -> None:
        """Actions and done logits have one row per window."""
        for kind in _KINDS:
            with self.subTest(kind=kind):
                config = model.NetworkConfig(kind=kind)
                actions, done_logits = model.PolicyNetwork(config)(
                    _windows(config.history)
                )
                self.assertEqual(actions.shape, (_BATCH, features.LABEL_WIDTH))
                self.assertEqual(done_logits.shape, (_BATCH,))

    def test_save_and_load_keep_outputs(self) -> None:
        """A loaded network gives the same outputs as the saved one."""
        for kind in _KINDS:
            with self.subTest(kind=kind):
                config = model.NetworkConfig(kind=kind)
                network = model.PolicyNetwork(config).eval()
                network.state_scaler.fit(
                    torch.randn(100, features.STATE_WIDTH)
                )
                windows = _windows(config.history)
                with tempfile.TemporaryDirectory() as folder:
                    path = os.path.join(folder, "network.pt")
                    model.save_network(network, path)
                    loaded = model.load_network(path)
                with torch.inference_mode():
                    expected = network(windows)
                    actual = loaded(windows)
                torch.testing.assert_close(actual, expected)

    def test_predict_returns_physical_units(self) -> None:
        """Predictions match forward with manual scaling."""
        network = model.PolicyNetwork(model.NetworkConfig()).eval()
        network.state_scaler.fit(torch.randn(100, features.STATE_WIDTH) * 3)
        network.label_scaler.fit(torch.randn(100, features.LABEL_WIDTH) * 3)
        windows = _windows(network.config.history)
        with torch.inference_mode():
            actions, done_logits = network(network.state_scaler(windows))
        expected = (
            network.label_scaler.inverse(actions),
            torch.sigmoid(done_logits),
        )
        torch.testing.assert_close(network.predict(windows), expected)

    def test_done_reads_only_the_newest_goal_inputs(self) -> None:
        """Only the newest goal inputs affect the done logit."""
        goal = features.DONE_INPUTS
        for kind in _KINDS:
            with self.subTest(kind=kind):
                config = model.NetworkConfig(kind=kind)
                network = model.PolicyNetwork(config).eval()
                windows = _windows(config.history)
                changed = windows + torch.randn_like(windows)
                changed[:, -1, goal] = windows[:, -1, goal]
                with torch.inference_mode():
                    _, expected = network(windows)
                    _, actual = network(changed)
                torch.testing.assert_close(actual, expected)

    def test_frozen_mlp_matches_predict(self) -> None:
        """The NumPy copy of an MLP predicts what the network does."""
        config = model.NetworkConfig(history=3)
        network = model.PolicyNetwork(config).eval()
        network.state_scaler.fit(torch.randn(100, features.STATE_WIDTH) * 3)
        network.label_scaler.fit(torch.randn(100, features.LABEL_WIDTH) * 3)
        frozen = model.FrozenMlp(network)
        windows = _windows(config.history)
        actions, done_probs = network.predict(windows)
        for row in range(_BATCH):
            frozen_actions, frozen_done = frozen.predict(windows[row].numpy())
            np.testing.assert_allclose(
                frozen_actions, actions[row].numpy(), rtol=1e-5, atol=1e-5
            )
            self.assertAlmostEqual(frozen_done, done_probs[row].item(), 6)

    def test_frozen_lstm_is_refused(self) -> None:
        """Only MLPs can be frozen into NumPy."""
        network = model.PolicyNetwork(model.NetworkConfig(kind="lstm"))
        with self.assertRaises(ValueError):
            model.FrozenMlp(network)

    def test_ensemble_of_copies_matches_one(self) -> None:
        """Averaging a network with itself changes nothing."""
        network = model.PolicyNetwork(model.NetworkConfig()).eval()
        windows = _windows(network.config.history)
        expected = network.predict(windows)
        actual = model.Ensemble([network, network]).predict(windows)
        torch.testing.assert_close(actual, expected)

    def test_ensemble_refuses_mixed_histories(self) -> None:
        """Members with different histories cannot be averaged."""
        with self.assertRaises(ValueError):
            model.Ensemble(
                [
                    model.PolicyNetwork(model.NetworkConfig(history=1)),
                    model.PolicyNetwork(model.NetworkConfig(history=3)),
                ]
            )

    def test_frozen_ensemble_matches_predict(self) -> None:
        """The NumPy copy of an ensemble predicts what it does."""
        networks = []
        for _ in range(3):
            network = model.PolicyNetwork(model.NetworkConfig()).eval()
            network.label_scaler.fit(
                torch.randn(100, features.LABEL_WIDTH) * 3
            )
            networks.append(network)
        ensemble = model.Ensemble(networks)
        windows = _windows(ensemble.config.history)
        actions, done_probs = ensemble.predict(windows)
        frozen = model.FrozenEnsemble(ensemble)
        for row in range(_BATCH):
            frozen_actions, frozen_done = frozen.predict(windows[row].numpy())
            np.testing.assert_allclose(
                frozen_actions, actions[row].numpy(), rtol=1e-5, atol=1e-5
            )
            self.assertAlmostEqual(frozen_done, done_probs[row].item(), 6)
