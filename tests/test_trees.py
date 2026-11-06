"""Tests of the gradient-boosted tree policy."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from behavior_cloning import features, paths, trees

_ROWS = 64
_ROUNDS = 3


def _boosters() -> dict[str, xgb.Booster]:
    """Return small boosters for each part, trained on random rows."""
    rng = np.random.default_rng(0)
    states = rng.normal(size=(_ROWS, features.STATE_WIDTH))
    labels = rng.normal(size=(_ROWS, features.LABEL_WIDTH))
    done = (states[:, 0] > 0.0).astype(np.float32)
    params = {"max_depth": 2, "nthread": 1, "seed": 0}
    return {
        "done": xgb.train(
            {**params, "objective": "binary:logistic"},
            xgb.DMatrix(states[:, features.DONE_INPUTS], label=done),
            _ROUNDS,
        ),
        "base": xgb.train(
            params,
            xgb.DMatrix(states, label=labels[:, trees.BASE_LABELS]),
            _ROUNDS,
        ),
        "arm": xgb.train(
            params,
            xgb.DMatrix(states, label=labels[:, trees.ARM_LABELS]),
            _ROUNDS,
        ),
    }


def _policy() -> trees.TreePolicy:
    """Return a tree policy built from small boosters."""
    return trees.TreePolicy(
        _boosters(), dict.fromkeys(trees.PARTS, _ROUNDS), trees.TreeConfig()
    )


def _window() -> npt.NDArray[np.float32]:
    """Return a one-step input window."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(1, features.STATE_WIDTH)).astype(np.float32)


class TreePolicyTest(unittest.TestCase):
    """Tree policies predict like networks and survive saving."""

    def test_output_shapes(self) -> None:
        """Actions have LABEL_WIDTH values and done is a probability."""
        actions, done_prob = _policy().predict(_window())
        self.assertEqual(actions.shape, (features.LABEL_WIDTH,))
        self.assertEqual(actions.dtype, np.float64)
        self.assertTrue(0.0 <= done_prob <= 1.0)

    def test_save_and_load_keep_predictions(self) -> None:
        """A loaded policy predicts what the saved one does."""
        boosters = _boosters()
        rounds = dict.fromkeys(trees.PARTS, _ROUNDS)
        policy = trees.TreePolicy(boosters, rounds, trees.TreeConfig())
        with tempfile.TemporaryDirectory() as folder:
            trees.save_policy(
                os.path.join(folder, "policy"),
                boosters,
                rounds,
                trees.TreeConfig(),
                {},
            )
            with mock.patch.object(paths, "CHECKPOINT_DIR", folder):
                self.assertTrue(trees.is_checkpoint("policy"))
                loaded = trees.load_policy("policy")
        expected_actions, expected_done = policy.predict(_window())
        actions, done_prob = loaded.predict(_window())
        np.testing.assert_array_equal(actions, expected_actions)
        self.assertEqual(done_prob, expected_done)

    def test_ensemble_of_copies_matches_one(self) -> None:
        """Averaging a policy with itself changes nothing."""
        policy = _policy()
        expected_actions, expected_done = policy.predict(_window())
        actions, done_prob = trees.TreeEnsemble([policy, policy]).predict(
            _window()
        )
        np.testing.assert_array_equal(actions, expected_actions)
        self.assertEqual(done_prob, expected_done)

    def test_empty_ensemble_is_refused(self) -> None:
        """An ensemble needs at least one policy."""
        with self.assertRaises(ValueError):
            trees.TreeEnsemble([])
