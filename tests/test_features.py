"""Tests of the network inputs and labels."""

import math
import unittest

import numpy as np
import numpy.typing as npt

from behavior_cloning import features, robot

_RNG = np.random.default_rng(0)
_STEPS = 5
_JOINTS = robot.NUM_ARM_JOINTS


def _random_state() -> dict[str, npt.NDArray[np.float64]]:
    """Return random inputs for a few steps."""
    return {
        "base_pos": _RNG.normal(size=(_STEPS, 3)),
        "base_bearing": _RNG.uniform(-math.pi, math.pi, _STEPS),
        "q_arm": _RNG.normal(size=(_STEPS, _JOINTS)),
        "ee_pos": _RNG.normal(size=(_STEPS, 3)),
        "target_pos": _RNG.normal(size=(_STEPS, 3)),
        "standoff": _RNG.uniform(0.6, 1.0, _STEPS),
    }


class StateFeaturesTest(unittest.TestCase):
    """The inputs have the documented layout and invariance."""

    def test_width(self) -> None:
        """Each step has STATE_WIDTH inputs."""
        states = features.state_features(**_random_state())
        self.assertEqual(states.shape, (_STEPS, features.STATE_WIDTH))
        self.assertEqual(states.dtype, np.float32)

    def test_single_step_matches_batch(self) -> None:
        """One step gives the same inputs alone as within a batch."""
        state = _random_state()
        batch = features.state_features(**state)
        single = features.state_features(
            **{name: values[2] for name, values in state.items()}
        )
        np.testing.assert_array_equal(single, batch[2])

    def test_inputs_ignore_scene_motion(self) -> None:
        """Moving and turning the whole scene keeps every input."""
        state = _random_state()
        angle = 0.7
        cos, sin = math.cos(angle), math.sin(angle)
        rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0, 0, 1]])
        shift = np.array([0.4, -1.3, 0.0])
        moved = dict(state)
        for name in ("base_pos", "ee_pos", "target_pos"):
            moved[name] = state[name] @ rotation.T + shift
        moved["base_bearing"] = state["base_bearing"] + angle
        np.testing.assert_allclose(
            features.state_features(**moved),
            features.state_features(**state),
            atol=1e-5,
        )

    def test_goal_columns(self) -> None:
        """Goal columns hold the standoff point, gap, and heading."""
        state = {name: values[0] for name, values in _random_state().items()}
        state.update(
            base_pos=np.array([1.0, 2.0, 0.0]),
            base_bearing=math.pi / 2,
            target_pos=np.array([1.0, 5.0, 0.5]),
            standoff=1.0,
        )
        goal = features.state_features(**state)[features.DONE_INPUTS]
        np.testing.assert_allclose(goal, [2.0, 0.0, 2.0, 1.0, 0.0], atol=1e-6)


class ActionLabelsTest(unittest.TestCase):
    """The labels have the documented layout."""

    def test_layout(self) -> None:
        """Labels hold v, w, the reference offsets, and qd_des."""
        q_arm = _RNG.normal(size=(_STEPS, _JOINTS))
        q_des_arm = _RNG.normal(size=(_STEPS, _JOINTS))
        qd_des_arm = _RNG.normal(size=(_STEPS, _JOINTS))
        labels = features.action_labels(
            np.ones((_STEPS, 1)),
            np.full((_STEPS, 1), 2.0),
            q_des_arm,
            q_arm,
            qd_des_arm,
        )
        self.assertEqual(labels.shape, (_STEPS, features.LABEL_WIDTH))
        np.testing.assert_array_equal(labels[:, features.V_LABEL], 1.0)
        np.testing.assert_array_equal(labels[:, features.W_LABEL], 2.0)
        np.testing.assert_allclose(
            labels[:, features.Q_OFFSET_LABELS], q_des_arm - q_arm, 1e-6
        )
        np.testing.assert_allclose(
            labels[:, features.QD_DES_LABELS], qd_des_arm, 1e-6
        )


class WindowIndicesTest(unittest.TestCase):
    """History windows stay inside their episode."""

    def test_windows_pad_with_first_row(self) -> None:
        """Windows repeat an episode's first row instead of crossing."""
        windows = features.window_indices(np.array([0, 2, 6]), 3)
        np.testing.assert_array_equal(
            windows,
            [[0, 0, 0], [0, 0, 1], [2, 2, 2], [2, 2, 3], [2, 3, 4], [3, 4, 5]],
        )


class SplitEpisodesTest(unittest.TestCase):
    """The episode split is reproducible and complete."""

    def test_split_is_disjoint_and_complete(self) -> None:
        """Every episode lands in exactly one set."""
        train, val = features.split_episodes(50, 0.2, 3)
        self.assertEqual(len(val), 10)
        np.testing.assert_array_equal(
            np.sort(np.concatenate((train, val))), np.arange(50)
        )

    def test_same_seed_gives_same_split(self) -> None:
        """A seed always gives the same split."""
        first = features.split_episodes(50, 0.2, 3)
        second = features.split_episodes(50, 0.2, 3)
        np.testing.assert_array_equal(first[1], second[1])

    def test_episode_rows(self) -> None:
        """The rows of chosen episodes are returned in order."""
        rows = features.episode_rows(np.array([0, 2, 6, 7]), np.array([2, 0]))
        np.testing.assert_array_equal(rows, [6, 0, 1])
