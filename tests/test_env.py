"""Tests of the simulation environment."""

import unittest

import numpy as np

from behavior_cloning import env, episode, expert, paths, robot

_CRITERIA = episode.StopCriteria(max_steps=5000)
_TARGET = (0.0, 1.1, 0.5)
_OTHER_TARGET = (0.9, -0.4, 0.8)
_OTHER_BASE_XY = (0.3, -0.2)
_OTHER_BASE_BEARING = 1.0


def _run_expert(
    environment: env.Environment,
    target_pos: tuple[float, float, float],
    base_xy: tuple[float, float] | None = None,
    base_bearing: float | None = None,
) -> episode.EpisodeResult:
    """Run one expert episode in an environment."""
    policy = expert.ExpertPolicy.from_environment(environment)
    return episode.run_episode(
        environment, policy, target_pos, _CRITERIA, base_xy, base_bearing
    )


class ResetTest(unittest.TestCase):
    """Episodes after a reset match episodes in a fresh simulation."""

    def _make_environment(self) -> env.Environment:
        """Build a simulation that closes when the test ends."""
        environment = env.Environment(
            paths.REPO_DIR, fixed_finger_angles=robot.FINGER_HOLD_ANGLES
        )
        self.addCleanup(environment.close)
        return environment

    def _assert_same_result(
        self, actual: episode.EpisodeResult, expected: episode.EpisodeResult
    ) -> None:
        """Assert that two episodes ended in the same state."""
        self.assertEqual(actual.steps, expected.steps)
        self.assertEqual(actual.stop_message, expected.stop_message)
        self.assertEqual(actual.final_obs.time, expected.final_obs.time)
        for field in ("base_pos", "q", "qd", "ee_pos"):
            np.testing.assert_array_equal(
                getattr(actual.final_obs, field),
                getattr(expected.final_obs, field),
                err_msg=field,
            )

    def test_repeated_episode_matches_first(self) -> None:
        """A second identical episode ends in the same state."""
        environment = self._make_environment()
        first = _run_expert(environment, _TARGET)
        second = _run_expert(environment, _TARGET)
        self._assert_same_result(second, first)

    def test_new_start_after_reset_matches_fresh_build(self) -> None:
        """A reset to a new start matches a freshly built simulation."""
        expected = _run_expert(
            self._make_environment(),
            _OTHER_TARGET,
            _OTHER_BASE_XY,
            _OTHER_BASE_BEARING,
        )
        environment = self._make_environment()
        _run_expert(environment, _TARGET)
        actual = _run_expert(
            environment, _OTHER_TARGET, _OTHER_BASE_XY, _OTHER_BASE_BEARING
        )
        self._assert_same_result(actual, expected)
