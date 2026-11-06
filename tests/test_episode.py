"""Tests of the episode loop."""

import unittest

from behavior_cloning import env, episode, expert, paths, robot

_MAX_STEPS = 300
_CRITERIA = episode.StopCriteria()


class StepCapTest(unittest.TestCase):
    """The step cap counts every simulation step once."""

    def test_cap_stops_after_max_steps(self) -> None:
        """A capped episode takes max_steps + 1 steps of one dt each."""
        environment = env.Environment(
            paths.REPO_DIR, fixed_finger_angles=robot.FINGER_HOLD_ANGLES
        )
        self.addCleanup(environment.close)
        result = episode.run_episode(
            environment,
            expert.ExpertPolicy.from_environment(environment),
            (0.0, 1.1, 0.5),
            episode.StopCriteria(max_steps=_MAX_STEPS),
        )
        self.assertEqual(result.steps, _MAX_STEPS + 1)
        self.assertAlmostEqual(
            result.final_obs.time, _MAX_STEPS * environment.dt
        )
        self.assertIsNotNone(result.stop_message)
        assert result.stop_message is not None
        self.assertTrue(result.stop_message.startswith("Reached step cap"))


class CheckStopTest(unittest.TestCase):
    """The stop checks report the right outcome."""

    def test_reached_takes_precedence_over_stall(self) -> None:
        """A target reached after a stall window is reported reached."""
        state = episode.StallState(best_ee_error=0.005, best_ee_error_time=1.0)
        message = episode.check_stop(_CRITERIA, state, 9000, 9.0, 0.005, True)
        assert message is not None
        self.assertTrue(message.startswith("Target reached"))
