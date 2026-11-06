"""Tests of the controllers."""

import math
import unittest

import numpy as np

from behavior_cloning import control

_PARAMS = control.BaseControllerParams()
_STANDOFF = 1.0


def _step(
    target_x: float,
    target_y: float,
    state: control.BaseNavState | None = None,
    time: float = 0.0,
) -> tuple[control.BaseNavState, float, float]:
    """Run one base step from the origin, facing +x, toward a target.

    Args:
        target_x: Target x (m).
        target_y: Target y (m).
        state: Navigation state to start from; a fresh one if None.
        time: Simulation time (s).

    Returns:
        A tuple (state, v_cmd, w_cmd) after the step.
    """
    state = state or control.BaseNavState()
    v_cmd, w_cmd = control.base_controller(
        state, np.zeros(3), 0.0, target_x, target_y, _STANDOFF, time, _PARAMS
    )
    return state, v_cmd, w_cmd


class BaseControllerTest(unittest.TestCase):
    """The base turns toward the target and closes the standoff gap."""

    def test_far_target_ahead_drives_at_full_speed(self) -> None:
        """A target far ahead gives full forward speed and no turn."""
        state, v_cmd, w_cmd = _step(3.0, 0.0)
        self.assertEqual(state.phase, "drive")
        self.assertAlmostEqual(v_cmd, _PARAMS.max_lin_vel)
        self.assertAlmostEqual(w_cmd, 0.0)

    def test_too_close_backs_up(self) -> None:
        """A target inside the standoff makes the base reverse."""
        state, v_cmd, _ = _step(0.5, 0.0)
        self.assertEqual(state.phase, "drive")
        self.assertAlmostEqual(v_cmd, -_PARAMS.max_lin_vel)

    def test_target_behind_turns_on_the_spot(self) -> None:
        """A target behind the base gives a full turn and no driving."""
        state, v_cmd, w_cmd = _step(-3.0, 0.0)
        self.assertEqual(state.phase, "align")
        self.assertEqual(v_cmd, 0.0)
        self.assertAlmostEqual(abs(w_cmd), _PARAMS.max_ang_vel)

    def test_small_heading_error_turns_and_drives(self) -> None:
        """Half the drive angle off gives half speed while turning."""
        angle = _PARAMS.drive_angle / 2
        state, v_cmd, w_cmd = _step(3 * math.cos(angle), 3 * math.sin(angle))
        self.assertEqual(state.phase, "drive")
        self.assertAlmostEqual(v_cmd, _PARAMS.max_lin_vel / 2)
        self.assertAlmostEqual(w_cmd, _PARAMS.kp_heading * angle)

    def test_at_standoff_but_turned_only_turns(self) -> None:
        """At the standoff but not facing it, the base only turns."""
        angle = 0.3
        state, v_cmd, w_cmd = _step(math.cos(angle), math.sin(angle))
        self.assertEqual(state.phase, "orient")
        self.assertFalse(state.done)
        self.assertEqual(v_cmd, 0.0)
        self.assertAlmostEqual(w_cmd, _PARAMS.kp_heading * angle)

    def test_parks_within_both_tolerances(self) -> None:
        """Near the standoff and facing the target, the base parks."""
        state, v_cmd, w_cmd = _step(_STANDOFF + 0.01, 0.0)
        self.assertTrue(state.done)
        self.assertEqual((v_cmd, w_cmd), (0.0, 0.0))

    def test_parked_base_stays_parked(self) -> None:
        """A parked base stays still wherever the target is."""
        parked = control.BaseNavState(done=True)
        state, v_cmd, w_cmd = _step(3.0, 0.0, parked)
        self.assertTrue(state.done)
        self.assertEqual((v_cmd, w_cmd), (0.0, 0.0))

    def test_timeout_parks(self) -> None:
        """After the timeout, the base parks wherever it is."""
        state, _, _ = _step(3.0, 0.0, time=_PARAMS.nav_timeout + 1.0)
        self.assertTrue(state.done)
