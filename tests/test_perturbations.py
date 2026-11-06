"""Tests of the perturbations and pushes."""

import typing
import unittest

import numpy as np

from behavior_cloning import env, learned, perturbations, robot

_SPEED = (0.05, 0.25)
_TIME = (0.5, 2.5)
_MOTORS = robot.FINGER_MOTORS.stop
_ACTION = learned.LearnedAction(robot.WHEEL_ZEROS, robot.ARM_ZEROS, False)


def _observation(time: float = 0.0, bearing: float = 0.0) -> env.Observation:
    """Return an observation with zero motor states."""
    return env.Observation(
        time=time,
        base_pos=np.zeros(3),
        base_bearing=bearing,
        q=np.zeros(_MOTORS),
        qd=np.zeros(_MOTORS),
        q_pin=np.zeros(1),
        ee_pos=np.zeros(3),
    )


class _StubEnvironment:
    """Environment stand-in that records the pushes it receives."""

    def __init__(self) -> None:
        """Start with no pushes recorded."""
        self.pushes: list[tuple[float, float]] = []

    def push(self, velocity_xy: tuple[float, float]) -> None:
        """Record the push."""
        self.pushes.append(velocity_xy)


class _StubPolicy:
    """Policy stand-in that records the observations it sees."""

    def __init__(self) -> None:
        """Start with no observations recorded."""
        self.seen: list[env.Observation] = []

    def reset(
        self, target_pos: robot.FloatArray, obs: env.Observation
    ) -> None:
        """Record the first observation."""
        del target_pos
        self.seen.append(obs)

    def act(self, obs: env.Observation) -> learned.LearnedAction:
        """Record the observation and return zero commands."""
        self.seen.append(obs)
        return _ACTION


class _StepLog:
    """Step callback that records the time after each step."""

    def __init__(self) -> None:
        """Start with no steps recorded."""
        self.times: list[float] = []

    def __call__(
        self,
        obs: env.Observation,
        action: learned.LearnedAction,
        next_obs: env.Observation,
    ) -> bool:
        """Record the step, which never ends the episode."""
        del obs, action
        self.times.append(next_obs.time)
        return False


class SamplePushTest(unittest.TestCase):
    """Training pushes are reproducible and within their ranges."""

    def test_same_seed_gives_same_push(self) -> None:
        """A seed always gives the same push."""
        self.assertEqual(
            perturbations.sample_push(7, _SPEED, _TIME),
            perturbations.sample_push(7, _SPEED, _TIME),
        )

    def test_pushes_lie_within_ranges(self) -> None:
        """Every push lies within the ranges, in both directions."""
        pushes = [
            perturbations.sample_push(seed, _SPEED, _TIME)
            for seed in range(200)
        ]
        for speed, time in pushes:
            self.assertTrue(_SPEED[0] <= abs(speed) <= _SPEED[1])
            self.assertTrue(_TIME[0] <= time <= _TIME[1])
        self.assertTrue(any(speed > 0.0 for speed, _ in pushes))
        self.assertTrue(any(speed < 0.0 for speed, _ in pushes))


class PushCallbackTest(unittest.TestCase):
    """The push lands once, to the base's left, at its time."""

    def test_pushes_once_to_the_left(self) -> None:
        """A base facing +y is pushed toward -x, once."""
        environment = _StubEnvironment()
        log = _StepLog()
        on_step = perturbations.push_callback(
            typing.cast(env.Environment, environment), 0.2, 1.0, log
        )
        times = [0.999, 1.0, 1.001]
        for time in times:
            next_obs = _observation(time, bearing=np.pi / 2)
            self.assertFalse(on_step(_observation(), _ACTION, next_obs))
        self.assertEqual(len(environment.pushes), 1)
        np.testing.assert_allclose(
            environment.pushes[0], (-0.2, 0.0), atol=1e-12
        )
        self.assertEqual(log.times, times)


class PerturbedPolicyTest(unittest.TestCase):
    """Noise reaches the policy only when asked for."""

    def test_no_noise_passes_observations_through(self) -> None:
        """Without noise the policy sees the true observation."""
        policy = _StubPolicy()
        perturbed = perturbations.PerturbedPolicy(
            policy, perturbations.Perturbation(), seed=0
        )
        obs = _observation()
        perturbed.reset(np.zeros(3), obs)
        perturbed.act(obs)
        self.assertIs(policy.seen[0], obs)
        self.assertIs(policy.seen[1], obs)

    def test_observation_noise_is_reproducible(self) -> None:
        """The same seed gives the same noisy measurement."""
        measured = []
        for _ in range(2):
            policy = _StubPolicy()
            perturbed = perturbations.PerturbedPolicy(
                policy, perturbations.Perturbation(obs_noise=0.01), seed=3
            )
            perturbed.reset(np.zeros(3), _observation())
            measured.append(policy.seen[0])
        first, second = measured
        np.testing.assert_array_equal(first.base_pos, second.base_pos)
        np.testing.assert_array_equal(first.q, second.q)
        self.assertFalse(np.array_equal(first.base_pos, np.zeros(3)))
        self.assertEqual(first.time, 0.0)
