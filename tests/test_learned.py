"""Tests of the learned policy."""

import typing
import unittest

import numpy as np
import torch

from behavior_cloning import (
    env,
    episode,
    expert,
    features,
    learned,
    model,
    paths,
    robot,
)

_MAX_STEPS = 25


class _RecordingNetwork(model.PolicyNetwork):
    """Untrained network that records its predict calls.

    Attributes:
        calls: Number of predict calls so far.
        last_windows: Input windows of the latest call.
    """

    def __init__(self) -> None:
        """Build an untrained network with no calls recorded."""
        super().__init__(model.NetworkConfig())
        self.calls = 0
        self.last_windows = torch.empty(0)

    @typing.override
    def predict(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the call, then predict as usual."""
        self.calls += 1
        self.last_windows = windows.clone()
        return super().predict(windows)


class LearnedPolicyTest(unittest.TestCase):
    """The learned policy feeds its network on time and correctly."""

    def _policy(
        self,
    ) -> tuple[env.Environment, _RecordingNetwork, learned.LearnedPolicy]:
        """Return a simulation, an untrained network, and its policy."""
        environment = env.Environment(
            paths.REPO_DIR, fixed_finger_angles=robot.FINGER_HOLD_ANGLES
        )
        self.addCleanup(environment.close)
        network = _RecordingNetwork()
        policy = learned.LearnedPolicy.from_environment(
            environment, network, fast_inference=False
        )
        return environment, network, policy

    def test_network_runs_once_per_policy_period(self) -> None:
        """The network runs on steps 0, 10, and 20 of 25."""
        environment, network, policy = self._policy()
        result = episode.run_episode(
            environment,
            policy,
            (0.0, 1.1, 0.5),
            episode.StopCriteria(max_steps=_MAX_STEPS),
        )
        self.assertEqual(result.steps, _MAX_STEPS + 1)
        self.assertEqual(network.calls, -(-_MAX_STEPS // robot.POLICY_PERIOD))

    def test_inputs_hold_the_expert_standoff(self) -> None:
        """The network's first input uses the expert's standoff."""
        environment, network, policy = self._policy()
        target = np.array([0.0, 1.1, 0.5])
        obs = environment.reset(target, None, None)
        policy.reset(target, obs)
        policy.act(obs)
        standoff = expert.standoff_distance(
            target[2:], expert.PlannerParams()
        )[0]
        expected = features.state_features(
            obs.base_pos,
            obs.base_bearing,
            obs.q[robot.ARM_MOTORS],
            obs.ee_pos,
            target,
            standoff,
        )
        np.testing.assert_array_equal(
            network.last_windows[0, -1].numpy(), expected
        )
