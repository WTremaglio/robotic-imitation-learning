"""Tests of prioritized replay."""

import unittest

import torch

from behavior_cloning import replay


class BetaScheduleTest(unittest.TestCase):
    """The importance-weight exponent rises linearly to 1."""

    def test_schedule_ends(self) -> None:
        """The exponent starts at its setting and ends at 1."""
        self.assertAlmostEqual(replay.beta_schedule(0.4, 1, 100), 0.4)
        self.assertAlmostEqual(replay.beta_schedule(0.4, 100, 100), 1.0)

    def test_schedule_is_linear(self) -> None:
        """Halfway through, the exponent is halfway to 1."""
        self.assertAlmostEqual(replay.beta_schedule(0.4, 51, 101), 0.7)

    def test_single_epoch(self) -> None:
        """A one-epoch run keeps the starting exponent."""
        self.assertAlmostEqual(replay.beta_schedule(0.4, 1, 1), 0.4)


class ReplayTest(unittest.TestCase):
    """Replay weights undo the sampling bias, the largest being 1."""

    def test_equal_priorities_give_unit_weights(self) -> None:
        """With equal priorities the weighted mean is the plain mean."""
        sampler = replay.Replay(4, alpha=0.6)
        sampler.sample(torch.Generator().manual_seed(0), beta=0.4)
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(
            sampler.weighted_mean(losses, torch.arange(4)), losses.mean()
        )

    def test_largest_weight_is_one(self) -> None:
        """The least likely row has weight 1, and every other less."""
        sampler = replay.Replay(4, alpha=0.6)
        sampler.update(torch.arange(4), torch.tensor([0.1, 1.0, 2.0, 4.0]))
        sampler.sample(torch.Generator().manual_seed(0), beta=1.0)
        weights = torch.stack(
            [
                sampler.weighted_mean(torch.ones(1), torch.tensor([row]))
                for row in range(4)
            ]
        )
        self.assertAlmostEqual(weights.max().item(), 1.0, places=6)
        self.assertEqual(weights.argmax().item(), 0)
