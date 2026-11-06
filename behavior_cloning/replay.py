"""Prioritized experience replay over the training rows (Schaul et al.).

Rows are drawn with probability rising with their latest loss, and
importance weights undo the bias this adds to the gradient.
"""

import torch

# Added to every priority, so no row stops being drawn.
_PRIORITY_FLOOR = 1e-3


class Replay:
    """Sampling priorities of the training rows.

    Each row is drawn with probability proportional to its latest loss
    raised to alpha, and importance weights undo the bias this adds.
    """

    def __init__(self, count: int, alpha: float) -> None:
        """Start with every row equally likely.

        Args:
            count: Number of training rows.
            alpha: Priority exponent.
        """
        self._alpha = alpha
        self._priorities = torch.ones(count)
        self._probs = torch.full((count,), 1.0 / count)
        self._beta = 1.0
        self._weight_scale = 1.0

    def sample(self, generator: torch.Generator, beta: float) -> torch.Tensor:
        """Draw one epoch of row positions from the current priorities.

        Args:
            generator: Random generator of the draw.
            beta: Importance-weight exponent for this epoch.

        Returns:
            Positions in the training rows, one per row, with repeats.
        """
        probs = self._priorities.pow(self._alpha)
        probs /= probs.sum()
        self._probs = probs
        self._beta = beta
        # The least likely row has the largest weight; scale it to 1.
        self._weight_scale = (len(probs) * probs.min()).pow(-beta).item()
        return torch.multinomial(
            probs, len(probs), replacement=True, generator=generator
        )

    def weighted_mean(
        self, losses: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return the importance-weighted mean of a batch's losses.

        Args:
            losses: Row losses of shape (batch,).
            positions: Positions of the batch's rows.

        Returns:
            The weighted mean loss.
        """
        weights = (len(self._probs) * self._probs[positions]).pow(-self._beta)
        return (losses * weights).mean() / self._weight_scale

    def update(self, positions: torch.Tensor, losses: torch.Tensor) -> None:
        """Set the batch rows' priorities to their latest losses.

        Args:
            positions: Positions of the batch's rows.
            losses: Their latest losses.
        """
        self._priorities[positions] = losses.detach() + _PRIORITY_FLOOR


def beta_schedule(start: float, epoch: int, epochs: int) -> float:
    """Return the importance-weight exponent of an epoch.

    Args:
        start: Exponent at the first epoch.
        epoch: Epoch, counted from 1.
        epochs: Maximum number of epochs.

    Returns:
        The exponent, rising linearly from start to 1 at the last epoch.
    """
    return start + (1.0 - start) * (epoch - 1) / max(epochs - 1, 1)
