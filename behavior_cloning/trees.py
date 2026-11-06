"""Gradient-boosted tree policy: an XGBoost counterpart to the networks.

Three boosted models split the task the way the networks' heads do: a
classifier reads the goal inputs and judges whether the base has
parked, one regressor predicts the base commands, and one predicts the
arm commands.
"""

import collections.abc
import dataclasses
import json
import math
import os

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from behavior_cloning import features, paths

# Training record of a tree policy; its presence marks the checkpoint.
RECORD_FILE = "trees.json"
# The boosted models, each saved as NAME.ubj.
PARTS: tuple[str, ...] = ("done", "base", "arm")
# Label columns of the base and arm regressors.
BASE_LABELS = slice(0, 2)
ARM_LABELS = slice(2, features.LABEL_WIDTH)


@dataclasses.dataclass(frozen=True)
class TreeConfig:
    """Shape of a tree policy.

    Attributes:
        max_depth: Maximum depth of each tree.
        history: Steps per input window; trees read the current step.
    """

    max_depth: int = 8
    history: int = 1


class TreePolicy:
    """Boosted models that predict like a policy network."""

    def __init__(
        self,
        boosters: collections.abc.Mapping[str, xgb.Booster],
        rounds: collections.abc.Mapping[str, int],
        config: TreeConfig,
    ) -> None:
        """Wrap trained boosters, each limited to its best rounds.

        Args:
            boosters: Trained booster of each part in PARTS.
            rounds: Boosting rounds each part uses.
            config: Shape of the policy.
        """
        self.config = config
        self._boosters = dict(boosters)
        self._ranges = {part: (0, rounds[part]) for part in PARTS}
        for booster in self._boosters.values():
            booster.set_param({"nthread": 1})

    def _run(
        self, part: str, inputs: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Return one part's predictions for one row of inputs."""
        return np.asarray(
            self._boosters[part].inplace_predict(
                inputs, iteration_range=self._ranges[part]
            )
        ).reshape(-1)

    def predict(
        self, window: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Predict physical actions and the base-done probability.

        Args:
            window: States in physical units, shape
                (history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_prob) of the actions, shape
            (LABEL_WIDTH,), and the base-done probability.
        """
        state = window[-1:]
        actions = np.empty(features.LABEL_WIDTH)
        actions[BASE_LABELS] = self._run("base", state)
        actions[ARM_LABELS] = self._run("arm", state)
        goal = np.ascontiguousarray(state[:, features.DONE_INPUTS])
        return actions, float(self._run("done", goal)[0])


class TreeEnsemble:
    """Tree policies whose predictions are averaged."""

    def __init__(self, policies: collections.abc.Sequence[TreePolicy]) -> None:
        """Group trained tree policies into one policy.

        Args:
            policies: Tree policies to average.

        Raises:
            ValueError: If there are no policies.
        """
        if not policies:
            raise ValueError("An ensemble needs at least one policy")
        self._policies = tuple(policies)
        self.config = policies[0].config

    def predict(
        self, window: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Average the policies' actions and base-done probabilities.

        Args:
            window: States in physical units, shape
                (history, STATE_WIDTH).

        Returns:
            A tuple (actions, done_prob) of the averaged actions, shape
            (LABEL_WIDTH,), and the averaged base-done probability.
        """
        predictions = [policy.predict(window) for policy in self._policies]
        actions = np.stack([actions for actions, _ in predictions])
        done_prob = math.fsum(done for _, done in predictions)
        return actions.mean(axis=0), done_prob / len(predictions)


def is_checkpoint(name: str) -> bool:
    """Return whether a checkpoint name holds a tree policy."""
    return os.path.isfile(
        os.path.join(paths.checkpoint_dir(name), RECORD_FILE)
    )


def save_policy(
    folder: str,
    boosters: collections.abc.Mapping[str, xgb.Booster],
    rounds: collections.abc.Mapping[str, int],
    config: TreeConfig,
    record: collections.abc.Mapping[str, object],
) -> None:
    """Save trained boosters with their shape and training record.

    Args:
        folder: New folder to write.
        boosters: Trained booster of each part in PARTS.
        rounds: Boosting rounds each part uses.
        config: Shape of the policy.
        record: Training settings and results to save alongside.
    """
    os.makedirs(folder)
    for part in PARTS:
        boosters[part].save_model(os.path.join(folder, f"{part}.ubj"))
    with open(
        os.path.join(folder, RECORD_FILE), "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "config": dataclasses.asdict(config),
                "rounds": dict(rounds),
                **record,
            },
            file,
            indent=2,
        )


def load_policy(name: str) -> TreePolicy:
    """Load a saved tree policy, ready to run.

    Args:
        name: Checkpoint name under checkpoints/.

    Returns:
        The tree policy.
    """
    folder = paths.checkpoint_dir(name)
    with open(os.path.join(folder, RECORD_FILE), encoding="utf-8") as file:
        record = json.load(file)
    boosters = {
        part: xgb.Booster(model_file=os.path.join(folder, f"{part}.ubj"))
        for part in PARTS
    }
    return TreePolicy(
        boosters, record["rounds"], TreeConfig(**record["config"])
    )
