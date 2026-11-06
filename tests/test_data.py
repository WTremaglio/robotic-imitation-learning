"""Tests of episode sampling, recording and merging."""

import dataclasses
import math
import os
import tempfile
import unittest

import numpy as np
import numpy.typing as npt

from behavior_cloning import data, env, episode, expert, learned, paths, robot

_RANGES = data.TaskRanges()
_NOISE = data.NoiseLevels(0.05, 0.1, 0.02)
_CRITERIA = episode.StopCriteria(max_steps=300)
_PERIOD = 10


def _make_expert(
    test: unittest.TestCase,
) -> tuple[env.Environment, expert.ExpertPolicy]:
    """Build a simulation, closed when the test ends, and its expert."""
    environment = env.Environment(
        paths.REPO_DIR, fixed_finger_angles=robot.FINGER_HOLD_ANGLES
    )
    test.addCleanup(environment.close)
    return environment, expert.ExpertPolicy.from_environment(environment)


class SampleSpecTest(unittest.TestCase):
    """Sampled start conditions are reproducible and within range."""

    def test_same_seed_gives_same_spec(self) -> None:
        """A seed always gives the same start conditions."""
        self.assertEqual(
            data.sample_spec(7, _RANGES, _NOISE),
            data.sample_spec(7, _RANGES, _NOISE),
        )

    def test_specs_lie_within_ranges(self) -> None:
        """Every sampled start lies within the ranges."""
        for seed in range(1000):
            spec = data.sample_spec(seed, _RANGES, _NOISE)
            x, y, z = spec.target_pos
            radius = math.hypot(x, y)
            self.assertLessEqual(
                math.hypot(*spec.base_xy), _RANGES.base_radius
            )
            self.assertTrue(-math.pi <= spec.base_bearing < math.pi)
            self.assertTrue(
                _RANGES.target_radius[0] <= radius <= _RANGES.target_radius[1]
            )
            self.assertTrue(
                _RANGES.target_height[0] <= z <= _RANGES.target_height[1]
            )


class DartExpertTest(unittest.TestCase):
    """DART noise perturbs the executed commands."""

    def test_noise_changes_executed_commands(self) -> None:
        """A noisy run executes other commands than a noise-free one."""
        environment, policy = _make_expert(self)
        noisy = data.sample_spec(0, _RANGES, _NOISE)
        clean = dataclasses.replace(noisy, noise=data.NoiseLevels())
        noisy_arrays = data.record_episode(
            environment, policy, noisy, _CRITERIA, _PERIOD
        )
        clean_arrays = data.record_episode(
            environment, policy, clean, _CRITERIA, _PERIOD
        )
        for name in ("wheel_cmds", "torque_arm"):
            self.assertFalse(
                np.array_equal(noisy_arrays[name], clean_arrays[name]),
                msg=name,
            )

    def test_parked_base_gets_no_noise(self) -> None:
        """Once the base is parked, the wheels are not perturbed."""
        environment, policy = _make_expert(self)
        arrays = data.record_episode(
            environment,
            policy,
            data.sample_spec(0, _RANGES, _NOISE),
            episode.StopCriteria(),
            _PERIOD,
        )
        # The first parked row may hold the expert's last wheel command.
        parked_rows = np.flatnonzero(arrays["base_done"])[1:]
        self.assertGreater(parked_rows.size, 0)
        np.testing.assert_array_equal(arrays["wheel_cmds"][parked_rows], 0.0)


class MergeEpisodesTest(unittest.TestCase):
    """Merged episodes keep every recorded row and episode value."""

    def test_merged_rows_match_episode_files(self) -> None:
        """Each episode's slice of the dataset equals its own file."""
        environment, policy = _make_expert(self)
        with tempfile.TemporaryDirectory() as folder:
            paths = [os.path.join(folder, f"{n}.npz") for n in range(2)]
            for seed, path in enumerate(paths):
                spec = data.sample_spec(seed, _RANGES, _NOISE)
                data.save_arrays(
                    path,
                    data.record_episode(
                        environment, policy, spec, _CRITERIA, _PERIOD
                    ),
                )
            merged = data.merge_episodes(
                [data.load_arrays(path) for path in paths]
            )
            starts = merged["episode_starts"]
            self.assertEqual(starts[-1], merged["time"].shape[0])
            for k, path in enumerate(paths):
                with np.load(path) as saved:
                    for name in data.ROW_FIELDS:
                        np.testing.assert_array_equal(
                            merged[name][starts[k] : starts[k + 1]],
                            saved[name],
                            err_msg=name,
                        )
                    for name in data.EPISODE_FIELDS:
                        np.testing.assert_array_equal(
                            merged[name][k], saved[name], err_msg=name
                        )


_OBS = env.Observation(
    time=0.0,
    base_pos=np.zeros(3),
    base_bearing=0.0,
    q=np.zeros(1),
    qd=np.zeros(1),
    q_pin=np.zeros(1),
    ee_pos=np.zeros(3),
)


class _StubExpert:
    """Expert stand-in with fixed labels that counts its resets."""

    def __init__(self) -> None:
        """Start with no resets counted."""
        self.resets = 0

    def reset(
        self, target_pos: npt.NDArray[np.float64], obs: env.Observation
    ) -> None:
        """Count the reset."""
        del target_pos, obs
        self.resets += 1

    def act(self, obs: env.Observation) -> expert.ExpertAction:
        """Return fixed commands and labels."""
        del obs
        joints = robot.NUM_ARM_JOINTS
        return expert.ExpertAction(
            wheel_cmds=np.ones(4),
            torque_arm=np.ones(joints),
            v_cmd=0.1,
            w_cmd=0.2,
            q_des_arm=np.zeros(joints),
            qd_des_arm=np.zeros(joints),
            gravity_arm=np.zeros(joints),
            phase="drive",
            base_done=False,
        )


class _StubLearner:
    """Learner stand-in with fixed commands that counts its resets."""

    def __init__(self) -> None:
        """Start with no resets counted."""
        self.resets = 0
        self.wheel_cmds = np.full(4, 7.0)
        self.torque_arm = np.full(robot.NUM_ARM_JOINTS, 3.0)

    def reset(
        self, target_pos: npt.NDArray[np.float64], obs: env.Observation
    ) -> None:
        """Count the reset."""
        del target_pos, obs
        self.resets += 1

    def act(self, obs: env.Observation) -> learned.LearnedAction:
        """Return the fixed commands."""
        del obs
        return learned.LearnedAction(self.wheel_cmds, self.torque_arm, True)


class DaggerExpertTest(unittest.TestCase):
    """The learner drives while the expert's labels are kept."""

    def test_learner_drives_and_expert_labels(self) -> None:
        """Executed commands are the learner's; labels the expert's."""
        expert_policy = _StubExpert()
        learner = _StubLearner()
        dagger = data.DaggerExpert(expert_policy, learner)
        dagger.reset(np.zeros(3), _OBS)
        action = dagger.act(_OBS)
        np.testing.assert_array_equal(action.wheel_cmds, learner.wheel_cmds)
        np.testing.assert_array_equal(action.torque_arm, learner.torque_arm)
        self.assertEqual((action.v_cmd, action.w_cmd), (0.1, 0.2))
        self.assertFalse(action.base_done)
        self.assertEqual((expert_policy.resets, learner.resets), (1, 1))
