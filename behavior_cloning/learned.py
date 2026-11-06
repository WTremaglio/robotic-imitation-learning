"""Learned policy: a trained model over fixed low-level control."""

import collections.abc
import dataclasses
import functools
import os
import typing

import numpy as np
import pinocchio as pin
import simulation_and_control
import torch

from behavior_cloning import (
    control,
    env,
    expert,
    features,
    model,
    paths,
    robot,
    trees,
)

type Network = (
    model.PolicyNetwork
    | model.Ensemble
    | trees.TreePolicy
    | trees.TreeEnsemble
)

# Probability above which the network's base-done output counts as done,
# and how many network steps in a row it must do so to park the base.
_DONE_THRESHOLD = 0.5
_DONE_STEPS = 5


@dataclasses.dataclass(slots=True)
class LearnedAction:
    """Commands of the learned policy at one step.

    Attributes:
        wheel_cmds: Wheel velocity commands (rad/s).
        torque_arm: Arm joint torques (N*m).
        base_done: Whether the network has judged the base parked.
    """

    wheel_cmds: robot.FloatArray
    torque_arm: robot.FloatArray
    base_done: bool


def _frozen(
    network: model.PolicyNetwork | model.Ensemble,
) -> model.FrozenMlp | model.FrozenEnsemble | None:
    """Return a NumPy copy of an MLP or MLP ensemble, or None."""
    if isinstance(network, model.PolicyNetwork):
        return (
            model.FrozenMlp(network) if network.config.kind == "mlp" else None
        )
    if all(member.config.kind == "mlp" for member in network.networks):
        return model.FrozenEnsemble(network)
    return None


def decision_function(
    network: Network,
    window: features.Float32Array,
    fast_inference: bool = True,
) -> collections.abc.Callable[[], tuple[robot.FloatArray, float]]:
    """Return a function that runs a trained model on a window.

    Tree policies always run in NumPy, MLPs do when fast_inference is
    set, and other networks run through PyTorch. The window is read on
    every call, so updating it in place changes the next decision.

    Args:
        network: Trained model, or an ensemble of them.
        window: States in physical units, shape
            (history, STATE_WIDTH).
        fast_inference: Whether an MLP runs through NumPy.

    Returns:
        A function returning the actions and base-done probability.
    """
    if isinstance(network, trees.TreePolicy | trees.TreeEnsemble):
        return functools.partial(network.predict, window)
    frozen = _frozen(network) if fast_inference else None
    if frozen is not None:
        return functools.partial(frozen.predict, window)
    windows = torch.from_numpy(window).unsqueeze(0)

    def predict() -> tuple[robot.FloatArray, float]:
        actions, done_probs = network.predict(windows)
        return actions[0].numpy().astype(np.float64), done_probs.item()

    return predict


class LearnedPolicy:
    """Trained model at 100 Hz over the fixed 1 kHz controllers.

    Like the expert, the arm holds its start pose until the model
    judges the base parked, and only the model's base commands are
    used; after that the wheels stop and only its arm commands are.
    """

    def __init__(
        self,
        network: Network,
        dyn_model: simulation_and_control.PinWrapper,
        lower_bound: robot.FloatArray,
        upper_bound: robot.FloatArray,
        dt: float,
        base_params: control.BaseControllerParams | None = None,
        arm_params: control.ArmControllerParams | None = None,
        planner_params: expert.PlannerParams | None = None,
        fast_inference: bool = True,
    ) -> None:
        """Wrap a trained model.

        Args:
            network: Trained model, or an ensemble of them.
            dyn_model: Pinocchio model and data of the whole robot.
            lower_bound: Lower joint reference bound from
                reference_bounds (rad).
            upper_bound: Upper joint reference bound from
                reference_bounds (rad).
            dt: Simulation time step (s).
            base_params: Base parameters, for the wheel geometry;
                defaults if None.
            arm_params: Arm parameters, for the PD gains and limits;
                defaults if None.
            planner_params: Planner parameters, for the goal state;
                defaults if None.
            fast_inference: Whether an MLP runs through NumPy, which
                avoids PyTorch overhead.
        """
        self._dyn_model = dyn_model
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._dt = dt
        self._base_params = base_params or control.BaseControllerParams()
        self._arm_params = arm_params or control.ArmControllerParams()
        self._planner_params = planner_params or expert.PlannerParams()
        self._window = np.zeros(
            (network.config.history, features.STATE_WIDTH), dtype=np.float32
        )
        self._predict = decision_function(
            network, self._window, fast_inference
        )
        self._target_pos = np.zeros(3)
        self._standoff = 0.0
        self._countdown = 1
        self._done_streak = 0
        self._base_done = False
        self._wheel_cmds = robot.WHEEL_ZEROS
        self._q_hold_arm = robot.ARM_ZEROS
        self._q_des_arm = self._q_hold_arm
        self._qd_des_arm = robot.ARM_ZEROS

    @classmethod
    def from_environment(
        cls,
        environment: env.Environment,
        network: Network,
        base_params: control.BaseControllerParams | None = None,
        arm_params: control.ArmControllerParams | None = None,
        planner_params: expert.PlannerParams | None = None,
        fast_inference: bool = True,
    ) -> typing.Self:
        """Create a learned policy for the robot in an environment.

        Args:
            environment: Simulation whose robot the policy controls.
            network: Trained model, or an ensemble of them.
            base_params: Base parameters; defaults if None.
            arm_params: Arm parameters; defaults if None.
            planner_params: Planner parameters; defaults if None.
            fast_inference: Whether an MLP runs through NumPy.

        Returns:
            The learned policy.
        """
        arm_params = arm_params or control.ArmControllerParams()
        lower_bound, upper_bound = control.reference_bounds(
            environment.arm_lower,
            environment.arm_upper,
            arm_params.limit_margin,
        )
        return cls(
            network,
            environment.dyn_model,
            lower_bound,
            upper_bound,
            environment.dt,
            base_params,
            arm_params,
            planner_params,
            fast_inference,
        )

    def reset(
        self, target_pos: robot.FloatArray, obs: env.Observation
    ) -> None:
        """Start a new episode with the history padded by obs.

        Args:
            target_pos: Target end-effector position, world frame (m).
            obs: First observation of the episode.
        """
        self._target_pos = target_pos
        self._standoff = float(
            expert.standoff_distance(target_pos[2:], self._planner_params)[0]
        )
        self._window[:] = self._state(obs)
        self._countdown = 1
        self._done_streak = 0
        self._base_done = False
        self._wheel_cmds = robot.WHEEL_ZEROS
        self._q_hold_arm = obs.q[robot.ARM_MOTORS].copy()
        self._q_des_arm = self._q_hold_arm
        self._qd_des_arm = robot.ARM_ZEROS

    def act(self, obs: env.Observation) -> LearnedAction:
        """Compute the commands for one simulation step.

        Args:
            obs: Current robot state.

        Returns:
            The commands for this step.
        """
        q_arm = obs.q[robot.ARM_MOTORS]
        countdown = self._countdown - 1
        if countdown:
            self._countdown = countdown
            q_des_arm = (
                self._q_des_arm + self._qd_des_arm * self._dt
                if self._base_done
                else self._q_hold_arm
            )
        else:
            self._countdown = robot.POLICY_PERIOD
            q_des_arm = self._run_network(obs, q_arm)
        params = self._arm_params
        if self._base_done:
            q_des_arm = control.clip(
                q_des_arm,
                q_arm - params.max_q_lead,
                q_arm + params.max_q_lead,
            )
            q_des_arm = control.clip(
                q_des_arm, self._lower_bound, self._upper_bound
            )
        self._q_des_arm = q_des_arm
        gravity_arm = pin.computeGeneralizedGravity(
            self._dyn_model.pin_model, self._dyn_model.pin_data, obs.q_pin
        )[robot.ARM_DOFS]
        torque_arm = control.joint_torque(
            params,
            self._q_des_arm,
            self._qd_des_arm,
            q_arm,
            obs.qd[robot.ARM_MOTORS],
            gravity_arm,
        )
        return LearnedAction(self._wheel_cmds, torque_arm, self._base_done)

    def _state(self, obs: env.Observation) -> features.Float32Array:
        """Return the network inputs of one observation."""
        return features.state_features(
            obs.base_pos,
            obs.base_bearing,
            obs.q[robot.ARM_MOTORS],
            obs.ee_pos,
            self._target_pos,
            self._standoff,
        )

    def _run_network(
        self, obs: env.Observation, q_arm: robot.FloatArray
    ) -> robot.FloatArray:
        """Update the history, run the model, and hold its outputs.

        Args:
            obs: Current robot state.
            q_arm: Measured arm joint angles (rad).

        Returns:
            The new arm joint reference (rad), before clipping: the held
            start pose until the base is done.
        """
        window = self._window
        window[:-1] = window[1:]
        window[-1] = self._state(obs)
        action, done_prob = self._predict()
        if not self._base_done:
            if done_prob > _DONE_THRESHOLD:
                self._done_streak += 1
                self._base_done = self._done_streak >= _DONE_STEPS
            else:
                self._done_streak = 0
        if self._base_done:
            self._wheel_cmds = robot.WHEEL_ZEROS
            self._qd_des_arm = action[features.QD_DES_LABELS]
            return q_arm + action[features.Q_OFFSET_LABELS]
        params = self._base_params
        self._wheel_cmds = control.wheel_velocities(
            min(
                max(float(action[features.V_LABEL]), -params.max_lin_vel),
                params.max_lin_vel,
            ),
            min(
                max(float(action[features.W_LABEL]), -params.max_ang_vel),
                params.max_ang_vel,
            ),
            params,
        )
        return self._q_hold_arm


def checkpoint_exists(name: str) -> bool:
    """Return whether a checkpoint holds a network or tree policy."""
    network_file = paths.network_file(name)
    return os.path.isfile(network_file) or trees.is_checkpoint(name)


def load_network(checkpoints: str) -> Network:
    """Load one checkpoint's model, or an ensemble of several.

    Args:
        checkpoints: Checkpoint name under checkpoints/, or several
            joined by commas, all networks or all tree policies.

    Returns:
        The model, or an ensemble that averages the models.

    Raises:
        ValueError: If the checkpoints mix networks and tree policies.
    """
    names = checkpoints.split(",")
    tree_names = [name for name in names if trees.is_checkpoint(name)]
    if tree_names and len(tree_names) < len(names):
        raise ValueError("An ensemble cannot mix networks and tree policies")
    if tree_names:
        policies = [trees.load_policy(name) for name in names]
        return (
            policies[0] if len(policies) == 1 else trees.TreeEnsemble(policies)
        )
    networks = [model.load_network(paths.network_file(name)) for name in names]
    return networks[0] if len(networks) == 1 else model.Ensemble(networks)


def load_policy(
    environment: env.Environment, checkpoints: str
) -> LearnedPolicy:
    """Load one checkpoint's policy, or an ensemble of several.

    Args:
        environment: Simulation whose robot the policy controls.
        checkpoints: Checkpoint name under checkpoints/, or several
            joined by commas.

    Returns:
        The learned policy, with default controller parameters.
    """
    return LearnedPolicy.from_environment(
        environment, load_network(checkpoints)
    )
