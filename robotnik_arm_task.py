"""Reach a target with the Robotnik mobile base and Kinova arm.

Runs the expert controller in PyBullet: the base turns toward the
target and drives to a standoff distance, then the arm reaches for it.

Environment variables:
    ROBOENV_HEADLESS: "1", "true", or "yes" runs without the GUI and
        skips the plots.
    ROBOENV_MAX_STEPS: Simulation step cap (default 40000; 0 disables).
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from behavior_cloning import env, episode, expert, robot

# Run configuration
HEADLESS = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
MAX_STEPS = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

# Steps between progress prints
PRINT_EVERY = 500

QUIT_KEY = ord("q")

# ======================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector
# in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)].
TARGET_EE_POS = np.array([0.0, 1.1, 0.5])

# Fixed gripper angles (rad), held throughout the simulation. Set to
# None to use the initial configuration from the config/URDF instead.
FIXED_FINGER_ANGLES = robot.FINGER_HOLD_ANGLES

# Stop once the EE is this close to the target (m); max iterations set
# via ROBOENV_MAX_STEPS.
REACH_TOLERANCE = 0.01
# ======================================================================


def main() -> None:
    """Run one expert reaching episode and report the results."""
    environment = env.Environment(
        os.path.dirname(os.path.abspath(__file__)),
        use_gui=not HEADLESS,
        fixed_finger_angles=FIXED_FINGER_ANGLES,
    )
    expert_policy = expert.ExpertPolicy.from_environment(environment)

    criteria = episode.StopCriteria(
        reach_tolerance=REACH_TOLERANCE, max_steps=MAX_STEPS
    )

    # Data logging for plotting (GUI runs only)
    time_all: list[float] = []
    ee_pos_all: list[robot.FloatArray] = []
    ee_error_all: list[robot.FloatArray] = []
    q_mes_all: list[robot.FloatArray] = []
    q_d_all: list[robot.FloatArray] = []
    steps_to_print = PRINT_EVERY

    def log_step(
        obs: env.Observation,
        action: expert.ExpertAction,
        next_obs: env.Observation,
    ) -> bool:
        """Print progress, log the step, and check for the quit key."""
        nonlocal steps_to_print
        steps_to_print -= 1
        if not steps_to_print:
            steps_to_print = PRINT_EVERY
            ee_error = TARGET_EE_POS - obs.ee_pos
            print(
                f"t={next_obs.time:.2f}s"
                f" | Base pos: {np.round(obs.base_pos[:2], 3)}"
                f" | EE pos: {np.round(obs.ee_pos, 3)}"
                f" | EE error: {np.linalg.norm(ee_error):.4f}m"
                f" | bearing {obs.base_bearing:.2f} phase {action.phase}"
                f" nav_done {action.base_done}"
            )
        if HEADLESS:
            return False
        time_all.append(obs.time)
        ee_pos_all.append(obs.ee_pos)
        ee_error_all.append(TARGET_EE_POS - obs.ee_pos)
        q_mes_all.append(obs.q[robot.ARM_MOTORS])
        q_d_all.append(action.q_des_arm)
        time.sleep(0.001)
        return environment.key_pressed(QUIT_KEY)

    initial_ee_pos = environment.reset(TARGET_EE_POS).ee_pos
    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Initial Cartesian EE Position: {np.round(initial_ee_pos, 4)}")
    print(
        "Initial Cartesian Error (m)  : "
        f"{np.linalg.norm(TARGET_EE_POS - initial_ee_pos):.4f}"
    )
    print("==================================================================")

    result = episode.run_episode(
        environment, expert_policy, TARGET_EE_POS, criteria, on_step=log_step
    )
    if result.stop_message:
        print(result.stop_message)
    obs = result.final_obs

    final_fingers = obs.q[robot.FINGER_MOTORS]

    # Final summary report
    print("==================================================================")
    print(f"Final Base Position          : {np.round(obs.base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(obs.ee_pos, 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(
        "Final Cartesian Error (m)    : "
        f"{np.linalg.norm(TARGET_EE_POS - obs.ee_pos):.4f}"
    )
    print(
        "Fixed Finger Angles          : "
        f"{np.round(environment.finger_angles, 4)}"
    )
    print(f"Final Finger Angles          : {np.round(final_fingers, 4)}")
    print(
        "Max Finger Deviation (rad)   : "
        f"{np.max(np.abs(final_fingers - environment.finger_angles)):.6f}"
    )
    print("==================================================================")

    if HEADLESS:
        print(f"Headless run finished at t={obs.time:.2f}s")
        return

    # ------------------------------------------------------------------
    # Plotting Performance
    # ------------------------------------------------------------------
    ee_pos_history = np.array(ee_pos_all)
    q_mes_history = np.array(q_mes_all)
    q_d_history = np.array(q_d_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    for idx, coord in enumerate("XYZ"):
        plt.subplot(3, 1, idx + 1)
        plt.plot(
            time_all,
            ee_pos_history[:, idx],
            label=f"Actual EE {coord}",
            color="b",
        )
        plt.axhline(
            y=TARGET_EE_POS[idx],
            color="r",
            linestyle="--",
            label=f"Target EE {coord}",
        )
        plt.ylabel(f"{coord} Position (m)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.suptitle("End-Effector Cartesian Position Reaching")
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label="EE Euclidean Error (m)", color="g")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title("Cartesian End-Effector Reaching Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(robot.NUM_ARM_JOINTS):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, q_mes_history[:, i], label=f"Measured J{i + 1}")
        plt.plot(
            time_all,
            q_d_history[:, i],
            label=f"Desired J{i + 1}",
            linestyle="--",
        )
        plt.ylabel(f"J{i + 1} (rad)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.suptitle("Arm Joint Position Profiles")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
