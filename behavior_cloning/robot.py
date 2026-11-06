"""Joint layout and shared constants of the Robotnik robot."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Simulator motor order: 4 wheels, 7 arm joints, 6 fingers.
WHEEL_MOTORS = slice(0, 4)
TORQUE_MOTORS = slice(0, 11)
ARM_MOTORS = slice(4, 11)
FINGER_MOTORS = slice(11, 17)
NUM_WHEELS = 4
NUM_ARM_JOINTS = 7

# Pinocchio velocity order: 6 floating-base DoFs, then the arm.
ARM_DOFS = slice(6, 13)

# Finger angles held throughout every episode (rad), in the order
# finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3,
# finger_tip_3.
FINGER_HOLD_ANGLES = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

# Simulation steps per policy step: 100 Hz at the 1 kHz simulation rate.
POLICY_PERIOD = 10

# Read-only zero commands, shared across steps.
WHEEL_ZEROS = np.zeros(NUM_WHEELS)
WHEEL_ZEROS.flags.writeable = False
ARM_ZEROS = np.zeros(NUM_ARM_JOINTS)
ARM_ZEROS.flags.writeable = False
