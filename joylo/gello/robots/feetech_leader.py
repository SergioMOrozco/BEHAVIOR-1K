"""
Leader-arm hardware interface for SO-100/SO-101-style arms using Feetech STS3215 servos.

Ported from leisaac's SO101Leader (github.com/LightwheelAI/leisaac), with the hardware-protocol
layer (FeetechMotorsBus, calibration format) vendored unchanged in gello/robots/feetech/. This
class does NOT extend leisaac's own `Device` base (it pulls in IsaacLab/carb dependencies) --
it implements gello's own `Robot` ABC instead, matching the pattern already used for Dynamixel
leader arms in dynamixel_robot.py.
"""

import json
from typing import Dict, Optional

import numpy as np

from gello.robots.base_robot import Robot
from gello.robots.feetech import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)

# xlerobot's actual arm joint limits (radians), read directly from its URDF. Identical for both
# arms since the left arm ("_2" suffix) is a mirrored duplicate with the same joint ranges. Used
# only as a safety clamp on the final home-relative radians value (see _rescale) -- NOT as the
# rescale target range, since the follower's joint-limit-range midpoint has no physical relation
# to the leader's calibrated home pose.
XLEROBOT_ARM_JOINT_LIMITS = {
    "shoulder_pan": (-2.1, 2.1),  # Rotation / Rotation_2
    "shoulder_lift": (-0.1, 3.45),  # Pitch / Pitch_2
    "elbow_flex": (-0.2, 3.14159),  # Elbow / Elbow_2
    "wrist_flex": (-1.8, 1.8),  # Wrist_Pitch / Wrist_Pitch_2
    "wrist_roll": (-3.14159, 3.14159),  # Wrist_Roll / Wrist_Roll_2
    "gripper": (0.0, 1.7),  # Jaw / Jaw_2
}

# STS3215 encoder resolution and the fixed "home" tick value calibration's set_half_turn_homings()
# assigns to wherever the arm was physically posed during calibration -- see _ticks_to_radians.
TICKS_PER_REV = 4096
HALF_TURN = (TICKS_PER_REV - 1) // 2

# Arm joints in xlerobot's arm_joint_names order, followed by the gripper joint separately.
ARM_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"


class FeetechLeaderArm(Robot):
    """
    A single SO-100/SO-101 leader arm, read-only, reporting joint positions in radians rescaled
    into a follower's actual joint-limit range (default: xlerobot's arm joints).
    """

    def __init__(
        self,
        port: str,
        calibration_path: str,
        joint_limits: Optional[Dict[str, tuple]] = None,
        sign_flip: Optional[Dict[str, bool]] = None,
    ):
        """
        Args:
            port: serial port the leader arm is connected to, e.g. "/dev/ttyACM0"
            calibration_path: path to a LeRobot-format calibration JSON (homing_offset/range_min/range_max
                per joint), e.g. ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/<name>.json
            joint_limits: follower joint limits in radians per joint name, used only as a safety
                clamp (default: xlerobot's arm)
            sign_flip: optional per-joint-name set of joints to mirror about the leader's own
                calibrated home pose (0 radians) -- e.g. {"shoulder_pan": True} -- for empirically
                fixing an axis that feels inverted, e.g. on a mirrored second arm.
        """
        self.port = port
        self.calibration_path = calibration_path
        self.joint_limits = joint_limits or XLEROBOT_ARM_JOINT_LIMITS
        self.sign_flip = sign_flip or {}
        # Additive offset applied after the home-relative rescale, so the follower doesn't have
        # to assume its URDF zero pose matches the leader's calibrated home pose -- see
        # calibrate_offset(). Confirmed empirically necessary for xlerobot: e.g. its "Pitch"
        # joint's URDF range is (-0.1, 3.45) while the leader's home-relative range for the same
        # joint is roughly (-1.75, 1.89) -- similar span, but centered on a completely different
        # part of the number line, not just a small residual mismatch.
        self._offset_arm = np.zeros(len(ARM_JOINT_ORDER), dtype=np.float64)
        self._offset_gripper = np.zeros(1, dtype=np.float64)

        self._calibration = self._load_calibration()
        self._phys_ranges = self._compute_phys_ranges(self._calibration)

        calibration = self._calibration
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        self.connect()

    def connect(self):
        self._bus.connect()
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        print(f"FeetechLeaderArm connected on {self.port}")

    def disconnect(self):
        self._bus.disconnect()
        print(f"FeetechLeaderArm on {self.port} disconnected")

    def _load_calibration(self) -> Dict[str, MotorCalibration]:
        with open(self.calibration_path) as f:
            json_data = json.load(f)
        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    @staticmethod
    def _ticks_to_radians(raw: float) -> float:
        """Raw (already homing-corrected) tick reading -> radians relative to the arm's
        calibrated home pose. Ported from lerobot_3d's RobotState.ticks_to_radians:
        Feetech servos compute Present_Position = Actual_Position - Homing_Offset on-device,
        so every tick value we ever see already has that applied; HALF_TURN is the fixed
        reference point calibration's set_half_turn_homings() defines as "home."
        """
        return (raw - HALF_TURN) * (2 * np.pi / TICKS_PER_REV)

    def _compute_phys_ranges(self, calibration: Dict[str, MotorCalibration]) -> Dict[str, dict]:
        """This leader's own calibrated physical range per joint, in radians relative to its
        own calibrated home pose -- ported from lerobot_3d's RobotState.compute_phys_ranges.
        Deliberately per-instance (not a shared/generic table): two physical units calibrated
        at different times can have meaningfully different range_min/range_max.
        """
        phys_ranges = {}
        for joint_name, calib in calibration.items():
            phys_ranges[joint_name] = {
                "lo": self._ticks_to_radians(calib.range_min),
                "hi": self._ticks_to_radians(calib.range_max),
                "drive_mode": bool(calib.drive_mode),
            }
        return phys_ranges

    #def _rescale(self, joint_name: str, norm_value: float) -> float:
    #    """Un-normalize a calibrated [-100,100]/[0,100] reading into radians relative to this
    #    leader's own calibrated home pose (0 rad = wherever the arm was physically posed during
    #    calibration) -- ported from lerobot_3d's RobotState.convert_lerobot_action_to_radians.
    #    No relationship to any follower's joint-limit range is assumed here -- see
    #    calibrate_offset() for how this gets mapped onto the follower's own joint space.

    #    Note: unlike lerobot_3d's source (which un-normalizes a value that has NOT yet had
    #    drive_mode's sign flip applied), our vendored FeetechMotorsBus._normalize() already
    #    applies that flip before we ever see `norm_value` (see motors_bus.py) -- so drive_mode is
    #    NOT re-applied here; doing so would double-flip it. Not currently exercised by either
    #    leader's calibration file (drive_mode=0 for every joint in both), but matters for any
    #    future calibration with drive_mode=1.
    #    """
    #    r = self._phys_ranges[joint_name]
    #    lo_clip = 0.0 if joint_name == GRIPPER_JOINT else -100.0
    #    norm = float(np.clip(norm_value, lo_clip, 100.0))
    #    frac = norm / 100.0 if joint_name == GRIPPER_JOINT else (100.0 + norm) / 200.0
    #    radians = r["lo"] + frac * (r["hi"] - r["lo"])
    #    if self.sign_flip.get(joint_name, False):
    #        radians = -radians
    #    return radians

    def _rescale(self, joint_name: str, norm_value: float) -> float:
        """Map the leader's calibrated position directly into the simulated
        robot's joint range.
        """
        sim_lo, sim_hi = self.joint_limits[joint_name]

        if joint_name == GRIPPER_JOINT:
            norm = float(np.clip(norm_value, 0.0, 100.0))
            frac = norm / 100.0
        else:
            norm = float(np.clip(norm_value, -100.0, 100.0))
            frac = (norm + 100.0) / 200.0

        # Reverse the direction inside the simulated joint range.
        if self.sign_flip.get(joint_name, False):
            frac = 1.0 - frac

        return sim_lo + frac * (sim_hi - sim_lo)

    def num_dofs(self) -> int:
        return len(ARM_JOINT_ORDER) + 1

    def _rescaled_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Home-relative rescale only, ignoring any offset -- used internally and by calibrate_offset()."""
        raw = self._bus.sync_read("Present_Position")  # dict[str, float], leisaac-normalized range
        arm = np.array([self._rescale(name, raw[name]) for name in ARM_JOINT_ORDER], dtype=np.float64)
        gripper = np.array([self._rescale(GRIPPER_JOINT, raw[GRIPPER_JOINT])], dtype=np.float64)
        return arm, gripper

    def calibrate_offset(self, target_arm: np.ndarray, target_gripper: np.ndarray) -> None:
        """
        Sets this leader's offset so that its CURRENT physical pose maps to (target_arm,
        target_gripper). Call this once right after connecting, passing the follower's current
        joint positions (typically its reset pose) -- this makes the follower start exactly where
        it already is, and track the leader's motion RELATIVE to that starting pose. This is
        necessary because xlerobot's URDF zero pose does NOT match the leader's calibrated home
        pose (confirmed empirically -- e.g. its "Pitch"/shoulder_lift joint's URDF range is
        (-0.1, 3.45), entirely offset from the leader's home-relative range of about (-1.75, 1.89)
        for the same joint).

        Args:
            target_arm: shape (5,), the follower arm's current joint positions (radians)
            target_gripper: shape (1,), the follower gripper's current joint position (radians)
        """
        current_arm, current_gripper = self._rescaled_state()
        self._offset_arm = target_arm - current_arm
        self._offset_gripper = target_gripper - current_gripper

    def get_arm_and_gripper_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._rescaled_state()

    #def get_arm_and_gripper_state(self) -> tuple[np.ndarray, np.ndarray]:
    #    """
    #    Returns:
    #        2-tuple:
    #            - np.ndarray of shape (5,): [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    #              wrist_roll] home-relative radians plus this leader's offset (see
    #              calibrate_offset()), clipped to the follower's joint limits as a safety net --
    #              ordered to match xlerobot's arm_joint_names / arm_action_idx.
    #            - np.ndarray of shape (1,): [gripper] mapped the same way, matching xlerobot's
    #              gripper joint / gripper_action_idx.
    #    """
    #    arm, gripper = self._rescaled_state()
    #    arm = arm + self._offset_arm
    #    gripper = gripper + self._offset_gripper
    #    arm_clipped = np.array(
    #        [np.clip(v, *self.joint_limits[name]) for v, name in zip(arm, ARM_JOINT_ORDER)], dtype=np.float64
    #    )
    #    gripper_clipped = np.clip(gripper, *self.joint_limits[GRIPPER_JOINT])
    #    return arm_clipped, gripper_clipped

    def get_joint_state(self) -> np.ndarray:
        arm, gripper = self.get_arm_and_gripper_state()
        return np.concatenate([arm, gripper])

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        # Read-only leader arm -- no force-feedback/torque command support needed for this use case.
        raise NotImplementedError("FeetechLeaderArm is read-only; command_joint_state is not supported.")

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_positions": self.get_joint_state()}
