"""
Teleoperate xlerobot's two arms with real SO-100/SO-101 leader arms while driving the mobile
base from the keyboard.

Combines run_xlerobot_leader_teleop.py's arm/gripper joint-position passthrough with a keyboard
base driver: W = forward, S = backward, A = turn left, D = turn right. xlerobot's base is
holonomic (can strafe), but this script intentionally exposes only these 4 keys -- no strafe.

Overriding the arm/gripper controllers to raw JointController passthrough (as in
run_xlerobot_leader_teleop.py) requires action_normalize=False at the robot level. That flag is a
blanket override applied to every controller's command_input_limits, but it only fires when True
(see Robot._load_controllers) -- so turning it off does not by itself put the base in raw mode.
xlerobot's base has no default_controllers entry in xlerobot.yaml, so it defaults to
HolonomicBaseJointController, whose own constructor default is command_input_limits=None (unlike
plain JointController, which defaults to "default") -- meaning without an explicit override here,
the base would silently receive raw m/s / rad/s commands instead of the normalized [-1, 1]
commands this script's key tracker produces. The "base" entry in controller_config below restores
normalized input for just the base, independent of the arm/gripper overrides.

Usage:
    python run_xlerobot_combined_teleop.py \\
        --right-port /dev/ttyACM0 --left-port /dev/ttyACM1 \\
        --right-calibration ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/bender_leader_arm.json \\
        --left-calibration  ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/clamps_leader_arm.json

Must be run with a display (do not set OMNIGIBSON_HEADLESS) -- keyboard input requires the Kit
app window.
"""

import argparse
import os
import sys
import time

import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # allow `import gello` when run directly

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

from gello.robots.feetech_leader import FeetechLeaderArm  # noqa: E402

gm.USE_GPU_DYNAMICS = False

# Direct joint-position passthrough: command_input_limits=None + use_delta_commands=False means
# the raw command is treated as an absolute target joint position, clipped only by the joint's
# own limits (no IK, no [-1,1] normalization).
JOINT_CONTROLLER_CFG = {
    "name": "JointController",
    "motor_type": "position",
    "use_delta_commands": False,
    "command_input_limits": None,
    "command_output_limits": None,
}

DEFAULT_CALIBRATION_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader")


class BaseKeyTracker:
    """
    Tracks W/A/S/D held-key state via its own keyboard subscription (independent of any other
    input handling), so multiple keys can be held at once. Call get_base_command() each step to
    get the current [x, 0, rz] normalized velocity command for xlerobot's holonomic base -- only
    forward/backward and turning are exposed, no strafe.
    """

    def __init__(self, speed=0.5):
        self.speed = speed
        self._held_keys = set()
        self.KEY_TO_AXIS_SIGN = {
            lazy.carb.input.KeyboardInput.W: (0, 1.0),  # forward
            lazy.carb.input.KeyboardInput.S: (0, -1.0),  # backward
            lazy.carb.input.KeyboardInput.A: (2, 1.0),  # turn left
            lazy.carb.input.KeyboardInput.D: (2, -1.0),  # turn right
        }
        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        self._sub = input_interface.subscribe_to_keyboard_events(keyboard, self._event_handler)

    def _event_handler(self, event, *args, **kwargs):
        if event.input not in self.KEY_TO_AXIS_SIGN:
            return True
        if event.type in (
            lazy.carb.input.KeyboardEventType.KEY_PRESS,
            lazy.carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self._held_keys.add(event.input)
        elif event.type == lazy.carb.input.KeyboardEventType.KEY_RELEASE:
            self._held_keys.discard(event.input)
        return True

    def get_base_command(self):
        cmd = [0.0, 0.0, 0.0]
        for key in self._held_keys:
            axis, sign = self.KEY_TO_AXIS_SIGN[key]
            cmd[axis] += sign * self.speed
        return th.tensor(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--right-port", default="/dev/ttyACM0", help="Serial port for the right-arm leader device")
    parser.add_argument("--left-port", default="/dev/ttyACM1", help="Serial port for the left-arm leader device")
    parser.add_argument(
        "--right-calibration",
        default=os.path.join(DEFAULT_CALIBRATION_DIR, "bender_leader_arm.json"),
        help="Calibration JSON for the right-arm leader device",
    )
    parser.add_argument(
        "--left-calibration",
        default=os.path.join(DEFAULT_CALIBRATION_DIR, "clamps_leader_arm.json"),
        help="Calibration JSON for the left-arm leader device",
    )
    parser.add_argument("--step-hz", type=float, default=60.0, help="Target control loop frequency")
    parser.add_argument(
        "--flip-right",
        action="append",
        default=[],
        metavar="JOINT_NAME",
        help="Mirror this joint's mapping for the right leader arm (repeatable)",
    )
    parser.add_argument(
        "--flip-left",
        action="append",
        default=[],
        metavar="JOINT_NAME",
        help="Mirror this joint's mapping for the left leader arm (repeatable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = dict(
        scene=dict(type="InteractiveTraversableScene",scene_model="Rs_int"),
        robots=[
            dict(
                model="xlerobot",
                obs_modalities=[],
                action_type="continuous",
                action_normalize=False,  # raw joint-position passthrough for arms, not normalized [-1,1]
                position=[0, 0, 0],
                controller_config={
                    "arm_right": JOINT_CONTROLLER_CFG,
                    "arm_left": JOINT_CONTROLLER_CFG,
                    "gripper_right": JOINT_CONTROLLER_CFG,
                    "gripper_left": JOINT_CONTROLLER_CFG,
                    "base": {"name": "HolonomicBaseJointController", "command_input_limits": "default"},
                },
            )
        ],
        objects=[
            {
                "type": "DatasetObject",
                "name": "delicious_apple",
                "category": "apple",
                "model": "agveuv",
                "position": [0, 0, 1.0],
            }
        ]
    )
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    robot.reset()
    robot.keep_still()

    base_tracker = BaseKeyTracker(speed=0.5)
    print("Base driving: W = forward, S = backward, A = turn left, D = turn right.")

    print(f"Connecting right leader arm on {args.right_port} (calibration: {args.right_calibration}) ...")
    right_leader = FeetechLeaderArm(
        port=args.right_port,
        calibration_path=args.right_calibration,
        #sign_flip={name: True for name in args.flip_right},
        sign_flip={
            "shoulder_lift": True,
            "wrist_roll": True,
        },
    )
    print(f"Connecting left leader arm on {args.left_port} (calibration: {args.left_calibration}) ...")
    left_leader = FeetechLeaderArm(
        port=args.left_port,
        calibration_path=args.left_calibration,
        #sign_flip={name: True for name in args.flip_left},
        sign_flip={
            "shoulder_lift": True,
            "wrist_roll": True,
        },
    )

    print("Connected. Teleoperating -- Ctrl+C to quit.")
    period = 1.0 / args.step_hz
    try:
        while True:
            t0 = time.time()
            action = th.zeros(robot.action_dim)

            right_arm, right_gripper = right_leader.get_arm_and_gripper_state()
            left_arm, left_gripper = left_leader.get_arm_and_gripper_state()

            action[robot.arm_action_idx["right"]] = th.from_numpy(right_arm).float()
            action[robot.gripper_action_idx["right"]] = th.from_numpy(right_gripper).float()
            action[robot.arm_action_idx["left"]] = th.from_numpy(left_arm).float()
            action[robot.gripper_action_idx["left"]] = th.from_numpy(left_gripper).float()
            action[robot.base_action_idx] = base_tracker.get_base_command()

            env.step(action)

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\nStopping teleop.")
    finally:
        right_leader.disconnect()
        left_leader.disconnect()
        og.shutdown()


if __name__ == "__main__":
    main()
