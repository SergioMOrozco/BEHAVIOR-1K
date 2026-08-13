"""
Diagnostic: print each leader-arm joint's raw reading and computed radians, continuously,
WITHOUT connecting to or driving the simulated robot. Move the physical leader arm(s) and
watch the numbers update -- this isolates the leader-arm hardware/mapping pipeline from
everything else (sim, controller config, action-vector wiring).

Usage:
    python diagnose_leader_mapping.py --port /dev/ttyACM0 \\
        --calibration ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/bender_leader_arm.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gello.robots.feetech_leader import ARM_JOINT_ORDER, GRIPPER_JOINT, FeetechLeaderArm  # noqa: E402

DEFAULT_CALIBRATION_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()

    leader = FeetechLeaderArm(port=args.port, calibration_path=args.calibration)

    print("\nphys_ranges (radians, relative to this leader's own calibrated home):")
    for name in ARM_JOINT_ORDER + [GRIPPER_JOINT]:
        r = leader._phys_ranges[name]
        print(f"  {name:15s} lo={r['lo']:+.4f} hi={r['hi']:+.4f} drive_mode={r['drive_mode']}")

    print("\nMove the leader arm through its full range on each joint one at a time.")
    print("Watch which column changes -- it should be the joint you're actually moving.")
    print("Press Ctrl+C to stop.\n")

    header = "  ".join(f"{name:>13s}" for name in ARM_JOINT_ORDER + [GRIPPER_JOINT])
    print("        raw(normalized) -->        " + header)

    try:
        while True:
            raw = leader._bus.sync_read("Present_Position")
            raw_str = "  ".join(f"{raw[name]:+13.2f}" for name in ARM_JOINT_ORDER + [GRIPPER_JOINT])
            arm, gripper = leader.get_arm_and_gripper_state()
            rad_str = "  ".join(f"{v:+13.4f}" for v in list(arm) + list(gripper))
            print(f"raw:      {raw_str}")
            print(f"radians:  {rad_str}")
            print()
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
