#!/usr/bin/env python3
"""Read calibrated Leader/Follower positions without commanding either arm."""

import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def parse_args() -> argparse.Namespace:
    cache = Path.home() / ".cache/huggingface/lerobot/calibration"
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument(
        "--leader-calibration",
        type=Path,
        default=cache / "teleoperators/so_leader/so101_leader_main.json",
    )
    parser.add_argument(
        "--follower-calibration",
        type=Path,
        default=cache / "robots/so_follower/so101_follower_main.json",
    )
    return parser.parse_args()


def motors() -> dict[str, Motor]:
    body = MotorNormMode.DEGREES
    return {
        "shoulder_pan": Motor(1, "sts3215", body),
        "shoulder_lift": Motor(2, "sts3215", body),
        "elbow_flex": Motor(3, "sts3215", body),
        "wrist_flex": Motor(4, "sts3215", body),
        "wrist_roll": Motor(5, "sts3215", body),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


def read_positions(port: str, path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())
    calibration = {name: MotorCalibration(**values) for name, values in raw.items()}
    bus = FeetechMotorsBus(port=port, motors=motors(), calibration=calibration)
    try:
        bus.connect(handshake=True)
        return bus.sync_read("Present_Position", normalize=True, num_retry=2)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def main() -> None:
    args = parse_args()
    leader = read_positions(args.leader_port, args.leader_calibration)
    follower = read_positions(args.follower_port, args.follower_calibration)
    print(f"{'JOINT':<16} | {'LEADER':>9} | {'FOLLOWER':>9} | {'ABS DIFF':>9}")
    for name in leader:
        difference = abs(leader[name] - follower[name])
        print(f"{name:<16} | {leader[name]:>9.2f} | {follower[name]:>9.2f} | {difference:>9.2f}")


if __name__ == "__main__":
    main()
