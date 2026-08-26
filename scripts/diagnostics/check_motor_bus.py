#!/usr/bin/env python3
"""Read-only SO-101 motor-bus handshake and raw position check."""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Prefer /dev/serial/by-id/...")
    parser.add_argument("--label", required=True, choices=("leader", "follower"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    body_mode = MotorNormMode.RANGE_M100_100
    motors = {
        "shoulder_pan": Motor(1, "sts3215", body_mode),
        "shoulder_lift": Motor(2, "sts3215", body_mode),
        "elbow_flex": Motor(3, "sts3215", body_mode),
        "wrist_flex": Motor(4, "sts3215", body_mode),
        "wrist_roll": Motor(5, "sts3215", body_mode),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    try:
        bus.connect(handshake=True)
        models = {name: bus.ping(name, num_retry=2, raise_on_error=True) for name in motors}
        positions = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        print(f"{args.label} bus check: OK")
        print(f"model_numbers={models}")
        print(f"raw_present_positions={positions}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
