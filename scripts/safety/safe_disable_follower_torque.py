#!/usr/bin/env python3
"""Best-effort torque shutdown that verifies every Follower motor independently."""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--num-retry", type=int, default=5)
    args = parser.parse_args()
    if args.passes < 1 or args.num_retry < 0:
        raise ValueError("--passes must be >= 1 and --num-retry must be >= 0")

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    state: dict[str, int | str] = {name: "NOT CHECKED" for name in MOTORS}
    pending = set(MOTORS)
    try:
        bus.connect(handshake=False)
        for pass_index in range(1, args.passes + 1):
            if not pending:
                break
            print(f"Disable pass {pass_index}/{args.passes}: {sorted(pending)}")
            for motor in list(MOTORS):
                if motor not in pending:
                    continue
                try:
                    bus.disable_torque(motor, num_retry=args.num_retry)
                except Exception as error:
                    print(f"WARN {motor}: disable write failed: {error}")
                try:
                    value = bus.read("Torque_Enable", motor, normalize=False, num_retry=args.num_retry)
                    state[motor] = int(value)
                    if value == 0:
                        pending.remove(motor)
                except Exception as error:
                    state[motor] = f"READ ERROR: {error}"
                    print(f"WARN {motor}: verification read failed: {error}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    print(f"\n{'JOINT':<16} {'ID':>3} {'TORQUE_ENABLE':>15} {'RESULT':>10}")
    for motor, definition in MOTORS.items():
        result = "OFF" if state[motor] == 0 else "FAILED"
        print(f"{motor:<16} {definition.id:>3} {str(state[motor]):>15} {result:>10}")
    if pending:
        print(f"\nRESULT: FAILED TO VERIFY TORQUE OFF: {sorted(pending)}")
        raise SystemExit(1)
    print("\nRESULT: ALL FOLLOWER TORQUE OFF")


if __name__ == "__main__":
    main()
