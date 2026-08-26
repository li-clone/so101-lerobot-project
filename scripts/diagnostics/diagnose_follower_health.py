#!/usr/bin/env python3
"""Read-only health and configuration snapshot for SO-101 Follower motors."""

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
REGISTERS = (
    "Torque_Enable", "Status", "Present_Voltage", "Present_Temperature",
    "Present_Current", "Present_Load", "Present_Position", "Goal_Position_2",
    "P_Coefficient", "I_Coefficient", "D_Coefficient", "Max_Torque_Limit",
    "Torque_Limit", "Protection_Current",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()
    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    snapshot: dict[str, dict[str, int | str]] = {name: {} for name in MOTORS}
    try:
        bus.connect(handshake=True)
        for register in REGISTERS:
            for motor in MOTORS:
                try:
                    snapshot[motor][register] = bus.read(register, motor, normalize=False, num_retry=2)
                except Exception as error:
                    snapshot[motor][register] = f"ERROR: {error}"
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    for motor, values in snapshot.items():
        print(f"\n[{motor}] id={MOTORS[motor].id}")
        for register in REGISTERS:
            value = values[register]
            if register == "Present_Voltage" and isinstance(value, (int, float)):
                print(f"{register}={value} (approximately {value / 10:.1f} V)")
            elif register == "Status" and isinstance(value, int):
                print(f"{register}={value} (0x{value:02x})")
            else:
                print(f"{register}={value}")


if __name__ == "__main__":
    main()
