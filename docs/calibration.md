# 机械臂标定

标定顺序为 Leader → Follower。清空机械臂运动范围，确认设备身份后再运行。

```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<LEADER_SERIAL_DEVICE> \
  --teleop.id=so101_leader_main

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE> \
  --robot.id=so101_follower_main
```

按终端提示让除 `wrist_roll` 之外的关节依次覆盖完整安全行程。不要为了得到更大的 min/max 强行撞机械限位。

## 对齐检查

把主从臂摆到大致相同的安全姿态，执行只读检查：

```bash
python scripts/diagnostics/check_calibrated_alignment.py \
  --leader-port /dev/serial/by-id/<LEADER_SERIAL_DEVICE> \
  --follower-port /dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
```

该输出用于发现方向、零点和明显幅度异常，不是要求所有关节数值完全相等。标定 JSON 包含具体硬件状态，只保存在本机 LeRobot calibration cache，不上传 GitHub。
