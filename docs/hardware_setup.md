# 环境与硬件

## 固定版本

本项目使用 LeRobot `v0.6.1`，commit：

```text
7e241bd630a3719a56157a497ce5d08f244784f1
```

推荐 Python 3.12，并在独立 Conda 环境中安装。不要修改系统 Python，也不需要为了 PyTorch wheel 单独安装系统 CUDA Toolkit。

```bash
conda create -n lerobot-so101 python=3.12 ffmpeg -c conda-forge
conda activate lerobot-so101
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e './upstream/lerobot[core_scripts,training,feetech]'
python -m pip check
```

CUDA wheel 要与本机驱动兼容；`cu128` 是本实验的记录值，不是所有机器的强制选择。

## 端口识别

每次插线后先只读识别，不要根据 `/dev/ttyACM0` 的编号猜测主从臂。

```bash
ls -l /dev/serial/by-id/
python scripts/diagnostics/check_motor_bus.py \
  --label follower --port /dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
python scripts/diagnostics/check_motor_bus.py \
  --label leader --port /dev/serial/by-id/<LEADER_SERIAL_DEVICE>
```

将确认后的值写入被 Git 忽略的 `configs/hardware_ports.local.yaml`。公开仓库只保留 `hardware_ports.example.yaml`。

## 电机 ID

| ID | 关节 |
|---:|---|
| 1 | `shoulder_pan` |
| 2 | `shoulder_lift` |
| 3 | `elbow_flex` |
| 4 | `wrist_flex` |
| 5 | `wrist_roll` |
| 6 | `gripper` |

厂商 PDF 未发现明确的再分发许可，因此不放入公共仓库。需要时从设备供应商的正式渠道获取，并结合 LeRobot `v0.6.1` 官方 SO-101 文档使用。
