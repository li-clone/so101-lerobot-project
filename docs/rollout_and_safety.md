# Rollout、评测与安全

## 推理前门禁

1. 清空运动范围并固定黑色目标盒。
2. 检查 Follower 总线和两路摄像头。
3. 确认当前姿态位于训练初始姿态范围。
4. 首次加载模型只做短时测试，确认方向和限幅。
5. 准备随时 `Ctrl+C` 和切断 Follower 12 V 电源。

## 单次 rollout

```bash
export FOLLOWER_PORT=/dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
export HANDEYE_CAMERA=/dev/v4l/by-path/<HANDEYE_VIDEO_INDEX0>
export ENVIRONMENT_CAMERA=/dev/v4l/by-id/<ENVIRONMENT_VIDEO_INDEX0>
export POLICY_PATH="$OUTPUT_ROOT/act_so101_pick_place_v2_70_v2_50k/checkpoints/030000/pretrained_model"
export N_ACTION_STEPS=50
export DURATION=60

bash scripts/evaluation/rollout_act.sh
```

v1 正式配置使用 `N_ACTION_STEPS=100`；v2 使用 `50`。`n_action_steps` 表示一次模型预测后连续执行的动作数量，值更小会更频繁地重新观察和规划。本实验的 `max_relative_target=None` 通过不传该可选限幅参数实现；不要把字符串 `none` 当作数值传入。

## 正式评测

- 预先定义近、中、远三组和成功判据。
- 每次只改变计划中的物体距离，不移动摄像头和目标盒。
- 无论成功失败都保留 trial，并记录尝试次数与失败原因。
- 成功：线束被抓起并稳定放入黑色目标区域。
- 抓到但推走目标盒、运输途中掉落或超时均记为失败。

需要同时保存视频和动作时，使用 episodic 评测入口；目标目录必须不存在，继续同一评测数据集时才设置 `RESUME=true`：

```bash
export EVAL_DATASET_ID=local/rollout_act_v2_n50_balanced_eval_15
export EVAL_DATASET_ROOT="$DATA_ROOT/evaluation/rollout_act_v2_n50_balanced_eval_15"
export NUM_EPISODES=5
export RESUME=false
bash scripts/evaluation/record_rollout_batch.sh
```

## 异常退出后的扭矩

某次断连曾在关闭 ID 3 时连续通信失败，导致后续关节未完成默认顺序卸载。不要强扭电机。先检查：

```bash
python scripts/diagnostics/diagnose_follower_health.py \
  --port /dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
```

若仍为 `Torque_Enable=1`，运行逐电机容错卸载：

```bash
python scripts/safety/safe_disable_follower_torque.py \
  --port /dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
```

脚本只有在六个电机全部读回 `0` 时才返回成功。否则立即断开 12 V 电源，等待降温并检查串口、电源和接线。
