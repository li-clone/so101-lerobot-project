# ACT 数据采集

## 任务与一致性

固定任务文本：

```text
Pick up the fixed yellow cable bundle and place it in the black target area.
```

每条示教应包含接近、抓取、抬起、移动、放置和退出。动作保持平滑，但不刻意放慢到与部署时明显不同。每条结束后将机械臂和场景恢复到训练分布内的安全初始状态。

## 运行采集脚本

```bash
export PROJECT_ROOT="$PWD"
export FOLLOWER_PORT=/dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
export LEADER_PORT=/dev/serial/by-id/<LEADER_SERIAL_DEVICE>
export HANDEYE_CAMERA=/dev/v4l/by-path/<HANDEYE_VIDEO_INDEX0>
export ENVIRONMENT_CAMERA=/dev/v4l/by-id/<ENVIRONMENT_VIDEO_INDEX0>
export DATA_ROOT="$PROJECT_ROOT/data"
export DATASET_ID=local/so101_pick_place_v2_supplement
export DATASET_ROOT="$DATA_ROOT/so101_pick_place_v2_supplement"
export NUM_EPISODES=10
export RESUME=false

bash scripts/dataset/record_act_batch.sh
```

- 单条最长 60 s，复位 15 s，20 FPS。
- `n` 提前结束当前 episode 并进入复位阶段。
- `r` 重录当前 episode；`Esc`/`q` 结束会话，以终端实际提示为准。
- 仅在继续同一个已存在且结构完整的数据集时设置 `RESUME=true`。

## 质量门禁

每批采集后检查两路视频、episode 数量、帧数和 parquet 可读性。抓空、目标出画、严重过曝、遮挡关键动作或异常碰撞的示教应重录；不能只保留画面好看的成功片段而忽略数据动作质量。

v2 数据由 v1 episodes 0–49 与针对远近距离补采的 episodes 50–69 合并。黄色线束方向保持竖直且与目标区边缘平行，主要改变距离，以解决 v1 的远距离覆盖不足。
