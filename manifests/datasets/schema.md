# Dataset schema summary

所有 ACT 训练与评测数据均为 LeRobot dataset codebase version `v3.0`，20 FPS，SO Follower。

| key | dtype / storage | shape | names |
|---|---|---|---|
| `observation.state` | float32 | `(6,)` | shoulder pan/lift, elbow flex, wrist flex/roll, gripper |
| `action` | float32 | `(6,)` | 同上 |
| `observation.images.handeye` | H.264 video | `(480,640,3)` | RGB |
| `observation.images.environment` | H.264 video | `(480,640,3)` | RGB |
| `timestamp` | float32 | `(1,)` | — |
| `frame_index` | int64 | `(1,)` | — |
| `episode_index` | int64 | `(1,)` | — |
| `index` | int64 | `(1,)` | — |
| `task_index` | int64 | `(1,)` | — |

视频为 `yuv420p`、无音频、CRF 23、preset fast；相机采集30 FPS，写入数据集时同步为20 FPS。
