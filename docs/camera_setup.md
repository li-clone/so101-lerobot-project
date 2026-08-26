# 双摄像头配置

本项目固定两路语义：

- `handeye`：第一视角，需持续看到夹爪、黄色线束和黑色放置区。
- `environment`：第三视角，需看到完整工作区、机械臂和目标区域。

## 识别稳定路径

```bash
lsusb
v4l2-ctl --list-devices
ls -l /dev/video* /dev/v4l/by-id/ /dev/v4l/by-path/
```

优先使用 `/dev/v4l/by-id/`；同型号设备缺少唯一 ID 时使用 `/dev/v4l/by-path/`。不要把易变化的 `/dev/video2`、`/dev/video4` 写入正式配置。

实验记录的格式为：

| 相机 | 分辨率 | 设备 FPS | fourcc | 数据集 FPS |
|---|---:|---:|---|---:|
| handeye | 640×480 | 30 | YUYV | 20 |
| environment | 640×480 | 30 | MJPG | 20 |

正式采集前使用 LeRobot 的带视觉遥操作或系统相机查看器确认画面。强光导致过曝时先遮光、拉帘或调整工作区；采集过程中不得改变机位、焦距、曝光环境和相机语义。

公开第三视角视频前必须裁掉人物和家庭环境，详见 [media/README.md](../media/README.md)。
