# 故障排查

## 校准时报所有 ID 缺失

如果 found motor list 为空，优先检查端口是否插错、12 V 是否接通、USB 是否被其他进程占用，以及 `/dev/serial/by-id/` 链接是否变化。不要在总线未连通时反复标定。

## 相机纯白或过曝

先离开直射阳光，遮光并重新等待自动曝光；确认使用的是 `video-index0`。不要仅凭刚插入时的第一帧判断相机损坏。

## Rerun 1 GiB memory limit

`Exceeded gRPC proxy server memory limit` 表示可视化历史堆积，可能阻塞采集通道。正式录制使用 `--display_data=false`，另行检查保存视频；需要显示时缩短会话或关闭旧的 Rerun 客户端。

## Rollout 卡住

立即 `Ctrl+C`，不要在电机仍使能时硬拉。依次确认无残留 rollout 进程、总线通信、温度、电压和 `Torque_Enable`。参考 [异常退出后的扭矩](rollout_and_safety.md#异常退出后的扭矩)。

## 数据集目录已存在

`FileExistsError` 表示创建模式指向已有目录。确认它是要继续的数据集后才使用 `--resume=true`；若是一次失败创建留下的空目录，应先人工核对内容并换用新的数据集 ID。仓库脚本不会自动删除数据。

## 训练进程消失但有 Traceback

日志中早期 worker Traceback 不一定代表最终失败。检查末尾是否有 `End of training`、所有计划 checkpoint 是否存在、模型文件是否完整，并运行 SHA-256 校验。
