# SO-101 × LeRobot：双视角 ACT 抓取放置

这个仓库记录一套可复现的 SO-101 模仿学习闭环：硬件检查、双摄像头采集、ACT 训练、实机 rollout、失败分析和针对性补数。当前公开阶段覆盖 ACT v1 与 ACT v2。

## 四个学习项目

| 阶段 | 项目 | 学习重点 |
|---|---|---|
| 1 | **ACT（本仓库）** | Transformer Action Chunking，建立单指令双视角基线 |
| 2 | [Diffusion Policy](https://github.com/li-clone/so101-lerobot-diffusion-policy) | 迭代去噪动作生成与闭环恢复行为 |
| 3 | [SmolVLA](https://github.com/li-clone/so101-lerobot-smolvla) | 语言条件、多目标区域与布局互换 |
| 4 | [MuJoCo Pick-and-Place](https://github.com/li-clone/mujoco-panda-pick-place-learning) | MuJoCo Jacobian、3D DLS IK、关节 PD 与内置 6D OSC 对比 |

导航体现学习演进，不是算法排行榜。前三个项目是 SO-101 真机项目，第 4 个是 Franka Panda 仿真项目；四者的数据、任务难度、评测协议和 loss 定义不同，结果不能直接横向比较。

任务描述：`Pick up the fixed yellow cable bundle and place it in the black target area.`

## 结果摘要

| 版本 | 训练数据 | 最佳 checkpoint | 最佳 eval loss | 部署参数 | 实机结果 |
|---|---:|---:|---:|---|---:|
| ACT v1 | 50 episodes / 16,881 frames | 45k | 0.3095 | `n_action_steps=100` | 8/10（80%） |
| ACT v2 | 70 episodes / 23,868 frames | 30k | 0.2643 | `n_action_steps=50` | 13/15（86.7%） |

v1 与 v2 的验证集和实机评测协议不同，上表不能作为严格的横向性能证明。它记录的是两个阶段各自的观察结果；详见 [v1 结果](docs/act_v1_results.md) 与 [v2 结果](docs/act_v2_results.md)。

![ACT validation curves](results/figures/act_eval_curves.svg)

![Hardware rollout success rate](results/figures/hardware_success_rate.svg)

## ACT 与 Diffusion Policy 对比

两组实验共享 SO-101、手眼与第三视角双摄像头、640×480 输入、20 FPS、6 维关节状态与动作、70 条演示、50k 训练步、batch size 16、seed 1000，以及近/中/远各 5 次的实机评测结构。

| 对比项 | ACT v2 | Diffusion Policy v1 |
|---|---|---|
| 训练数据 | 70 episodes / 23,868 frames | 70 episodes / 21,016 frames |
| 核心模型 | CVAE + Transformer | 条件 1D U-Net + DDPM |
| 视觉编码 | 双视角 ResNet-18 | 双视角独立 ResNet-18 |
| 历史观测 | 1 帧 | 2 帧 |
| 可学习参数 | 51,597,190 | 277,819,846 |
| 状态/动作归一化 | Mean/Std | Min/Max |
| 优化器学习率 | `1e-5` | `1e-4` |
| 最佳已保存点 | 30k | 5k |
| 部署动作数 | `n_action_steps=50` | `n_action_steps=32` |
| 推理方式 | 一次前向生成动作块 | 10 次迭代去噪生成动作块 |
| 计算特征 | 延迟低，更接近目标控制频率 | 计算量更大，需减少去噪步数 |

### 实机结果

| 策略 | 近 | 中 | 远 | 总计 |
|---|---:|---:|---:|---:|
| ACT v2（30k） | 4/5 | 5/5 | 4/5 | 13/15（86.7%） |
| Diffusion v1（5k） | 5/5 | 5/5 | 5/5 | **15/15（100%）** |

Diffusion 的 15 次成功中，12 次首次抓取成功；3 次首次抓取失败后均自主回抓成功。ACT v2 的失败包括一次近距离抓取后推动黑色目标盒，以及一次远距离重复空抓仍未成功。

本结果是本次两套相似视角、相同距离分布实验中的观察结果。两者训练数据并非同一数据集，摄像头机位也不是逐像素一致；每个策略只有 15 次实机试验。因此 `15/15` 与 `13/15` 不能推广为 Diffusion Policy 在一般任务上必然优于 ACT。不同策略的 eval loss 定义和尺度不同，也不能直接横向比较。Diffusion 的完整训练、10步去噪部署和逐次评测证据位于对应独立仓库。

## 硬件与软件

- SO-101 Leader + Follower，Feetech STS3215，6 DoF
- 手眼相机 + 第三视角相机，640×480
- Ubuntu、Python 3.12、CUDA GPU
- LeRobot `v0.6.1`，固定 commit `7e241bd630a3719a56157a497ce5d08f244784f1`
- ACT，ResNet-18 双视角输入，动作维度 6

## 快速开始

```bash
git clone --recurse-submodules https://github.com/li-clone/so101-lerobot-project.git
cd so101-lerobot-project
git submodule update --init --recursive
test "$(git -C upstream/lerobot rev-parse HEAD)" = \
  "7e241bd630a3719a56157a497ce5d08f244784f1"

cp configs/hardware_ports.example.yaml configs/hardware_ports.local.yaml
```

然后按本机设备修改 `hardware_ports.local.yaml`。不要把本地端口、校准文件或凭据提交到 Git。

## 文档导航

- [环境与硬件](docs/hardware_setup.md)
- [机械臂标定](docs/calibration.md)
- [双摄像头配置](docs/camera_setup.md)
- [数据采集](docs/data_collection.md)
- [ACT 训练](docs/training.md)
- [Rollout 与安全](docs/rollout_and_safety.md)
- [故障排查](docs/troubleshooting.md)
- [数据与模型上传清单](docs/artifact_upload.md)
- [公开视频处理](media/README.md)

## 安全警告

实机推理前清空工作区，准备急停或断电手段，先用短时 rollout 验证方向和初始姿态。异常退出后必须检查 `Torque_Enable`；默认断连失败时使用：

```bash
python scripts/safety/safe_disable_follower_torque.py \
  --port /dev/serial/by-id/<FOLLOWER_SERIAL_DEVICE>
```

若脚本无法确认全部关扭矩，立即切断 Follower 12 V 电源。不要手动强扭仍处于使能状态的关节。

## 数据与许可证

GitHub 只保存代码、小型摘要、校验清单和脱敏媒体，不包含原始数据、模型权重、完整日志、校准文件或厂商 PDF。数据与模型的建议存放方式见 [artifact_upload.md](docs/artifact_upload.md)。本仓库自有代码使用 [MIT License](LICENSE)；LeRobot、厂商资料和其他第三方内容仍适用各自许可证。
