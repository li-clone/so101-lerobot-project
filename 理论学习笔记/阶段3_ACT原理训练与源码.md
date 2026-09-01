# 阶段 3：ACT 原理、训练与源码

## 0. 本阶段定位

ACT 是本轮理论学习的核心。本阶段将阶段 1 的 Behavior Cloning、机器人时序数据，与阶段 2 的 Transformer 连接成完整策略。

完成本阶段后，你应该能够：

- 解释 ACT 为什么预测 action chunk，而不是单个 action。
- 区分 `chunk_size`、`n_action_steps` 和控制频率。
- 解释 ACT 为什么引入 CVAE 和 latent variable。
- 画出 ACT 训练与推理两条不同的数据路径。
- 理解重参数化、L1 loss、KL loss、padding mask。
- 区分动作队列执行与 temporal ensembling。
- 从 `ACTPolicy.forward()` 追踪到总 loss。
- 从 `ACTPolicy.select_action()` 追踪到一个真机 action。
- 判断主要 ACT 配置修改会影响训练、推理还是 checkpoint 形状。

本阶段不要求完整推导概率论，但要求每个公式都能与源码变量对应。

---

## 1. ACT 想解决什么问题

普通单步 Behavior Cloning：

```text
当前 observation oₜ
→ policy
→ 当前动作 aₜ
```

每个控制时刻都重新预测一个动作。它存在几个困难：

- 单步误差不断进入下一时刻，容易累积。
- 每个动作单独预测，时间连续性较弱。
- 长任务包含大量控制步，模型需要连续作出很多正确决策。
- 相似 observation 可能对应不同但都合理的动作风格。

ACT 的核心形式是：

```text
当前 observation oₜ
→ ACT
→ [aₜ, aₜ₊₁, ..., aₜ₊H₋₁]
```

其中 $H$ 就是 action chunk 的长度。

ACT 的主要设计可以概括为：

1. Action Chunking：一次预测未来一段动作。
2. Transformer Encoder-Decoder：融合视觉、状态并并行生成动作位置。
3. CVAE：在训练时编码动作序列的风格或多样性。
4. 动作队列或 Temporal Ensembling：把动作块转成逐步执行的 action。

参考：[ACT 原论文](https://arxiv.org/abs/2304.13705)

---

## 2. Action Chunking

### 2.1 单步预测

```text
oₜ   → aₜ
oₜ₊₁ → aₜ₊₁
oₜ₊₂ → aₜ₊₂
```

模型每一步只考虑一个动作标签。

### 2.2 动作块预测

```text
oₜ
→ [aₜ, aₜ₊₁, aₜ₊₂, ..., aₜ₊H₋₁]
```

模型在一次 forward 中同时学习整段动作之间的关系。

如果 `chunk_size=100`、控制频率为 30 Hz，那么整个预测范围约为：

$$
\frac{100}{30}\approx3.33\text{ 秒}
$$

这里的 3.33 秒是预测范围，不一定表示系统会把 100 个动作全部执行后才重新观察。

---

## 3. Action Chunking 为什么可能有效

### 3.1 学习局部技能片段

单个动作的语义很弱：

```text
某个关节增加 0.3
```

但一段动作可能形成更明确的局部技能：

```text
接近物体
→ 调整夹爪
→ 闭合
→ 抬起
```

### 3.2 建模时间连续性

Decoder action queries 之间通过 self-attention 交换信息，模型可以联合生成更连贯的动作序列。

### 3.3 缩短高层决策链

如果一次 policy forward 负责多个环境步，那么高层重新决策次数减少。

但要注意：

> 执行动作块越长，短期动作可能越连贯；重新观察越少，对突发变化的反应越慢。

所以 action chunking 同时带来平滑性与开环时长之间的取舍。

---

## 4. 三个容易混淆的时间参数

### 4.1 `chunk_size`

模型每次预测多少个未来 action，也是训练 target window 的长度。

```text
predicted_actions shape
= (B, chunk_size, action_dim)
```

### 4.2 `n_action_steps`

在不使用 temporal ensembling 时，每次预测一个 chunk 后，实际取前多少个动作放进执行队列。

例如：

```text
chunk_size = 100
n_action_steps = 20
```

执行逻辑：

```text
预测 100 个动作
→ 只取前 20 个
→ 顺序执行这 20 个
→ 丢弃剩余 80 个
→ 重新观察并再次预测
```

### 4.3 控制频率 FPS

决定每个 action 对应多长物理时间。

如果控制频率为 30 Hz：

```text
n_action_steps = 20
→ 大约每 20/30 = 0.67 秒重新推理一次
```

因此：

```text
预测范围 = chunk_size / FPS
重新规划间隔 = n_action_steps / FPS
```

当前 LeRobot ACT 要求：

```text
n_action_steps <= chunk_size
```

---

## 5. ACT 整体结构

```text
训练输入
├── 多相机图像
├── robot state
└── expert action chunk

expert action chunk + robot state
→ VAE Encoder
→ μ, log(σ²)
→ sample latent z

图像
→ CNN/ResNet
→ visual tokens

[latent token + state token + visual tokens]
→ Main Transformer Encoder
→ context

action position queries
→ Main Transformer Decoder
→ action representations
→ Linear action head
→ predicted action chunk

predicted chunk 与 expert chunk
→ L1 loss

latent posterior 与标准正态分布
→ KL loss

L1 + kl_weight × KL
→ total loss
```

ACT 中存在两个容易混淆的 Encoder：

1. VAE Encoder：只在启用 VAE 的训练路径中编码 expert actions。
2. Main Transformer Encoder：融合视觉、状态和 latent，训练与推理都会使用。

---

## 6. 视觉编码路径

每台相机图像：

```text
(B, 3, H, W)
→ ResNet backbone
→ (B, C, H_f, W_f)
→ 1×1 Conv projection
→ (B, dim_model, H_f, W_f)
→ flatten
→ (H_f × W_f, B, dim_model)
```

双相机的 visual tokens 会沿 sequence 维度拼接。

CNN 负责提取局部视觉特征；Transformer 负责建立：

- 不同视觉区域之间的关系。
- 两个相机之间的关系。
- 视觉与 robot state 之间的关系。

当前 LeRobot ACT 默认视觉 backbone 为带 ImageNet 预训练权重的 ResNet-18，但实际 checkpoint 以保存的 config 为准。

---

## 7. Main Transformer 路径回顾

Main Encoder tokens：

```text
[latent token]
[robot-state token]
[(可选) environment-state token]
[camera 1 visual tokens]
[camera 2 visual tokens]
```

Main Decoder：

```text
chunk_size 个 action position queries
→ decoder self-attention
→ cross-attention encoder context
→ action representations
→ action head
```

输出：

```text
actions_hat shape = (B, chunk_size, action_dim)
```

阶段 2 已经详细学习这部分。本阶段的重点是 latent 从哪里来，以及 output 如何训练和执行。

---

## 8. 为什么 ACT 需要 CVAE

相似 observation 可能对应不同示范风格：

```text
快速接近后减速
缓慢持续接近

先调整腕部再移动肩部
先移动肩部再调整腕部
```

如果确定性模型只学习一个输出，可能把不同模式平均，产生不自然动作。

ACT 使用 Conditional Variational Autoencoder（CVAE），希望用 latent variable $z$ 表达示范动作序列中没有被 observation 完全决定的变化。

可以把 $z$ 暂时理解成：

> 当前这段 expert action chunk 的隐藏风格摘要。

例如，它可能编码动作速度、微小轨迹偏好或示范者风格，但模型没有被显式要求让某一维代表某种人类语义。

---

## 9. Autoencoder 与 VAE 的区别

### 普通 Autoencoder

```text
输入 action chunk
→ Encoder
→ 一个固定 latent vector
→ Decoder
→ 重建 action chunk
```

问题是 latent 空间可能不连续、不规则，推理时很难知道应该从哪里取一个合理 latent。

### Variational Autoencoder

Encoder 不直接输出一个固定 $z$，而是输出一个分布的参数：

```text
μ
log(σ²)
```

表示：

$$
q_\phi(z\mid s,A)=\mathcal N(\mu,\sigma^2)
$$

其中：

- $s$：当前 robot state。
- $A$：expert action chunk。
- $q_\phi$：训练时的近似 posterior。

再从这个分布中采样 latent：

$$
z\sim q_\phi(z\mid s,A)
$$

CVAE 的“Conditional”体现在动作生成还依赖视觉和 robot state 等条件，而不是只依赖 $z$。

---

## 10. VAE Encoder 输入什么

当前 LeRobot ACT 在启用 VAE 的训练路径中，将这些 token 送入 VAE Encoder：

```text
[CLS token]
[robot-state token]
[expert action token 0]
[expert action token 1]
...
[expert action token H-1]
```

源码注释对应：

```text
[cls, robot_state, action_sequence]
```

每个 expert action 先通过线性层：

```text
action_dim
→ dim_model
```

如果：

- batch size = 16。
- chunk size = 100。
- action dim = 6。
- dim model = 512。

那么：

```text
raw action chunk     = (16, 100, 6)
action embeddings    = (16, 100, 512)
state embedding      = (16, 1, 512)
CLS embedding        = (16, 1, 512)
VAE encoder sequence = (16, 102, 512)
```

padding mask 也会在 CLS 和 state 前补两个 `False`：

```text
[False, False] + action_is_pad
```

这样 VAE Encoder 不会把 episode 尾部的 padding actions 当作真实示范。

---

## 11. CLS Token 怎样得到 Latent 分布

经过 VAE Transformer Encoder 后，取 CLS 位置的输出：

```text
CLS output shape = (B, dim_model)
```

再通过线性层输出两组 latent 参数：

```text
latent_pdf_params shape = (B, 2 × latent_dim)
```

拆分为：

```text
μ        shape = (B, latent_dim)
log(σ²)  shape = (B, latent_dim)
```

当前默认 `latent_dim=32`，则：

```text
μ shape       = (B, 32)
log(σ²) shape = (B, 32)
```

为什么取 CLS？

因为 CLS token 通过 self-attention 汇总了 state 和整个 action chunk 的信息，可以作为整段动作的压缩表示。

---

## 12. 重参数化技巧

直接从一个依赖神经网络参数的分布采样，会让普通反向传播难以穿过随机采样操作。

VAE 使用：

$$
\epsilon\sim\mathcal N(0,I)
$$

$$
z=\mu+\sigma\odot\epsilon
$$

而：

$$
\sigma=\exp\left(\frac{1}{2}\log\sigma^2\right)
$$

源码对应：

```python
latent_sample = (
    mu
    + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
)
```

这样随机性来自独立的 $\epsilon$，而 $\mu$、$\sigma$ 仍然位于可微计算图中。

---

## 13. Latent 如何进入 Main Transformer

采样得到：

```text
z shape = (B, latent_dim)
```

经过线性投影：

```text
latent_dim
→ dim_model
```

变成一个 latent token：

```text
latent token shape = (B, dim_model)
```

再与 robot-state token、visual tokens 一起进入 Main Transformer Encoder。

所以训练时的信息流是：

```text
expert action chunk
→ VAE Encoder
→ latent z
→ Main Encoder
→ Main Decoder
→ reconstruct expert action chunk
```

这就是为什么 ACT 的 VAE 路径在训练时看得到 expert action，但最终 action prediction 仍然来自 Main Transformer。

---

## 14. 推理时为什么不能使用 Expert Action

训练时 dataset 提供：

```text
observation + expert action chunk
```

真机推理时只有：

```text
当前 observation
```

并不存在未来 expert action，因此不能运行：

```text
expert action chunk → VAE Encoder → posterior z
```

当前 LeRobot ACT 在非训练路径中，将 latent 设置为全零：

```python
latent_sample = torch.zeros(
    batch_size,
    latent_dim,
)
```

这可以理解为使用 latent prior 的中心或默认风格。

因此存在非常重要的训练/推理差别：

| 路径 | Latent 来源 | 是否有 expert action |
|---|---|---|
| 训练 | VAE Encoder 输出的 posterior 中采样 | 有 |
| 推理 | 全零 latent | 没有 |

KL loss 的作用之一，就是让训练得到的 posterior 不要离标准正态分布太远，使训练 latent 与推理时的默认 prior 区域保持兼容。

---

## 15. KL Divergence 的直觉

ACT 希望训练 posterior：

$$
q_\phi(z\mid s,A)=\mathcal N(\mu,\sigma^2)
$$

接近标准正态 prior：

$$
p(z)=\mathcal N(0,I)
$$

KL divergence 衡量两个分布的差异：

$$
D_{KL}(q_\phi(z\mid s,A)\|p(z))
$$

对于对角高斯，源码计算：

$$
D_{KL}
=
-\frac{1}{2}
\sum_j
\left(
1+\log\sigma_j^2-\mu_j^2-\sigma_j^2
\right)
$$

直觉：

- 如果 $\mu$ 离 0 很远，KL 增大。
- 如果 $\sigma$ 离 1 很远，KL 增大。
- KL 约束 latent 空间保持规则和连续。

---

## 16. ACT 的 Action Reconstruction Loss

模型输出：

```text
actions_hat = (B, chunk_size, action_dim)
```

标签：

```text
batch["action"] = (B, chunk_size, action_dim)
```

当前 LeRobot ACT 使用逐元素 L1：

$$
L_{L1}=
\frac{1}{N_{valid}}
\sum |A-\hat A|
$$

先计算：

```python
abs_err = F.l1_loss(
    batch[ACTION],
    actions_hat,
    reduction="none",
)
```

再使用：

```python
valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
```

屏蔽 episode 尾部的 padding action。

如果：

```text
action_is_pad shape = (B, H)
```

`unsqueeze(-1)` 后：

```text
valid_mask shape = (B, H, 1)
```

它会 broadcast 到所有 action dimensions。

---

## 17. ACT 总 Loss

启用 VAE 时：

$$
L_{ACT}
=
L_{L1}
+
\beta L_{KL}
$$

其中：

```text
β = kl_weight
```

当前默认配置为：

```text
kl_weight = 10.0
```

源码：

```python
loss = l1_loss + mean_kld * self.config.kl_weight
```

如果关闭 VAE：

```text
loss = l1_loss
```

### `kl_weight` 太小

- Posterior 可以离 prior 很远。
- 模型可能高度依赖训练时 action 编码得到的 latent。
- 推理时使用零 latent，可能产生更明显差距。

### `kl_weight` 太大

- Posterior 被过度压向标准正态。
- Latent 可能难以携带动作风格信息。
- 重建动作的能力可能下降。

因此它控制：

```text
动作重建能力
↔
latent 规则性
```

---

## 18. Posterior Collapse 的概念

如果 decoder 足够强，它可能忽略 latent，直接依靠图像和 state 预测动作。

表现为：

```text
不同 action chunk
→ VAE posterior 差异很小
→ latent 对输出影响很弱
```

这种现象称为 posterior collapse。

仅看 KL 数值不能立刻下结论，但如果 KL 很快接近极小值，同时 latent 改变对输出几乎没有影响，就值得检查。

本项目阶段只需要知道这一风险，不需要专门实现复杂的 KL annealing。

---

## 19. 完整训练数据流

```text
LeRobotDataset
→ 当前 observation
→ expert action window
→ action_is_pad

Preprocess
├── image normalization
├── state normalization
└── action normalization

ACTPolicy.forward(batch)
├── 收集多相机到 OBS_IMAGES
└── self.model(batch)

ACT.forward(batch)
├── VAE Encoder([CLS, state, expert actions])
├── μ, log(σ²)
├── reparameterization sample z
├── CNN visual tokens
├── Main Transformer Encoder
├── Main Transformer Decoder
└── predicted action chunk

ACTPolicy.forward
├── masked L1 loss
├── KL loss
└── total loss

total loss
→ backward
→ optimizer.step
→ checkpoint
```

训练阶段的 `forward()` 返回的是：

```text
loss, loss_dict
```

不是直接把预测 action 发送给机器人。

---

## 20. 完整推理数据流

```text
robot.get_observation()
→ image + robot state
→ preprocess / normalization
→ ACTPolicy.select_action(batch)
→ predict_action_chunk(batch)
→ model.eval()
→ latent = zeros
→ CNN + Main Encoder + Main Decoder
→ normalized action chunk
→ queue 或 temporal ensemble
→ 当前一个 normalized action
→ postprocess / inverse normalization
→ safety limit
→ robot.send_action()
```

推理阶段：

- 不需要 expert action。
- 不计算 L1/KL loss。
- 不运行训练时的 VAE Encoder posterior 路径。
- 不执行 backward。
- 使用 `torch.no_grad()`。

---

## 21. 动作队列模式

当 `temporal_ensemble_coeff=None` 时，当前 LeRobot ACT 使用动作队列。

伪代码：

```python
if action_queue is empty:
    chunk = predict_action_chunk(observation)
    selected = chunk[:, :n_action_steps]
    action_queue.extend(selected)

action = action_queue.popleft()
```

例子：

```text
chunk_size = 100
n_action_steps = 20
```

第一次调用：

```text
运行一次模型 forward
→ 得到 100 个动作
→ 取前 20 个进入 queue
→ 返回第 1 个动作
```

接下来 19 次 `select_action()`：

```text
不重新运行模型
→ 依次弹出 queue 中动作
```

队列为空后才重新读取当前 observation 并生成新 chunk。

因此策略层面的重新规划频率由 `n_action_steps` 决定。

---

## 22. Temporal Ensembling

如果每个控制步都重新预测一个 chunk，那么多个历史 chunk 会对同一个未来时刻给出重叠预测。

例如对真实时刻 $t$：

```text
在 t-2 时预测的 chunk：其中第 2 个位置预测 aₜ
在 t-1 时预测的 chunk：其中第 1 个位置预测 aₜ
在 t   时预测的 chunk：其中第 0 个位置预测 aₜ
```

Temporal ensembling 将这些对同一时刻的预测做加权平均：

$$
\bar a_t=
\frac{\sum_i w_i\hat a_t^{(i)}}
{\sum_i w_i}
$$

当前 LeRobot 实现使用指数权重：

$$
w_i=\exp(-m i)
$$

其中 $m$ 是 `temporal_ensemble_coeff`。

当前源码约定：

- `m=0`：均匀权重。
- `m>0`：更偏向较旧预测。
- `m<0`：更偏向较新预测。

这个正负方向容易凭直觉记反，实际使用时应以所安装版本的 `ACTTemporalEnsembler` 注释和代码为准。

Temporal ensembling 的目的：

- 平滑不同 chunk 之间的预测差异。
- 减少每次重新推理导致的动作跳变。
- 利用多个时间点对同一动作的预测。

---

## 23. 为什么 Temporal Ensembling 要求每步推理

要形成重叠预测，policy 必须每个控制步生成一个新 chunk。

因此当前 LeRobot 配置要求：

```text
temporal_ensemble_coeff is not None
→ n_action_steps 必须为 1
```

此时：

```text
每一步读取 observation
→ 预测完整 chunk
→ 与历史重叠预测聚合
→ 执行一个 ensembled action
```

这会增加推理计算量，但提供更密集的视觉反馈和动作平滑。

原始 ACT 工作使用过 `0.01`；当前 LeRobot 默认关闭 temporal ensembling。

---

## 24. 两种执行模式对比

| 模式 | Policy forward 频率 | 优点 | 代价 |
|---|---:|---|---|
| Queue，`n_action_steps > 1` | 每执行若干动作一次 | 推理负担低 | 中间较长时间不重新利用视觉 |
| Queue，`n_action_steps = 1` | 每步一次 | 反馈更频繁 | 不聚合历史预测，计算量较高 |
| Temporal ensemble | 每步一次 | 重叠预测更平滑 | 计算量和状态管理更高 |

不要把“预测 100 个动作”与“必须执行 100 个动作”混为一谈。

---

## 25. 当前 ACT 主要配置

以下是当前 LeRobot main 分支默认值，用于理解字段；你的工程必须以 checkpoint 中的实际 config 为准。

| 参数 | 当前默认值 | 作用 |
|---|---:|---|
| `n_obs_steps` | 1 | 当前实现只处理一个 observation step |
| `chunk_size` | 100 | 预测及训练 action window 长度 |
| `n_action_steps` | 100 | Queue 模式每个 chunk 执行多少步 |
| `vision_backbone` | `resnet18` | 图像特征提取器 |
| `dim_model` | 512 | Transformer token 维度 |
| `n_heads` | 8 | Attention heads |
| `n_encoder_layers` | 4 | Main Encoder 层数 |
| `n_decoder_layers` | 1 | Main Decoder 层数 |
| `use_vae` | `True` | 是否启用 variational objective |
| `latent_dim` | 32 | Latent 维度 |
| `n_vae_encoder_layers` | 4 | VAE Encoder 层数 |
| `kl_weight` | 10.0 | KL loss 权重 |
| `temporal_ensemble_coeff` | `None` | 是否启用 temporal ensembling |
| `dropout` | 0.1 | Transformer dropout |

当前默认 normalization：

```text
VISUAL → MEAN_STD
STATE  → MEAN_STD
ACTION → MEAN_STD
```

参考：[LeRobot ACTConfig](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)

---

## 26. 归一化与 Action Head

ACT 训练时通常不是直接预测原始电机尺度，而是在 normalization 后的 action 空间计算 loss。

```text
raw expert action
→ normalize
→ ACT target
```

推理时：

```text
ACT normalized output
→ inverse normalization
→ robot action scale
```

所以 action head 输出的 6 个数是否能直接理解为电机角度，要看 postprocessor 和 normalization stats。

如果归一化统计错误，即使模型内部输出合理，也可能在真机上变成错误尺度。

部署所需的不只是 `model.safetensors`，还包括：

- `config.json`。
- feature 定义及顺序。
- normalization processor/statistics。
- 与训练一致的相机 key。

---

## 27. 源码类的职责

### `ACTPolicy`

LeRobot 策略外壳，负责：

- 训练用 `forward()`。
- 推理用 `predict_action_chunk()`。
- 单步执行用 `select_action()`。
- 动作 queue。
- temporal ensembler。
- loss 计算。

### `ACT`

底层神经网络，负责：

- VAE Encoder。
- CNN backbone。
- Main Encoder-Decoder。
- action head。
- 输出 action chunk 和 latent distribution 参数。

### `ACTEncoder` / `ACTEncoderLayer`

同时被用于：

- VAE Encoder。
- Main Transformer Encoder。

具体用途通过 `is_vae_encoder` 区分。

### `ACTDecoder` / `ACTDecoderLayer`

负责 action queries 的 self-attention、对 encoder context 的 cross-attention 和 FFN。

### `ACTTemporalEnsembler`

维护不同时间预测的重叠 action，并返回当前聚合动作。

---

## 28. 训练源码调用链

```text
lerobot-train
→ DataLoader batch
→ preprocess batch
→ ACTPolicy.forward(batch)
    ├── 整理 OBS_IMAGES
    ├── actions_hat, (mu, log_var) = self.model(batch)
    ├── masked L1
    ├── KL
    └── return total loss, loss_dict
→ loss.backward()
→ optimizer.step()
→ 保存 checkpoint
```

阅读入口：

- `ACTPolicy.forward()`：先看 loss。
- `ACT.forward()`：再看完整网络。
- `ACTConfig.action_delta_indices`：理解 Dataset 为什么取 `chunk_size` 个 action。

---

## 29. 推理源码调用链

Queue 模式：

```text
rollout loop
→ ACTPolicy.select_action(batch)
→ queue 是否为空？
    ├── 否：popleft()
    └── 是：predict_action_chunk(batch)
             → ACT.forward in eval mode
             → latent=zeros
             → action chunk
             → 取前 n_action_steps 入队
→ 返回当前一个 action
→ postprocess
→ robot.send_action()
```

Temporal ensemble 模式：

```text
select_action(batch)
→ 每步 predict_action_chunk
→ temporal_ensembler.update(chunk)
→ 返回当前聚合 action
```

参考：[LeRobot ACT 实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)

---

## 30. 一个完整 Shape 例子

假设：

- batch size = 16。
- action dim = 6。
- chunk size = 100。
- latent dim = 32。
- dim model = 512。
- 两台相机各产生 300 visual tokens。

### VAE 路径

```text
expert actions       (16, 100, 6)
action embeddings    (16, 100, 512)
CLS + state + action (16, 102, 512)
CLS output           (16, 512)
μ                     (16, 32)
log(σ²)               (16, 32)
sample z              (16, 32)
latent token          (16, 512)
```

### Main Transformer

```text
visual tokens         600
state token             1
latent token            1
encoder sequence      602
encoder input         (602, 16, 512)
decoder input         (100, 16, 512)
decoder output        (100, 16, 512)
actions_hat           (16, 100, 6)
action_is_pad         (16, 100)
```

### Loss

```text
elementwise abs_err   (16, 100, 6)
valid mask            (16, 100, 1)
μ / log variance      (16, 32)
total loss            scalar
```

---

## 31. 如何理解训练日志

### `l1_loss` 下降

说明在训练或验证 batch 上，normalized action chunk 更接近 expert action。

它不直接证明：

- 真机成功率提高。
- 长时间 rollout 更稳定。
- 模型具备恢复能力。

### `kld_loss`

反映 posterior 与标准正态 prior 的差异。

- 很大：latent posterior 可能远离 prior。
- 很小：可能是良好正则化，也可能 latent 被忽略。

必须结合 action reconstruction、validation 和 rollout 观察。

### Total loss

```text
total = l1 + kl_weight × kld
```

如果修改 `kl_weight`，不同实验的 total loss 数值不能直接比较，因为目标函数本身已经改变。

---

## 32. 常见失败现象与理论关联

### 动作抖动

可能关联：

- Chunk 之间预测不连续。
- 每次重规划差异较大。
- 数据动作本身不平滑。
- normalization 或控制接口问题。
- 可考虑检查 temporal ensembling，但不能先假设一定是模型结构。

### 对变化反应太慢

可能关联：

- `n_action_steps` 太大，视觉反馈使用不够频繁。
- 动作队列仍在执行旧 observation 下的预测。

### 动作流畅但方向错误

可能关联：

- 数据覆盖或视觉泛化问题。
- 相机 feature 顺序或预处理不一致。
- Action chunk 学到了连贯但错误的轨迹。

### 训练 loss 低但部署失败

可能关联：

- Distribution shift。
- 训练/推理 latent 路径差异。
- 数据泄漏或过拟合。
- Action 尺度、归一化或关节顺序错误。
- 关键抓取时刻在平均 loss 中占比很小。

### Episode 末尾训练异常

可能关联：

- Padding mask 未正确处理。
- 短 episode 中有效 action 比例较低。

---

## 33. 参数修改的依赖关系

### 修改 `chunk_size`

会同时影响：

- Dataset action window。
- `action_is_pad` 长度。
- VAE Encoder 序列长度。
- Decoder action queries 数量。
- `decoder_pos_embed` 参数形状。
- Action output shape。

因此旧 checkpoint 通常不能无条件加载到不同 `chunk_size` 的结构。

### 修改 `n_action_steps`

主要改变 queue 模式的执行与重规划间隔，不改变模型 action head 的输出长度。

但它会改变真实 rollout 分布和推理频率。

### 修改 `temporal_ensemble_coeff`

主要改变推理动作聚合；当前实现启用时要求 `n_action_steps=1`。

### 修改 `latent_dim`

改变 VAE 输出层和 latent projection 的权重形状，影响 checkpoint 兼容。

### 修改 `kl_weight`

不改变网络 shape，但改变优化目标和 latent 使用方式。

### 修改 `dim_model` 或 `n_heads`

改变 Transformer 大量权重形状；`dim_model` 必须能够合理分配到各 attention heads。

---

## 34. 建议的源码消融思路

在建立 baseline 后，理论上可以设计：

- `use_vae=True` 与 `False`：验证 latent objective 的作用。
- 不同 `n_action_steps`：比较反馈频率与动作连续性。
- Queue 与 temporal ensemble：比较平滑度和推理成本。
- 不同 `chunk_size`：比较预测范围和学习难度。
- 不同 `kl_weight`：观察重建与 latent regularization。

每次只改变一个主要变量，并保持：

- 数据集。
- 随机种子或至少记录种子。
- 训练步数。
- 评估初始条件。
- 成功定义。

否则无法把结果可靠归因于某个 ACT 设计。

---

## 35. 常见误区

### 误区 1：ACT 预测 100 个动作，就必须全部执行

错误。执行多少由 `n_action_steps` 或 temporal ensemble 决定。

### 误区 2：CVAE Encoder 在真机推理时也读取未来 expert actions

错误。推理时没有 expert actions，当前实现使用零 latent。

### 误区 3：Latent 是语言指令

错误。ACT latent 是从训练 action chunk 中学习的隐变量，不是自然语言 token。

### 误区 4：KL loss 越小越好

错误。过小可能意味着 latent 没有携带有效信息。

### 误区 5：Temporal ensembling 就是执行完整 chunk

错误。它每步生成新 chunk，并聚合同一时刻的重叠预测。

### 误区 6：`n_action_steps` 与 `chunk_size` 完全相同

错误。前者是执行长度，后者是预测和训练目标长度。

### 误区 7：Action head 输出一定是原始舵机角度

错误。它通常处于 normalized action space，之后还要 postprocess。

### 误区 8：训练 total loss 可以跨不同 `kl_weight` 直接比较

错误。Loss 定义已经发生变化。

---

## 36. 本阶段知识图

```text
                  训练专用路径
expert action chunk + state
            ↓
       VAE Encoder
            ↓
       μ, log(σ²)
            ↓ reparameterization
          latent z
            │
            ▼
images → CNN → visual tokens
state  → Linear → state token
latent → Linear → latent token
            ↓
     Main Transformer Encoder
            ↓ context
     Main Transformer Decoder
            ↑ action position queries
            ↓
     predicted action chunk
       │              │
       │              └→ 推理：queue / temporal ensemble
       │                              ↓
       │                        current action
       │
       └→ 训练：masked L1

posterior → KL to N(0,I)

total loss = L1 + kl_weight × KL
```

---

## 37. 阶段 3 自测

建议先回答问题 1～3，我们再逐步讨论源码和配置。

### 问题 1：Action Chunk

假设：

- `chunk_size=100`。
- `n_action_steps=20`。
- 控制频率为 30 Hz。

请回答：

1. Policy 每次 forward 预测多长时间范围的动作？
2. Queue 模式多久重新运行一次 policy？
3. 剩余 80 个动作如何处理？

### 问题 2：训练与推理 Latent

请解释：

1. 训练时 latent $z$ 从哪里来？
2. 推理时为什么不能使用同样的 VAE Encoder 路径？
3. 当前 LeRobot ACT 推理时使用什么 latent？

### 问题 3：VAE Shape

假设：

- batch size = 16。
- chunk size = 100。
- action dim = 6。
- dim model = 512。
- latent dim = 32。

请写出：

1. Expert action chunk shape。
2. Action embeddings shape。
3. `[CLS, state, actions]` 的 VAE Encoder input shape。
4. $\mu$ 和 $\log\sigma^2$ 的 shape。

### 问题 4：重参数化

为什么不直接写“从 $\mathcal N(\mu,\sigma^2)$ 采样”，而要写：

$$
z=\mu+\sigma\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

它对反向传播有什么帮助？

### 问题 5：Loss

如果：

```text
l1_loss = 0.08
kld_loss = 0.03
kl_weight = 10
```

总 loss 是多少？

如果把 `kl_weight` 改为 1，总 loss 数值下降，能否直接说明模型变好了？

### 问题 6：Padding Mask

为什么 `action_is_pad` 从 `(B, H)` 变成 `(B, H, 1)` 后，可以屏蔽 `(B, H, action_dim)` 的 L1 error？

### 问题 7：Queue

`chunk_size=100`、`n_action_steps=20` 时，连续调用 25 次 `select_action()`，模型 forward 通常会执行几次？分别在哪些调用附近发生？

### 问题 8：Temporal Ensembling

请解释多个历史 chunk 为什么会对同一个真实时刻产生重叠预测，以及为什么对它们加权平均可能让动作更平滑。

### 问题 9：配置依赖

下面哪些参数会改变 checkpoint 中参数张量的 shape，哪些主要改变推理行为？

```text
chunk_size
n_action_steps
latent_dim
temporal_ensemble_coeff
kl_weight
```

### 问题 10：源码路径

请分别说明以下方法的职责：

- `ACTPolicy.forward()`。
- `ACT.forward()`。
- `ACTPolicy.predict_action_chunk()`。
- `ACTPolicy.select_action()`。

---

## 38. 参考资料

- [ACT 原论文（arXiv）](https://arxiv.org/abs/2304.13705)
- [ACT 论文正式版本（RSS 2023）](https://www.roboticsproceedings.org/rss19/p016.pdf)
- [LeRobot ACTConfig](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)
- [LeRobot ACT 实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)
- [LeRobot PreTrainedPolicy 接口](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py)

