# 阶段 7：Diffusion Policy 原理与 LeRobot 源码

## 0. 本阶段定位

前面我们已经学习了两类机器人策略：

```text
普通 BC：observation → 单步 action

ACT：observation → 未来 action chunk
```

这一阶段学习第三种重要方法：

```text
Diffusion Policy：
observation + 随机动作轨迹
→ 多次条件去噪
→ 合理的未来动作轨迹
```

它把图像扩散模型的思想，从“生成像素”迁移到了“生成动作序列”。

本阶段目标不是推完扩散模型的全部概率论，而是做到：

1. 能解释为什么直接回归可能产生错误的平均动作。
2. 能区分机器人时间步和扩散去噪时间步。
3. 能说清 Diffusion Policy 的训练与推理流程。
4. 能读懂动作加噪、噪声预测、MSE loss 和 scheduler。
5. 能解释 observation 如何成为去噪条件。
6. 能区分 `n_obs_steps`、`horizon`、`n_action_steps`。
7. 能比较 ACT 与 Diffusion Policy 的共同点和差异。
8. 能沿 LeRobot 源码定位训练 loss 和推理 action queue。

---

## 1. 从图像 Diffusion 连接到 Action Diffusion

### 1.1 图像生成

图像扩散模型的直觉是：

```text
随机噪声图像
→ 逐步去噪
→ 合理图像
```

模型学习的不是“直接画出整张图”，而是：

```text
当前带噪图像
+ 扩散时间步
+ 条件（例如文字）
→ 应该去掉的噪声
```

### 1.2 动作生成

Diffusion Policy 将被生成的数据从图像换成动作轨迹：

```text
随机动作序列
→ 逐步去噪
→ 符合当前观察的动作序列
```

条件也发生了变化：

```text
图像生成的条件：文字、类别等

动作生成的条件：
相机图像 + 机器人状态 + 可选环境状态
```

因此核心分布是：

$$
p(\mathbf A_t\mid\mathbf O_t)
$$

其中：

- $\mathbf O_t$：当前及历史 observation；
- $\mathbf A_t$：一段动作轨迹；
- Policy 要学习在该观察条件下，哪些动作轨迹是合理的。

---

## 2. 为什么不直接用 MSE 回归动作

假设同一个场景中，绕过障碍物有两种正确方法：

```text
轨迹 A：从左侧绕开
轨迹 B：从右侧绕开
```

普通确定性回归模型如果使用 MSE，可能倾向于预测条件均值：

```text
左侧轨迹
＋
右侧轨迹
→
中间轨迹
```

但中间可能正好撞上障碍物。

这就是多模态动作分布：

$$
p(\mathbf A\mid\mathbf O)
$$

可能有多个高概率区域，而不是只有一个正确答案。

Diffusion Policy 不要求一次前向传播输出唯一均值，而是尝试学习整个条件动作分布，并从中采样一条一致轨迹。

理想情况下：

```text
某次采样 → 完整选择左侧方案
另一次采样 → 完整选择右侧方案
```

而不是把两条路线逐维平均。

但要注意：

> “能够表达多模态”不等于“任何小数据集都一定能学好所有模式”。

数据数量、模式比例、视觉可区分性、模型容量与训练质量仍然重要。

---

## 3. 两条完全不同的时间轴

这是本阶段最重要的概念。

Diffusion Policy 同时存在两种“时间步”。

### 3.1 机器人时间 $t$

表示真实 rollout 中的时间：

```text
t=0：机械臂开始运动
t=1：发送下一个动作
t=2：继续运动
...
```

动作序列可写作：

$$
\mathbf A_t=
[\mathbf a_t,\mathbf a_{t+1},\ldots,\mathbf a_{t+H-1}]
$$

这里相邻位置对应真实执行中的相邻控制步。

### 3.2 扩散时间 $k$

表示一条动作轨迹被加了多少噪声，或正在进行第几次去噪：

```text
k=0：干净专家动作轨迹
k较小：只有少量噪声
k较大：噪声很多
k=K：接近随机高斯噪声
```

它不是机械臂实际执行的时间。

### 3.3 正确区分

可以把一段动作写成：

$$
\mathbf A_t^k
$$

其中：

- 下标 $t$：这段轨迹在机器人 rollout 中从哪里开始；
- 上标 $k$：这段轨迹当前有多少扩散噪声。

同一个张量中：

```text
横向位置：未来机器人动作 a_t, a_{t+1}, ...
去噪迭代：A^K → A^{K-1} → ... → A^0
```

必须避免下面的误解：

> 去噪 100 次，不代表机械臂执行了 100 个动作。

100 次去噪可能只是为了生成一次 action chunk，然后才从该 chunk 中取出若干动作交给机器人执行。

---

## 4. 张量形状

假设：

- batch size：$B=16$；
- action horizon：$H=64$；
- action dimension：$D=6$。

干净动作轨迹形状为：

```text
clean_actions: (16, 64, 6)
```

随机噪声形状完全相同：

```text
epsilon: (16, 64, 6)
```

加噪后的动作轨迹仍然是：

```text
noisy_actions: (16, 64, 6)
```

噪声预测网络输出也通常保持相同形状：

```text
predicted_noise: (16, 64, 6)
```

扩散不会改变动作序列的基本形状，只会改变张量中数值的“干净程度”。

若每个 batch 样本独立随机选择一个扩散时间步：

```text
timesteps: (16,)
```

例如：

```text
[3, 81, 45, 20, ...]
```

意味着 batch 中不同轨迹可以接受不同强度的噪声。

---

## 5. Forward Diffusion：给专家动作加噪

### 5.1 逐步形式

前向扩散可以想象为：

$$
\mathbf A^0
\rightarrow
\mathbf A^1
\rightarrow
\mathbf A^2
\rightarrow
\cdots
\rightarrow
\mathbf A^K
$$

$\mathbf A^0$ 是干净专家动作，噪声逐渐增加。

单步常写为：

$$
q(\mathbf A^k\mid\mathbf A^{k-1})
=
\mathcal N
\left(
\sqrt{1-\beta_k}\mathbf A^{k-1},
\beta_k\mathbf I
\right)
$$

$\beta_k$ 控制第 $k$ 步加入多少噪声。

### 5.2 训练时不需要真的逐步加噪

通过累积系数：

$$
\alpha_k=1-\beta_k
$$

$$
\bar\alpha_k=\prod_{i=1}^{k}\alpha_i
$$

可以直接从干净动作得到任意噪声等级：

$$
\mathbf A^k
=
\sqrt{\bar\alpha_k}\mathbf A^0
+
\sqrt{1-\bar\alpha_k}\boldsymbol\epsilon
$$

其中：

$$
\boldsymbol\epsilon\sim\mathcal N(0,\mathbf I)
$$

这意味着训练一个 batch 时可以：

```text
随机选 k
→ 一步构造 A^k
→ 让网络预测所加噪声
```

而不必真的依次计算：

```text
A⁰ → A¹ → A² → ... → Aᵏ
```

这点非常重要：

> 训练时通常随机学习某一个噪声等级；推理时才需要从高噪声开始多步迭代去噪。

---

## 6. Condition：模型凭什么知道该生成什么动作

如果只给随机动作噪声，不给 observation，模型只能学习整个数据集的总体动作分布：

$$
p(\mathbf A)
$$

它不知道：

- 杯子在哪里；
- 机械臂当前在哪里；
- 应该抓哪个方向；
- 当前处于抓取的哪一阶段。

因此 Diffusion Policy 学习的是条件分布：

$$
p(\mathbf A\mid\mathbf O)
$$

噪声预测器接收：

$$
\boldsymbol\epsilon_\theta
(\mathbf A^k,k,\mathbf O)
$$

三个核心输入分别是：

1. 当前带噪动作轨迹 $\mathbf A^k$；
2. 扩散时间步 $k$；
3. observation 条件 $\mathbf O$。

### 6.1 为什么必须输入 $k$

同样一段带噪数据，在不同噪声等级下，去噪策略不同。

```text
k很大：动作几乎全是噪声，需要大幅修正
k很小：动作已经接近真实轨迹，只需精细调整
```

如果不告诉网络当前 $k$，网络无法判断自己面对的是强噪声还是弱噪声。

### 6.2 Observation condition 包含什么

在视觉机器人策略中通常包括：

```text
一个或多个相机图像
+
当前及最近若干步机器人 state
+
可选环境状态
```

视觉编码器先将图像转换为特征，随后与关节状态等信息组合成 condition。

---

## 7. Diffusion Policy 的训练流程

训练时我们已经拥有专家动作轨迹。

完整流程是：

```text
1. 读取 observation history O
2. 读取干净专家动作轨迹 A⁰
3. 采样 Gaussian noise ε
4. 随机采样 diffusion timestep k
5. 根据 scheduler 构造 noisy action Aᵏ
6. 网络根据 (Aᵏ, k, O) 预测噪声 ε_hat
7. 计算 ε_hat 与真实 ε 的 MSE
8. backward + optimizer step
```

公式为：

$$
\hat{\boldsymbol\epsilon}
=
\boldsymbol\epsilon_\theta
(\mathbf A^k,k,\mathbf O)
$$

$$
\mathcal L_{diffusion}
=
\left\|
\boldsymbol\epsilon
-
\boldsymbol\epsilon_\theta(\mathbf A^k,k,\mathbf O)
\right\|_2^2
$$

### 7.1 模型为什么预测噪声

因为训练时噪声是我们自己随机生成并加入的，所以真实标签天然已知：

```text
输入：带噪动作
标签：刚刚加入的噪声 ε
```

这仍然是一种 supervised learning，只不过监督目标不再是直接动作，而是人为构造的噪声。

### 7.2 预测噪声不是唯一参数化

扩散模型也可以预测：

- 原始干净样本 $\mathbf A^0$；
- velocity 参数化；
- 其他等价目标。

当前 LeRobot Diffusion 配置支持 `prediction_type="epsilon"` 或 `"sample"`；默认配置使用噪声预测 `epsilon`。这属于实现选择，不改变“学习条件动作分布”的总体思想。

---

## 8. Diffusion Policy 的推理流程

推理时没有专家未来动作，所以不能从干净轨迹加噪。

它从随机高斯动作轨迹开始：

$$
\mathbf A^K\sim\mathcal N(0,\mathbf I)
$$

然后反复去噪：

$$
\mathbf A^K
\rightarrow
\mathbf A^{K-1}
\rightarrow
\cdots
\rightarrow
\mathbf A^0
$$

每一步执行：

```text
当前 noisy action Aᵏ
+ 当前 observation condition O
+ diffusion timestep k
→ 预测噪声
→ scheduler 计算 Aᵏ⁻¹
```

最终得到：

```text
denoised action trajectory: (B, horizon, action_dim)
```

再取其中计划执行的动作放入 queue。

### 8.1 推理不等于 backward

反向扩散中的“reverse”不是深度学习的反向传播。

```text
Reverse diffusion：从噪声逐步生成动作
Backward：根据 loss 计算参数梯度
```

推理阶段通常使用 `torch.no_grad()`，不会更新模型参数。

### 8.2 多次去噪的代价

ACT 通常一次模型 forward 就产生一个 action chunk。

Diffusion Policy 为了产生一个 chunk，需要多次调用去噪网络：

```text
一个 action chunk
≈ K 次 denoising network forward
```

因此它通常有更高推理开销。减少推理去噪步数可以加速，但可能影响生成质量；具体结果取决于 scheduler、模型与任务。

---

## 9. Noise Scheduler 的作用

Scheduler 决定：

### 训练时

- 每个 $k$ 对应多少噪声；
- 如何从 $\mathbf A^0$ 构造 $\mathbf A^k$。

### 推理时

- 使用哪些去噪时间步；
- 如何根据模型输出从 $\mathbf A^k$ 更新到 $\mathbf A^{k-1}$。

可以把它理解为：

```text
模型：告诉你噪声大概是什么
Scheduler：告诉你下一步应该怎么更新样本
```

常见调度器包括 DDPM 与 DDIM。

### 9.1 DDPM 与 DDIM 的初步区别

现阶段只需知道：

- DDPM 通常保留随机采样过程；
- DDIM 可以用更少的推理步，并可设为更确定性的采样；
- 二者训练思想相近，但推理更新规则不同。

不要把：

```text
num_train_timesteps
```

与：

```text
num_inference_steps
```

当成同一个概念。

训练可以定义较多噪声等级，而推理选择其中一部分时间步加速采样。

---

## 10. 为什么动作必须归一化

假设六个 action 维度的范围差异很大：

```text
关节1：[-90°, 90°]
关节2：[-30°, 120°]
夹爪：[0, 1]
```

如果直接加标准高斯噪声：

- 对角度维度，噪声可能太小；
- 对夹爪维度，噪声可能太大；
- MSE 会被数值范围大的维度主导。

因此通常先将动作归一化到相近范围，例如：

$$
[-1,1]
$$

然后在归一化空间中：

```text
加噪
→ 去噪
→ sample clipping
→ 反归一化
→ 发送真实机器人 action
```

这也解释了为什么推理中的 `clip_sample` 必须与 action normalization 匹配。

---

## 11. 网络结构：视觉编码器 + 条件 1D U-Net

Diffusion Policy 论文研究了 CNN 和 Transformer 两类去噪网络。当前 LeRobot 内置 Diffusion Policy 的主干是条件 1D U-Net。

总体结构可以理解为：

```text
camera images
→ ResNet visual encoder
→ visual features
                       ┐
robot state history ───┼→ global condition
                       ┘

noisy action trajectory
+ diffusion timestep embedding
+ global condition
→ conditional 1D U-Net
→ predicted noise / predicted clean sample
```

### 11.1 为什么是 1D U-Net

图像 U-Net 沿图像高度和宽度处理空间结构。

动作 U-Net 主要沿时间轴处理动作序列：

```text
a_t, a_{t+1}, ..., a_{t+H-1}
```

所以这里的 1D 主要指时间维度上的卷积和上下采样。

### 11.2 U-Net 的直觉

U-Net 通过：

```text
下采样：扩大时间感受野，理解整段动作结构
上采样：恢复每个时间位置的细节
skip connection：保留局部动作信息
```

最终为 action trajectory 中每个时间步、每个 action 维度预测噪声。

### 11.3 FiLM condition

LeRobot 的条件 U-Net 使用 FiLM 思想将 observation 特征注入网络。

直觉上不是简单地只在输入处拼接一次 condition，而是利用 condition 调整中间特征：

$$
\mathbf h'=gamma(\mathbf c)\odot\mathbf h+\beta(\mathbf c)
$$

其中：

- $\mathbf h$：U-Net 中间特征；
- $\mathbf c$：视觉、状态和扩散时间等条件；
- $\gamma,\beta$：由条件生成的缩放与偏置。

这让去噪过程持续受到当前 observation 的引导。

---

## 12. 三个 Horizon 概念

Diffusion Policy 论文用三个时间范围平衡历史信息、动作连续性和反馈速度。

### 12.1 Observation Horizon

记作 $T_o$，对应 LeRobot 的：

```text
n_obs_steps
```

表示模型使用多少个最近 observation。

例如：

```text
n_obs_steps = 2
```

模型会使用当前帧和前一帧的信息。

历史观察有助于判断运动趋势和任务阶段，但也增加输入与计算量。

### 12.2 Prediction Horizon

记作 $T_p$，在 LeRobot 配置中对应：

```text
horizon
```

表示扩散模型共同生成多长的动作时间序列。

较长 prediction horizon 有助于整段轨迹一致，但输出维度、计算量和远期不确定性也会增加。

### 12.3 Execution Horizon

记作 $T_a$，对应 LeRobot 的：

```text
n_action_steps
```

表示一次生成后真正放入执行队列并连续执行多少步。

```text
horizon：模型生成的范围
n_action_steps：实际采用的范围
```

二者不必相等。

### 12.4 为什么生成 64 步却只执行 32 步

远期动作帮助模型形成有方向的一致计划，但越远的预测越容易过时。

因此可以：

```text
看过去少量 observation
→ 生成较长未来轨迹
→ 只执行靠近当前的一段
→ 再观察并重新生成
```

这就是 receding horizon control，中文常称滚动时域控制或递推时域控制。

---

## 13. Receding Horizon：既规划又反馈

假设：

```text
horizon = 64
n_action_steps = 16
control frequency = 30 Hz
```

一次推理生成 64 个动作，但只执行 16 个：

$$
\frac{16}{30}\approx0.53\text{ s}
$$

随后重新读取 observation 并规划。

因此：

```text
16步内部：Policy层近似开环
16步结束：使用新 observation 重规划
舵机层：始终有自己的位置闭环
```

这与 ACT queue 模式的控制层级非常相似。

### 13.1 Execution horizon 太长

- 推理频率下降；
- 旧计划执行更久；
- 环境变化后反应较慢。

### 13.2 Execution horizon 太短

- 重新规划更频繁；
- 推理计算压力更大；
- 若推理延迟高，控制周期可能无法维持；
- 不同采样结果之间可能产生边界变化。

---

## 14. 当前 LeRobot 默认配置的阅读方法

截至本讲义生成时，LeRobot `main` 分支的 `DiffusionConfig` 默认包含：

```python
n_obs_steps = 2
horizon = 64
n_action_steps = 32

vision_backbone = "resnet18"
noise_scheduler_type = "DDPM"
num_train_timesteps = 100
prediction_type = "epsilon"
```

这些数值是帮助你阅读源码的“当前默认值”，不是对 SO-101 任务的固定最优配置；软件版本、任务、频率、数据量和硬件变化后都可能需要调整。

若 action dimension 为 6，则训练 action 张量形状可理解为：

```text
(B, 64, 6)
```

推理时模型先生成完整的 `(B,64,6)`，再从当前时间对应的位置取出 `32` 步供 action queue 使用。

---

## 15. LeRobot 训练源码链

当前源码中可以沿下面的逻辑阅读：

```text
DiffusionPolicy.forward(batch)
→ DiffusionModel.compute_loss(batch)

batch:
  observation.state
  observation.images / environment_state
  action
  action_is_pad

→ encode observation as global_cond
→ trajectory = batch["action"]
→ eps = randn_like(trajectory)
→ random timesteps k
→ noisy_trajectory = scheduler.add_noise(trajectory, eps, k)
→ pred = unet(noisy_trajectory, k, global_cond)
→ target = eps                  # prediction_type="epsilon"
→ MSE(pred, target)
```

对应张量：

```text
trajectory:       (B, horizon, action_dim)
eps:              (B, horizon, action_dim)
noisy_trajectory: (B, horizon, action_dim)
pred:             (B, horizon, action_dim)
```

这条链和你熟悉的训练流程完全一致：

```text
Dataset
→ DataLoader
→ Policy forward
→ Diffusion loss
→ backward
→ optimizer
→ checkpoint
```

只是 label 从“直接 action”变成了“人为加入的噪声”。

---

## 16. Padding 与 Loss

在 episode 末尾，未来动作数量可能不够 `horizon`，Dataset 需要进行 padding。

这与 ACT 中的固定长度 action chunk 问题相同。

LeRobot batch 可以包含：

```text
action_is_pad: (B, horizon)
```

当前配置提供：

```text
do_mask_loss_for_padding
```

若开启，逻辑为：

```text
valid mask:           (B, horizon)
unsqueeze:            (B, horizon, 1)
broadcast over D:     (B, horizon, action_dim)
```

然后只对有效动作位置计算噪声预测误差。

不过当前 LeRobot 默认值为 `False`，源码注释说明这是为了保持与原始 Diffusion Policy 实现的行为一致。因此阅读实验时必须检查实际配置，不能仅凭一般经验断言 padding 一定被 mask。

---

## 17. LeRobot 推理源码链

推理生成的核心逻辑是：

```text
sample = Gaussian noise

for k in scheduler.timesteps:
    model_output = unet(sample, k, global_cond)
    sample = scheduler.step(model_output, k, sample).prev_sample

return denoised action trajectory
```

外层 Policy 逻辑为：

```text
select_action(observation)
→ 保存最近 n_obs_steps 个 observation

如果 action queue 为空：
    → predict_action_chunk()
    → diffusion.generate_actions()
    → 生成 horizon 长度轨迹
    → 截取 n_action_steps
    → 加入 action queue

从 queue 弹出一个 action
→ postprocess / unnormalize
→ robot.send_action()
```

因此 `select_action()` 每次只返回一个动作，不代表扩散模型每次只生成一个动作。

---

## 18. 随机性从哪里来

Diffusion Policy 推理通常从随机噪声开始：

$$
\mathbf A^K\sim\mathcal N(0,\mathbf I)
$$

因此同一个 observation 使用不同随机种子，可能生成不同动作轨迹。

这使它能够表达多个动作模式，但也带来问题：

- 不同 chunk 之间可能选中不同模式；
- rollout 可重复性下降；
- 调试时必须记录随机种子；
- 多模态并不总是好事，有时任务只允许一个精确动作。

随机性并不是“随便行动”。去噪过程仍受到 observation 和训练分布约束。

可以理解为：

```text
随机噪声决定从分布的哪里开始采样
observation condition 将采样引导到当前场景的合理动作区域
```

---

## 19. ACT 与 Diffusion Policy 对比

### 19.1 共同点

二者都属于模仿学习，并且都可以：

- 输入图像与机器人 state；
- 预测未来动作序列；
- 使用 action chunk；
- 使用 queue 执行部分动作；
- 通过重新观察形成 Policy 层闭环；
- 依赖高质量 demonstration 数据。

### 19.2 核心区别

| 维度 | ACT | Diffusion Policy |
|---|---|---|
| 动作生成 | Transformer decoder 一次生成 chunk | 从噪声多步去噪生成 chunk |
| 分布建模 | CVAE latent 表达变化 | 扩散采样表达复杂分布 |
| 常见训练目标 | Action L1 + KL | Noise/sample MSE |
| 推理次数 | 通常一次主模型 forward | 多次 denoising forward |
| 推理随机性 | 取决于 latent 用法；常见部署可用固定 $z=0$ | 初始噪声通常带来采样随机性 |
| 序列结构 | Action queries + Transformer | 时间 1D U-Net 或 Transformer |
| 推理成本 | 相对较低 | 通常较高 |
| 多模态表达 | 由 CVAE latent 提供 | 由条件生成分布与采样提供 |

### 19.3 不能简单说谁一定更好

效果取决于：

- 数据量与一致性；
- 任务是否明显多模态；
- 控制频率和延迟预算；
- GPU 推理能力；
- action 表示；
- 图像质量和相机稳定性；
- 超参数与训练时间。

对小规模真实机器人数据而言，模型更复杂并不自动意味着成功率更高。

---

## 20. Diffusion Policy 是否使用 FK / IK

与关节空间 ACT 类似，关节空间 Diffusion Policy 通常也不显式调用 FK 或 IK：

```text
image + joint state + noisy joint trajectory
→ denoising network
→ future joint trajectory
```

它从示教数据中隐式学习视觉、当前姿态和关节动作之间的关系。

因此：

- 它的源码不一定包含显式运动学求解器；
- 它仍然受真实机械臂运动学约束；
- 它不会天然保证目标可达、无碰撞或远离奇异点；
- 训练分布外的姿态仍可能失败。

如果 action space 本身定义为末端位姿或末端增量，则系统下游仍可能需要 IK 或笛卡尔控制器将其转换成关节命令。是否显式使用 IK 取决于 action representation 和控制接口，而不只是模型名字。

---

## 21. 常见误区

### 误区 1：Diffusion Policy 给图片加噪

错误。它通常将 observation 作为条件，主要对未来 action trajectory 加噪和去噪。

### 误区 2：训练时每个样本都要完整运行 100 次加噪

错误。训练时通常随机选择一个 $k$，直接构造对应的 noisy action。

### 误区 3：推理时先有正确动作，再给它去噪

错误。推理时没有专家动作，从随机高斯动作轨迹开始。

### 误区 4：扩散时间步就是机器人控制时间步

错误。二者是两条不同时间轴。

### 误区 5：去噪 100 步就会执行 100 个动作

错误。100 是去噪迭代数；最终 action chunk 长度由 `horizon` 等参数决定。

### 误区 6：`horizon=64` 就一定执行64步

错误。实际一次保留执行多少步由 `n_action_steps` 决定。

### 误区 7：Diffusion 一定不会输出平均动作

错误。它具有更强的多模态表达能力，但数据、训练或模型失败时仍可能生成不理想轨迹。

### 误区 8：Reverse diffusion 就是 backward

错误。Reverse diffusion 是采样；backward 是梯度计算。

### 误区 9：Loss 低就说明真机一定成功

错误。还存在视觉分布变化、控制延迟、动作裁剪、动力学、数据覆盖等问题。

### 误区 10：Diffusion Policy 不需要动作归一化

错误。合理的归一化对噪声尺度、loss 平衡和 sample clipping 都非常重要。

---

## 22. 常见失败现象与理论分析

### 情况 A：生成动作整体抖动

可能原因：

- demonstration 本身不平滑；
- 去噪质量不足；
- 推理步数过少；
- normalization 或 scheduler 不匹配；
- chunk 边界采样不一致；
- 视觉输入噪声；
- 底层控制或机械间隙。

### 情况 B：每次 rollout 路径差异很大

可能来自扩散采样随机性，也可能说明条件约束不够强或数据存在多个模式。

分析时应固定随机种子，区分：

```text
模型随机性
vs
真实环境随机性
```

### 情况 C：训练 loss 下降，但动作像随机数

优先检查：

- action normalization；
- prediction type 与 target 是否一致；
- scheduler 配置；
- 推理是否完整执行 reverse steps；
- checkpoint 与 pre/postprocessor 是否匹配。

### 情况 D：动作方向基本正确，但反应很迟钝

可能原因：

- `n_action_steps` 太大；
- inference latency 太高；
- observation history 或 action queue 使用了旧数据；
- 远期计划执行过久。

### 情况 E：同一场景有时左绕、有时右绕，切换时撞物体

模型可能学会多个模式，但 chunk 之间缺少模式一致性。

可以从以下方向思考：

- 缩短执行 horizon；
- 增强条件信息；
- 使示教策略更一致；
- 保持随机种子用于诊断；
- 使用带历史动作或模式记忆的设计。

---

## 23. 本阶段知识图

```text
TRAINING
========

Observation O ────────────────┐
                              │ condition
Expert action A⁰              ▼
      │                  Denoising Network
      │ add random noise       │
      ▼                        ▼
Noisy action Aᵏ ─────────► predicted ε_hat
      ▲                        │
      │                        ▼
random k, ε              MSE(ε_hat, ε)


INFERENCE
=========

Latest observation history O
              │
              ▼
Gaussian action noise Aᴷ
              │
              ▼
    Aᴷ → Aᴷ⁻¹ → ... → A⁰
       repeated denoising
              │
              ▼
Full action horizon
              │ take n_action_steps
              ▼
         Action queue
              │ one action each control step
              ▼
           Robot
              │
              └── new observation → replan
```

---

## 24. 阶段 7 自测

建议先回答问题 1～3，再继续后面的源码与比较题。

### 问题 1：两条时间轴

模型生成形状为 `(B,64,6)` 的动作轨迹，并使用 100 个 diffusion timesteps。

1. `64` 表示什么？
2. `100` 表示什么？
3. 为什么不能说“机械臂会执行100个动作”？

### 问题 2：训练张量

已知：

```text
B = 16
horizon = 64
action_dim = 6
```

请写出以下张量形状：

1. `clean_actions`；
2. `epsilon`；
3. `noisy_actions`；
4. `predicted_noise`；
5. 每个 batch 样本独立采样一个 `timestep` 时，`timesteps` 的形状。

### 问题 3：训练流程排序

请按正确顺序排列：

```text
A. 计算 predicted_noise 与真实 epsilon 的 MSE
B. 从 Dataset 读取干净专家动作
C. 随机采样 diffusion timestep k
D. 网络根据 noisy action、k 和 observation 预测噪声
E. 采样 Gaussian noise epsilon
F. 使用 scheduler 构造 noisy action
```

并回答：训练时是否必须为一个样本完整运行100次去噪网络？

### 问题 4：推理

推理时没有专家动作。

1. 初始动作轨迹从哪里来？
2. observation 在推理中起什么作用？
3. Reverse diffusion 与 `loss.backward()` 是否是同一过程？

### 问题 5：多模态

同一 observation 下，专家一半从障碍物左侧绕行，一半从右侧绕行。

1. 普通 MSE 确定性 BC 可能预测什么？
2. 为什么该结果可能失败？
3. Diffusion Policy 理想情况下如何处理？
4. 为什么不能因此断言 Diffusion Policy 一定成功？

### 问题 6：三个 Horizon

配置为：

```text
n_obs_steps = 2
horizon = 64
n_action_steps = 16
control_frequency = 30 Hz
```

1. 模型使用几个 observation 时间步？
2. 一次生成多少个动作位置？
3. 一次连续执行多少步后重新生成？
4. Policy 大约每隔多少秒重规划一次？
5. 理想情况下 Policy 推理频率约为多少 Hz？

### 问题 7：源码理解

补全：

```text
trajectory = batch["action"]
epsilon = __________
timestep = __________
noisy_trajectory = scheduler.__________(trajectory, epsilon, timestep)
pred = unet(noisy_trajectory, timestep, global_cond)
loss = MSE(pred, __________)
```

假设 `prediction_type="epsilon"`。

### 问题 8：ACT 与 Diffusion Policy

请分别比较：

1. 动作 chunk 如何生成；
2. 训练 loss；
3. 推理计算量；
4. 多模态来源；
5. 二者在 action queue 和 Policy 层开闭环方面有什么共同点。

### 问题 9：Normalization

假设五个关节 action 使用角度范围 `[-100,100]`，夹爪 action 使用 `[0,1]`。

1. 为什么直接在原始空间加入同尺度 Gaussian noise 不合理？
2. 为什么未经归一化时 MSE 可能主要被关节角度维度控制？
3. 推理生成 normalized action 后，在发送给机器人前还要经过什么过程？

### 问题 10：综合判断

判断正误并解释：

1. Diffusion Policy 对相机图片逐步加噪，从而生成动作。
2. 训练时的 noise 是随机生成的，因此 noise prediction 仍可视为监督学习。
3. `num_inference_steps=20` 表示机器人执行20个动作。
4. `horizon=64`、`n_action_steps=16` 表示生成64步但本轮只执行其中16步。
5. 同一个 observation 使用不同随机噪声，可能生成不同动作轨迹。
6. Diffusion Policy 不显式调用 IK，说明真实机械臂不再受运动学约束。
7. 执行 horizon 越长，Policy 对突发环境变化的反应通常越快。
8. ACT 与 Diffusion Policy 都可以进行 action chunking，但动作生成机制不同。

---

## 25. 本阶段完成标准

如果你能不看讲义解释下面这段话，就完成了阶段7：

> Diffusion Policy 学习条件动作分布 $p(\mathbf A\mid\mathbf O)$。训练时从专家动作轨迹出发，随机选择扩散时间步并加入已知高斯噪声，网络根据 noisy action、diffusion timestep 和 observation 预测噪声，通过 MSE 更新参数。推理时没有专家动作，而是从随机动作噪声开始多步去噪，得到完整 action horizon，只执行其中 `n_action_steps`，再用新 observation 重规划。扩散时间步描述去噪过程，不是机器人真实执行时间。

---

## 26. 参考资料

- [Diffusion Policy 论文（arXiv）](https://arxiv.org/abs/2303.04137)
- [Diffusion Policy 官方项目](https://diffusion-policy.cs.columbia.edu/)
- [LeRobot Diffusion 配置源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py)
- [LeRobot Diffusion 模型源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/modeling_diffusion.py)
- [LeRobot Diffusion 训练示例](https://github.com/huggingface/lerobot/blob/main/examples/tutorial/diffusion/diffusion_training_example.py)
