# 阶段 8：VLA 与 SmolVLA

## 0. 本阶段定位

前面我们已经学习了：

~~~text
视觉 + 机器人状态
→ ACT
→ 未来关节动作序列
~~~

ACT能够学习一个具体任务，但它通常没有明确的语言任务输入。例如同一场景中同时存在红色积木和蓝色积木，仅凭图像无法知道用户想抓哪一个。

VLA加入语言条件：

~~~text
Vision
+
Language
+
Robot State
→
Action
~~~

本阶段目标不是立即部署VLA，而是做到：

1. 理解VLM与VLA的关系。
2. 理解语言为什么能消除任务歧义。
3. 知道VLA不只有一种动作生成方式。
4. 理解预训练、微调和从头训练的区别。
5. 看懂SmolVLA的VLM与Action Expert两部分。
6. 理解SmolVLA如何用Flow Matching生成连续action chunk。
7. 能区分Flow Matching时间与机器人控制时间。
8. 能读懂SmolVLA训练与推理的源码主链。
9. 理解多机器人、多任务预训练为什么困难。
10. 知道VLA的能力边界，不把语言理解误认为可靠执行。

---

## 1. 为什么机器人需要语言

假设相机中同时存在：

~~~text
红色杯子
蓝色杯子
收纳盒
桌面边缘
~~~

没有语言时，Policy只能根据训练分布猜测任务：

~~~text
image + robot state
→ 抓哪个物体？
→ 放到哪里？
~~~

加入指令后：

~~~text
“把蓝色杯子放进收纳盒”
~~~

任务变得更明确。

语言可以表达：

- 目标物体：蓝色杯子；
- 动作类型：拿起、推动、打开；
- 目标位置：盒子、桌边；
- 属性约束：最大的、左侧的；
- 动作顺序：先拿杯子，再关抽屉。

因此VLA学习的是类似下面的条件策略：

$$
\pi(\mathbf A\mid\mathbf I,\mathbf L,\mathbf s)
$$

其中：

- I：一个或多个相机图像；
- L：语言指令；
- s：机器人状态；
- A：未来动作或动作序列。

语言的核心作用是提供task condition，而不是代替视觉和机器人状态。

---

## 2. VLM复习

VLM是Vision-Language Model。

典型输入输出是：

~~~text
图片 + 文字
→ 多模态语义表示或文字回答
~~~

例如：

~~~text
图片：桌面上有一个红色杯子
问题：“杯子是什么颜色？”
输出：“红色”
~~~

一个简化VLM包括：

~~~text
Image
→ Vision Encoder
→ Visual Tokens
                    ┐
Language Tokens ────┼→ Multimodal Transformer
                    ┘
→ Language Output
~~~

### 2.1 Vision Encoder

将图像转换为视觉特征：

$$
\mathbf I
\rightarrow
[\mathbf v_1,\mathbf v_2,\ldots,\mathbf v_N]
$$

每个visual token表示图像某部分或压缩后的视觉信息。

### 2.2 Tokenizer与Language Embedding

指令：

~~~text
Pick up the blue cube
~~~

先被切分成token id，再映射为embedding：

$$
[w_1,w_2,\ldots,w_M]
\rightarrow
[\mathbf e_1,\mathbf e_2,\ldots,\mathbf e_M]
$$

### 2.3 Multimodal Fusion

Transformer让语言token与视觉token发生信息交互，从而理解：

~~~text
“blue cube”
对应图像里的哪个区域
~~~

VLM擅长语义理解，但普通VLM的输出通常是文字，不是可以直接写给舵机的连续控制命令。

---

## 3. 从VLM到VLA

VLA的关键变化是把机器人动作纳入模型输出能力。

~~~text
VLM：
image + language
→ text

VLA：
image + language + robot state
→ robot action
~~~

机器人状态为什么仍然重要？

相同图像与指令下，如果机械臂当前位置不同，下一步动作也应不同：

~~~text
夹爪在杯子左侧
→ 向右靠近

夹爪在杯子右侧
→ 向左靠近
~~~

所以VLA并不是简单的：

~~~text
图片 + 语言 → 动作
~~~

更准确的是：

~~~text
视觉理解
+
任务语义
+
当前身体状态
→
与当前机器人匹配的动作
~~~

---

## 4. VLA不是单一架构

不同VLA最显著的差别之一，是如何表示和生成action。

### 4.1 离散Action Token

将连续动作量化为离散区间：

~~~text
joint value
→ quantization bin
→ action token
~~~

然后像语言模型生成文字一样自回归生成：

$$
p(z_1,z_2,\ldots,z_K\mid I,L)
=
\prod_i p(z_i\mid z_{<i},I,L)
$$

RT-2和经典OpenVLA路线具有这种思想。

优点：

- 可复用语言模型的token预测框架；
- 文字与动作可放入相似输出空间；
- 架构统一。

代价：

- 量化可能损失连续控制精度；
- 自回归要逐token生成，可能较慢；
- 前一个token错误会影响后续token；
- token序列还要解码回连续动作。

### 4.2 连续Action Expert

另一类方法保留连续动作：

~~~text
VLM
→ 提供视觉和语言上下文

Action Expert
→ 生成连续action chunk
~~~

Action Expert可以使用：

- 回归；
- Diffusion；
- Flow Matching；
- 其他连续生成模型。

SmolVLA属于这一类：

> SmolVLA不是让语言模型逐个输出离散动作token，而是使用Flow Matching Action Expert生成连续动作序列。

### 4.3 为什么不能用“是否有语言”判断具体结构

VLA只描述总体输入输出范式：

~~~text
Vision + Language → Action
~~~

它不规定：

- action必须离散还是连续；
- 必须自回归还是并行生成；
- 必须使用Diffusion还是Flow Matching；
- 必须输出单步还是chunk；
- 必须使用哪一种VLM。

阅读论文时要继续追问：

~~~text
视觉怎么编码？
语言怎么进入？
机器人state怎么进入？
动作如何表示？
动作解码器如何训练？
推理要运行几次网络？
~~~

---

## 5. 为什么使用预训练VLM

从头训练ACT时，模型主要从你的机器人数据学习视觉与动作关系。

而VLA通常从预训练VLM出发。VLM在大规模图文数据中已经学习了一些：

- 物体与属性；
- 颜色和形状；
- 空间关系；
- 语言指令；
- 常见场景语义。

机器人预训练再让模型学习：

~~~text
理解物体和语言
→ 转换为机器人的控制行为
~~~

因此可以粗略表示为：

~~~text
大规模图文预训练
→ VLM

VLM
+ 多任务机器人数据
→ VLA Base Model

VLA Base Model
+ 你的SO-101任务数据
→ Task-specific Policy
~~~

但必须避免过度理解：

> VLM知道“杯子是什么”，不等于它自然知道SO-101每个舵机应该转多少。

语义知识与控制能力是两件事，动作能力仍需机器人轨迹数据训练。

---

## 6. Pretraining、Fine-tuning与From Scratch

### 6.1 从头训练ACT

~~~text
随机初始化Policy
+ 你的demonstrations
→ 学会具体任务
~~~

### 6.2 从头构建SmolVLA架构

“从头”通常也不代表所有参数完全随机。可能是：

~~~text
预训练VLM
+
新初始化Action Expert
+
机器人数据
→ VLA
~~~

### 6.3 微调SmolVLA Base

~~~text
已经完成机器人预训练的SmolVLA Base
+
你的数据
→ 适配你的相机、机器人和任务
~~~

这利用了两层先验：

1. VLM的视觉—语言知识；
2. 多任务机器人数据中的视觉—动作和控制知识。

### 6.4 Fine-tuning不是简单记住新标签

微调需要同时完成：

- 适应相机位置和画面；
- 适应你的robot state/action语义；
- 将任务文字与实际轨迹关联；
- 保留有用的预训练知识；
- 避免小数据过拟合。

如果数据接口、动作顺序或归一化错误，预训练不会自动修复这些工程语义。

---

## 7. Robot Foundation Model是什么意思

Foundation model不是“万能模型”的同义词。

更准确的含义是：

~~~text
在较多任务、环境或机器人数据上预训练
→ 学到可迁移的通用表示或行为先验
→ 再适配下游任务
~~~

机器人基础模型希望具备：

- 多任务；
- 多物体；
- 多环境；
- 语言条件；
- 跨数据集迁移；
- 可能的跨embodiment迁移。

但真实机器人数据远少于互联网图文数据，而且不同数据集之间存在：

- robot morphology不同；
- action dimension不同；
- joint order不同；
- camera命名不同；
- control frequency不同；
- absolute/delta不同；
- position/velocity不同；
- task annotation质量不同。

这就是VLA预训练比普通VLM预训练困难的重要原因。

---

## 8. Embodiment：模型控制的是哪一种身体

Embodiment可以理解为机器人的身体结构与控制接口。

不同机器人可能是：

~~~text
SO-101：若干关节位置 + gripper
Panda：7个手臂关节 + gripper
ALOHA：双臂
Mobile manipulator：底盘 + 机械臂
~~~

即使语言指令都是“拿起杯子”，对应action维度和物理含义也不同。

### 8.1 Padding不能自动解决语义问题

SmolVLA当前配置允许把较短state/action向量padding到固定最大维度，例如32维。

这样可以统一神经网络张量形状：

~~~text
SO-101 action 6维
→ padding
→ 32维模型输入

其他机器人 action 14维
→ padding
→ 32维模型输入
~~~

但padding只解决shape问题，不会自动告诉模型：

- 第3维属于哪个joint；
- 正方向是什么；
- 单位是什么；
- 该维度是位置还是速度。

仍然需要一致的数据规范、预处理和足够的跨embodiment训练。

---

## 9. SmolVLA总体结构

SmolVLA是Hugging Face发布的紧凑型开源VLA。官方发布版本通常称为SmolVLA-450M。

其核心结构是：

~~~text
Multiple Camera Images
→ Vision Encoder
→ Visual Tokens
                           ┐
Language Instruction      │
→ Tokenizer → Text Tokens ├→ VLM Context
                           │
Robot State → Linear Token┘
                           │
                           ▼
                    Action Expert
                 + noisy action chunk
                 + flow time
                           │
                           ▼
                  Continuous Actions
~~~

核心可以分为两大部分：

1. SmolVLM2提供视觉与语言理解；
2. Flow Matching Transformer Action Expert生成连续动作chunk。

---

## 10. SmolVLM2部分

官方架构说明中，SmolVLM2由：

~~~text
SigLIP Vision Encoder
+
SmolLM2 Language Decoder
~~~

组成。

### 10.1 图像输入

SmolVLA支持多个相机视角，例如：

- 第三人称相机；
- 腕部相机；
- 额外环境相机。

图像经过resize与padding，保持纵横比，然后转为SigLIP需要的像素范围。

官方模型设计将每帧压缩为较少的visual tokens，以降低注意力计算开销。官方介绍中每帧使用64个visual tokens。

### 10.2 语言输入

任务描述先经过tokenizer：

~~~text
“Pick up the blue cube”
→ token ids
→ language embeddings
~~~

当前配置中的最大语言长度是一个有限值；超长指令可能被截断或需要额外处理。因此任务描述应清楚、简洁、一致。

### 10.3 State Token

机器人state先padding到统一维度，再经过线性层：

$$
\mathbf e_s=\mathbf W_s\mathbf s+\mathbf b_s
$$

它把机器人状态投影到与VLM token兼容的hidden dimension。

官方描述中，sensorimotor state被压缩为一个state token。

### 10.4 Prefix

图像、语言和state共同形成条件上下文：

~~~text
Prefix =
[image tokens]
+
[language tokens]
+
[state token]
~~~

可以把Prefix理解为：

> 当前看到了什么、用户要求什么、机器人现在在哪里。

---

## 11. Action Expert

Action Expert负责回答：

> 在这个Prefix条件下，未来连续动作轨迹应该是什么？

SmolVLA的Action Expert是一个较小的Transformer，官方介绍约为100M参数，hidden width约为VLM的75%。

它接收：

- VLM Prefix；
- 当前带噪动作轨迹；
- Flow Matching时间 t。

输出：

- 动作空间中的velocity field。

### 11.1 为什么单独使用Action Expert

VLM擅长处理语义，但机器人控制还要求：

- 连续高精度数值；
- 多个关节同步输出；
- 时间连续性；
- 较低推理延迟；
- action chunk生成。

单独Action Expert让VLM负责“理解”，让动作网络负责“控制生成”。

### 11.2 Cross-Attention与Self-Attention

Action Expert需要两种关系。

Cross-Attention：

~~~text
Action tokens
→ attend to
Image + Language + State context
~~~

作用是让动作符合当前场景和指令。

Self-Attention：

~~~text
Action token之间互相注意
~~~

作用是建模动作序列内部的时间关系与平滑性。

SmolVLA使用交错的cross-attention与self-attention结构，在条件对齐和动作时间一致性之间取得平衡。

---

## 12. 为什么SmolVLA使用Flow Matching

普通回归可以直接预测动作，但容易受到多模态和平均动作问题影响。

离散动作token可以复用语言模型，但存在量化误差和自回归延迟。

Flow Matching保留连续动作，并从噪声逐步生成整段action chunk：

~~~text
Gaussian Noise
→ 沿学习到的vector field移动
→ Continuous Action Chunk
~~~

它与Diffusion Policy有相似直觉：

- 都从噪声生成动作；
- 都学习条件动作分布；
- 都需要多步迭代；
- 都能一次生成action chunk。

但数学训练目标和采样形式不同：

~~~text
Diffusion：
学习不同噪声等级下的去噪或score
→ scheduler执行reverse diffusion

Flow Matching：
直接学习连接noise与data的连续velocity field
→ ODE积分生成动作
~~~

---

## 13. Flow Matching的路径

令：

- A：干净专家动作chunk；
- ε：与A同形状的Gaussian noise；
- t：Flow Matching时间，范围在0到1附近。

当前SmolVLA源码使用：

$$
\mathbf x_t=t\boldsymbol\epsilon+(1-t)\mathbf A
$$

因此：

### 当 t=0

$$
\mathbf x_0=\mathbf A
$$

位于干净动作端。

### 当 t=1

$$
\mathbf x_1=\boldsymbol\epsilon
$$

位于噪声端。

这是一条从动作到噪声的直线路径。

对t求导：

$$
\mathbf u_t
=
\frac{d\mathbf x_t}{dt}
=
\boldsymbol\epsilon-\mathbf A
$$

网络学习：

$$
\mathbf v_\theta(
\mathbf x_t,t,\mathbf I,\mathbf L,\mathbf s
)
\approx
\mathbf u_t
$$

Loss为：

$$
\mathcal L
=
\left\|
\mathbf v_\theta-\mathbf u_t
\right\|_2^2
$$

---

## 14. 为什么target是 noise − action，却能从noise生成action

这一点最容易看反。

训练路径定义为：

~~~text
t=0：action
t=1：noise
~~~

所以正方向velocity是：

$$
\epsilon-A
$$

推理却从noise端开始：

~~~text
x_1 = noise
~~~

然后把时间从1积分回0。

Euler更新可以写成：

$$
\mathbf x_{t+\Delta t}
=
\mathbf x_t
+
\Delta t\,
\mathbf v_\theta(\mathbf x_t,t,\text{condition})
$$

推理时：

$$
\Delta t<0
$$

所以虽然网络预测的是朝向noise的正向velocity，但乘上负时间步后，状态会反向移动：

~~~text
noise
→ action
~~~

可以记为：

> 网络学习一张“动作到噪声”的速度地图，推理时沿地图反方向走回动作。

不同论文可能采用相反的t定义，因此读源码时必须看x_t和target公式，不能只凭记忆判断方向。

---

## 15. SmolVLA训练流程

训练batch包含：

~~~text
images
language tokens
language attention mask
robot state
expert action chunk
action_is_pad
~~~

主流程：

~~~text
1. 图像 → visual embeddings
2. 指令 → language embeddings
3. state → state token
4. 合成Prefix condition
5. expert actions → pad到统一action dim
6. 采样Gaussian noise ε
7. 采样flow time t
8. 构造 x_t = tε + (1-t)A
9. 构造target velocity u_t = ε - A
10. Action Expert预测 v_t
11. MSE(v_t, u_t)
12. mask episode padding
13. backward + optimizer
~~~

### 15.1 张量例子

假设：

~~~text
batch size = 16
chunk_size = 50
max_action_dim = 32
SO-101真实action_dim = 6
~~~

内部动作形状可能是：

~~~text
prepared_actions: (16, 50, 32)
noise:            (16, 50, 32)
x_t:              (16, 50, 32)
target_velocity:  (16, 50, 32)
pred_velocity:    (16, 50, 32)
~~~

其中前6维对应真实SO-101 action，其余维度用于统一模型形状的padding。

Loss最终只应针对真实action维度与episode有效时间位置正确计算。

### 15.2 Flow time的形状

每个batch样本可以独立采样一个t：

~~~text
time: (16,)
~~~

广播为：

~~~text
time_expanded: (16, 1, 1)
~~~

然后同时作用于该样本的50个动作位置和32个动作维度。

---

## 16. SmolVLA推理流程

推理时没有expert action。

首先建立Prefix：

~~~text
images + language + current state
→ VLM contextual features
~~~

然后创建随机动作噪声：

~~~text
noise: (B, chunk_size, max_action_dim)
~~~

从：

$$
\mathbf x_1=\boldsymbol\epsilon
$$

开始，通过数值积分逐步走到t=0：

~~~text
x_1
→ x_0.9
→ x_0.8
→ ...
→ x_0
~~~

当前LeRobot默认配置使用有限次Euler integration；默认num_steps为10。

最终：

~~~text
(B, 50, 32)
→ 取真实action前D维
→ (B, 50, D)
→ action queue
→ 每个控制周期弹出一个action
~~~

注意：

> num_steps=10表示Flow Matching积分10次，不代表机器人执行10个动作。

机器人动作数量由chunk_size和n_action_steps决定。

---

## 17. Prefix KV Cache

推理的多个Flow Matching积分步中：

- image不变；
- language不变；
- current state condition不变；
- noisy action x_t不断变化。

因此可以先计算一次Prefix的Key/Value cache：

~~~text
Images + Language + State
→ VLM Prefix KV cache
~~~

后续每次积分只更新Action suffix，并让它查询缓存的Prefix。

这样避免在10个Flow步骤中重复完整计算视觉和语言部分。

概念上：

~~~text
第一次：
encode Prefix + build cache

每个Flow step：
new noisy action suffix
→ attend to cached Prefix
→ predict velocity
~~~

这是一项推理效率优化，不改变Policy所建模的条件分布。

---

## 18. 当前LeRobot配置如何阅读

截至本讲义生成时，LeRobot main分支SmolVLA默认配置包含：

~~~text
n_obs_steps = 1
chunk_size = 50
n_action_steps = 50

max_state_dim = 32
max_action_dim = 32

tokenizer_max_length = 48
num_steps = 10

freeze_vision_encoder = True
train_expert_only = True
train_state_proj = True

num_vlm_layers = 16
self_attn_every_n_layers = 2
expert_width_multiplier = 0.75
~~~

默认normalization包括：

~~~text
VISUAL：IDENTITY
STATE：MEAN_STD
ACTION：MEAN_STD
~~~

这些是当前源码默认值，不是所有版本、所有checkpoint和SO-101任务的固定最优值。阅读实际实验时应检查保存的config。

### 18.1 chunk_size与n_action_steps

- chunk_size：模型一次生成多少动作位置；
- n_action_steps：放入执行queue的动作数；
- n_action_steps不能大于chunk_size。

两者都为50时，一次生成的50步全部进入queue。

如果减小n_action_steps，则可以更早基于新observation重新规划。

---

## 19. SmolVLA的同步Action Queue

普通同步执行逻辑与ACT很像：

~~~text
action queue为空
→ 读取最新image、language、state
→ SmolVLA生成action chunk
→ 将n_action_steps放入queue
→ 每次select_action弹出一个action
→ queue为空后再次推理
~~~

Policy层：

- chunk内部没有使用新observation重新生成；
- queue耗尽后才重新规划；
- n_action_steps越大，Policy反馈越慢。

Motor层仍可以保持自己的高频位置闭环。

所以VLA并不天然意味着每个动作都进行语言和视觉重新推理。

---

## 20. 异步推理与Real-Time Chunking

SmolVLA论文和官方介绍强调异步推理：

~~~text
Robot：
持续执行当前chunk

Policy Server：
并行处理新observation并生成下一个chunk
~~~

### 同步方式

~~~text
queue执行完
→ 等待推理
→ 得到新chunk
→ 继续执行
~~~

若推理耗时明显，会产生停顿。

### 异步方式

~~~text
当前queue尚未耗尽
→ 提前发送最新observation
→ 后台生成新chunk
→ 新chunk到达时与剩余动作衔接
~~~

它的优势是减少等待，并让Policy更早利用新observation。

但它增加了新的问题：

- 推理完成时机器人已经移动；
- 新chunk基于稍旧的observation；
- 新旧chunk可能重叠；
- 必须处理inference delay；
- chunk拼接不当会跳变。

当前LeRobot源码还提供RTC相关配置，用于处理实时chunk生成与已有剩余动作之间的关系。理论上应把它理解为执行系统策略，而不是VLA模型本体突然学会了新的任务。

---

## 21. SmolVLA源码训练主链

源码阅读入口：

~~~text
SmolVLAPolicy.forward(batch)
→ prepare_images(batch)
→ prepare_state(batch)
→ language tokens / masks
→ prepare_action(batch)
→ VLAFlowMatching.forward(...)
~~~

Flow Matching核心：

~~~python
noise = sample_noise(actions.shape)
time = sample_time(batch_size)

x_t = time * noise + (1 - time) * actions
u_t = noise - actions

prefix = embed_prefix(images, language, state)
suffix = embed_suffix(x_t, time)

suffix_output = vlm_with_expert(prefix, suffix)
v_t = action_out_proj(suffix_output)

loss = mse(v_t, u_t)
~~~

随后外层：

~~~text
裁掉非真实action维度
→ action_is_pad mask
→ 对有效位置求平均
→ backward
~~~

这条链仍然符合熟悉的深度学习训练流程：

~~~text
Dataset
→ DataLoader
→ Model forward
→ Flow Matching loss
→ backward
→ optimizer
→ checkpoint
~~~

---

## 22. SmolVLA源码推理主链

~~~text
SmolVLAPolicy.select_action(batch)

如果queue为空：
    → prepare images/state/language
    → model.sample_actions(...)
    → encode Prefix and KV cache
    → sample Gaussian action noise
    → Euler integrate for num_steps
    → unpad to real action dimension
    → put n_action_steps into queue

→ popleft one action
~~~

内部形状假设：

~~~text
noise:
(B, chunk_size, max_action_dim)

generated padded actions:
(B, chunk_size, max_action_dim)

unpad:
(B, chunk_size, real_action_dim)

queue internal orientation:
(n_action_steps, B, real_action_dim)

select_action output:
(B, real_action_dim)
~~~

---

## 23. Language Annotation为什么重要

如果训练数据里的文字是：

~~~text
episode 1：“pick”
episode 2：“task”
episode 3：“move object”
~~~

模型很难学习清晰的语言—行为对应关系。

好的任务描述应该：

- 使用明确动作动词；
- 指出目标物体；
- 指出目标位置；
- 同一任务表达尽量一致；
- 不描述轨迹中没有发生的行为；
- 避免无意义标签。

例如：

~~~text
差：“task 1”

较好：“Pick up the blue cube”

更完整：“Pick up the blue cube and place it in the bin”
~~~

但语言越详细也不一定越好。如果数据无法区分“轻轻放下”和“快速扔下”，文字中添加这种区别不会自动创造对应动作监督。

---

## 24. VLA到底能泛化什么

可以将泛化分成不同层级。

### 24.1 Visual Generalization

- 光线变化；
- 背景变化；
- 物体外观变化；
- 相机轻微变化。

### 24.2 Language Generalization

- 同义表达；
- 不同任务描述；
- 属性和空间关系。

### 24.3 Task Generalization

- 训练过多种抓取与放置后，适应新的组合任务。

### 24.4 Embodiment Generalization

- 将多机器人数据中的知识迁移到新机器人。

这些能力难度依次提高，而且并非有VLA名字就自动拥有。

特别是embodiment generalization还受到：

- action interface；
- robot geometry；
- camera；
- control frequency；
- dataset balance；
- normalization；
- fine-tuning数据量。

VLM的语义泛化通常强于VLA的精确控制泛化。

---

## 25. SmolVLA与ACT对比

| 维度 | ACT | SmolVLA |
|---|---|---|
| 核心目标 | 学习视觉/状态到action chunk | 语言条件的通用视觉机器人策略 |
| 语言输入 | 通常没有 | 有 |
| 视觉模型 | CNN backbone | 预训练SmolVLM2视觉语言模型 |
| 动作生成 | Transformer + CVAE | Flow Matching Action Expert |
| 训练Loss | L1 action + KL | Flow velocity MSE |
| 动作形式 | 连续chunk | 连续chunk |
| 推理 | 通常一次主forward | 多次Flow ODE积分 |
| 预训练 | 常从任务数据训练 | VLM预训练 + 机器人VLA预训练 |
| 模型规模 | 相对小 | 更大但属于紧凑型VLA |
| 多任务能力 | 主要取决于任务数据 | 显式语言条件，适合多任务 |
| 数据要求 | 单任务可较小 | 微调仍需高质量任务数据 |
| Queue控制 | 支持 | 支持，也可结合异步/RTC |

二者共同点：

- 都是imitation learning；
- 都依赖demonstration；
- 都输出未来action chunk；
- 都会遇到distribution shift；
- 都不能自动保证安全、可达和无碰撞；
- 都需要正确的action normalization与机器人接口。

---

## 26. SmolVLA与Diffusion Policy对比

| 维度 | Diffusion Policy | SmolVLA |
|---|---|---|
| 条件 | 视觉、state | 视觉、语言、state |
| 生成对象 | 连续action trajectory | 连续action chunk |
| 起点 | Gaussian noise | Gaussian noise |
| 学习目标 | noise/sample/score相关目标 | velocity field |
| 推理 | reverse diffusion scheduler | ODE数值积分 |
| 语义骨干 | 通常较小视觉编码器 | 预训练VLM |
| 多任务语言控制 | 不一定 | 核心设计之一 |

两者外观相似，但不能把Flow Matching简单称为“完全相同的Diffusion”。

---

## 27. VLA仍然不等于规划器或安全系统

即使VLA理解了指令，也可能：

- 抓错物体；
- 生成不可达动作；
- 撞到障碍物；
- 超过关节范围；
- 在遮挡后继续旧计划；
- 因语言歧义选择错误任务；
- 在新相机视角下失败。

因此实际系统仍需：

~~~text
VLA Policy
+
action normalization
+
joint/workspace limits
+
collision and safety checks
+
bottom-level control
+
runtime monitoring
~~~

VLA的语言理解不能替代：

- FK/IK；
- 控制理论；
- 碰撞检测；
- 机械安全；
- 数据质量；
- 真机评估。

---

## 28. 常见失败现象

### 28.1 换一句话就不会做

可能原因：

- 微调数据只有一种固定描述；
- 预训练语言能力在微调中被破坏；
- task annotation质量差；
- 模型实际上忽略language，只记住视觉轨迹。

### 28.2 指令改变但动作不改变

说明模型可能没有真正建立语言—动作条件关系。若数据中每种画面只对应一个任务，模型即使完全忽略语言也能降低loss。

要验证语言是否有效，应构造：

~~~text
相同或相似场景
+
不同指令
→
不同正确动作
~~~

### 28.3 识别对了物体但抓取不准

这说明语义层可能正确，但连续控制层失败。检查：

- action expert；
- robot state/action normalization；
- camera calibration；
- demonstrations；
- control frequency；
- bottom-level tracking。

### 28.4 动作语义正确但很抖

可能来自：

- action chunk边界；
- Flow积分步数；
- 随机采样；
- action normalization；
- demonstration不平滑；
- control layer和机械间隙。

### 28.5 预训练模型微调后反而变差

可能原因：

- learning rate过高；
- 数据太少或偏差太大；
- 输入/输出语义不匹配；
- 训练了不该大幅更新的backbone；
- task description不一致；
- checkpoint、processor和dataset stats不匹配。

---

## 29. 常见误区

1. **VLA就是VLM后面接一个Linear层。**  
   错。实际还要解决state输入、动作表示、时序生成、控制频率和机器人适配。

2. **有语言输入就一定理解语言。**  
   错。模型可能在训练中学会忽略语言。

3. **VLM认识杯子，所以自然会控制SO-101抓杯子。**  
   错。语义知识不等于motor control。

4. **所有VLA都把action变成文字token。**  
   错。SmolVLA使用连续Flow Matching Action Expert。

5. **Flow Matching的10个积分步就是10个机器人动作。**  
   错。积分步是生成一次action chunk的内部计算。

6. **Padding到32维就自动支持任意机器人。**  
   错。Padding只统一shape，不统一物理语义。

7. **SmolVLA有VLM，所以不需要robot state。**  
   错。当前身体状态决定下一步应该如何运动。

8. **预训练后只用极少数据一定成功。**  
   错。仍取决于任务变化、相机、动作接口和数据质量。

9. **VLA是基础模型，所以不需要fine-tuning。**  
   错。官方文档也强调base model应针对具体setup微调。

10. **VLA会自动满足安全和运动学约束。**  
    错。仍需控制与安全层。

---

## 30. 本阶段知识图

~~~text
Camera 1 ─┐
Camera 2 ─┼→ SigLIP Vision Encoder → Visual Tokens ─┐
Camera N ─┘                                          │
                                                    │
Instruction → Tokenizer → Language Tokens ──────────┼→ VLM Prefix
                                                    │
Robot State → Padding → Linear → State Token ───────┘
                                                    │
                                                    ▼
Gaussian Action Noise ────────────────► Action Expert
Flow Time t ──────────────────────────► Cross/Self Attention
VLM Prefix ───────────────────────────► Velocity Field
                                                    │
                                                    ▼
                                      Reverse ODE Integration
                                                    │
                                                    ▼
                                      Continuous Action Chunk
                                                    │
                                           n_action_steps
                                                    ▼
                                               Action Queue
                                                    │
                                                    ▼
                                                  Robot
~~~

---

## 31. 阶段 8 自测

建议先回答问题1～3，再继续Flow Matching和源码题。

### 问题 1：VLM与VLA

请解释：

1. VLM的典型输入输出是什么？
2. VLA相对VLM多解决了什么问题？
3. 为什么VLA除了图像和语言，通常还需要robot state？
4. VLM认识“蓝色杯子”，为什么不等于它已经会控制SO-101抓取？

### 问题 2：语言条件

同一张图像中有红色方块和蓝色方块，训练数据只有一种任务：“抓起蓝色方块”。

1. 即使把语言输入模型，模型为什么仍可能学会忽略语言？
2. 要验证模型是否真正使用语言，数据集应该增加什么类型的样本？
3. 推理时输入“抓起红色方块”，但训练轨迹从未抓过红色方块，能否保证成功？为什么？

### 问题 3：VLA动作生成方式

请比较：

1. 离散action token如何产生动作？
2. 连续Action Expert如何产生动作？
3. SmolVLA属于哪一种？
4. 为什么SmolVLA不是普通的逐token自回归动作模型？

### 问题 4：SmolVLA结构

请补全：

~~~text
Images
→ __________
→ visual tokens

Language
→ __________
→ language tokens

Robot state
→ padding + __________
→ state token

三者组成 __________ condition
→ Action Expert
→ continuous action chunk
~~~

### 问题 5：Flow Matching训练

已知：

$$
x_t=t\epsilon+(1-t)A
$$

$$
u_t=\epsilon-A
$$

回答：

1. t=0时，x_t是什么？
2. t=1时，x_t是什么？
3. 网络输入有哪些？
4. 网络预测什么？
5. Loss是什么？

### 问题 6：Flow Matching推理方向

训练路径定义为action到noise，但推理从noise开始。

1. 推理起点是什么？
2. 时间从1走向0还是从0走向1？
3. 为什么target velocity是noise-action，仍然可以反向生成action？
4. num_steps=10表示生成多少个机器人动作吗？

### 问题 7：张量形状

设：

~~~text
B = 16
chunk_size = 50
max_action_dim = 32
real_action_dim = 6
~~~

请写出：

1. prepared action；
2. noise；
3. x_t；
4. predicted velocity；
5. unpad后的generated action chunk；
6. select_action最终单步输出。

### 问题 8：Chunk与闭环

配置：

~~~text
chunk_size = 50
n_action_steps = 20
control frequency = 30 Hz
~~~

1. 模型一次生成多少步？
2. queue实际采用多少步？
3. 同步模式约多久重新读取observation并生成chunk？
4. 理想Policy生成频率约是多少？
5. 为什么n_action_steps越大不一定越好？

### 问题 9：Pretraining与Fine-tuning

请解释：

1. VLM预训练提供什么知识？
2. VLA机器人预训练增加什么能力？
3. 你的SO-101任务微调又在适配什么？
4. 为什么预训练不能自动修复joint order或action normalization错误？

### 问题 10：综合判断

判断正误并解释：

1. 所有VLA都使用离散action token。
2. SmolVLA使用预训练VLM处理图像和语言，再由Action Expert生成连续动作。
3. action padding到32维后，所有机器人动作语义自然一致。
4. Flow Matching时间t就是机器人rollout时间。
5. num_steps控制Flow ODE积分次数，chunk_size控制生成动作位置数量。
6. 相同场景永远只有一种任务时，模型可能忽略language仍获得低loss。
7. VLA具有语言理解能力，因此不再需要关节限位和安全裁剪。
8. ACT和SmolVLA都能生成action chunk，但架构、loss与预训练方式不同。

---

## 32. 本阶段完成标准

如果你能不看讲义解释下面这段话，就完成了阶段8：

> VLA在视觉Policy中加入语言任务条件，并通常利用预训练VLM获得视觉—语言语义能力，但精确机器人控制仍需机器人轨迹数据。SmolVLA由SmolVLM2和连续Action Expert组成：图像、语言与robot state形成Prefix，Action Expert通过Flow Matching从Gaussian noise生成action chunk。训练时在action和noise之间采样x_t并预测velocity；推理从noise端反向积分到action端。Flow积分步、action chunk长度和真实控制时间是三个不同概念。预训练能够提供迁移能力，但不能替代正确的数据语义、fine-tuning、控制与安全系统。

---

## 33. 参考资料

- [SmolVLA论文](https://arxiv.org/abs/2506.01844)
- [Hugging Face SmolVLA官方介绍](https://huggingface.co/blog/smolvla)
- [LeRobot SmolVLA文档](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot SmolVLA模型源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [LeRobot SmolVLA配置源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py)
- [RT-2论文](https://arxiv.org/abs/2307.15818)
- [OpenVLA论文](https://arxiv.org/abs/2406.09246)
- [π0论文](https://www.physicalintelligence.company/download/pi0.pdf)
