# 阶段 2：为 ACT 服务的 Transformer

## 0. 本阶段定位

这一阶段不追求完整复现语言模型，而是建立足够精确的 Transformer 心智模型，使你能够读懂 ACT 中的：

- token 与 embedding。
- Query、Key、Value。
- scaled dot-product attention。
- self-attention 与 cross-attention。
- multi-head attention。
- positional embedding。
- Transformer encoder 与 decoder。
- image、robot state、latent 和 action query 的数据流。
- 主要张量形状。

完成本阶段后，你应该能沿着 ACT 源码回答：

```text
哪些数据被变成了 token？
encoder 在融合什么？
decoder 的 query 是什么？
cross-attention 从哪里读取信息？
为什么最后会输出一段动作？
```

CVAE 的概率含义、ACT loss 和 temporal ensembling 将留到阶段 3。

---

## 1. 为什么机器人策略需要 Transformer

SO-101 的策略需要同时处理多种信息：

```text
第三人称相机
+
第一人称相机
+
机器人当前关节状态
+
动作序列中的不同时间位置
```

模型需要建立的关系包括：

- 图像中物体在哪里。
- 夹爪相对于物体在哪里。
- 当前关节状态是否允许直接接近。
- 未来第 1 个动作和第 20 个动作如何保持连续。
- 两个相机中的视觉信息如何对应。

CNN 擅长提取局部视觉特征，但单独的 CNN 并不直接说明：

```text
当前关节状态应该关注哪块视觉区域？
未来不同动作位置应该如何共享场景信息？
```

Transformer 的作用可以概括为：

> 将不同来源的信息表示成 token，再通过 attention 学习 token 之间的关系。

---

## 2. Token：Transformer 处理的基本单位

在语言模型中，一个 token 可以对应一个词或子词。

在 ACT 中，token 可以对应：

- 图像特征图中的一个空间位置。
- 整个机器人关节状态。
- latent style 信息。
- 未来动作序列中的一个时间位置。

重要的是：

> Token 不是必须代表文字。任何信息只要被转换成固定维度向量，都可以作为 token。

假设 Transformer 的隐藏维度为 $D=512$，那么每个 token 都必须最终表示为长度 512 的向量：

```text
image token       → 512 维
robot-state token → 512 维
latent token      → 512 维
action query      → 512 维
```

---

## 3. Embedding 与 Projection

不同模态原始维度不同：

```text
机器人状态：Dₛ 维
动作：Dₐ 维
CNN 图像特征：C 维
latent：D_z 维
```

Transformer 无法直接把不同维度的向量放入同一个序列，因此先通过可学习投影统一到 `dim_model = D`。

例如机器人状态：

$$
x_{state}=W_{state}s_t+b_{state}
$$

如果：

```text
sₜ shape = (B, 6)
W_state 将 6 维映射到 512 维
```

那么输出：

```text
state token shape = (B, 512)
```

LeRobot ACT 中对应的模块是：

```python
self.encoder_robot_state_input_proj = nn.Linear(
    state_dim,
    dim_model,
)
```

图像特征则使用 `1×1 convolution` 将 CNN channel 投影到 `dim_model`。

可以把 projection 理解为“翻译器”：

```text
各模态自己的数值语言
        ↓
统一翻译成 Transformer 的 D 维语言
```

---

## 4. 图像怎样变成视觉 Token

输入相机图像通常形如：

```text
(B, 3, H, W)
```

经过 CNN backbone 后，得到更小的 feature map：

```text
(B, C, H_f, W_f)
```

再通过 `1×1 convolution` 投影到 Transformer 维度：

```text
(B, D, H_f, W_f)
```

然后将空间位置展开：

```text
(B, D, H_f, W_f)
        ↓ flatten spatial dimensions
(B, H_f × W_f, D)
```

feature map 上每个空间格子变成一个 token。

例如：

```text
H_f = 15
W_f = 20
```

一台相机会产生：

$$
15\times20=300\text{ visual tokens}
$$

两台相机则产生 600 个 visual tokens。

当前 LeRobot ACT 源码使用：

```python
cam_features = einops.rearrange(
    cam_features,
    "b c h w -> (h w) b c",
)
```

所以源码实际排列是：

```text
(visual sequence, batch, dim_model)
```

而很多教程使用：

```text
(batch, visual sequence, dim_model)
```

两种表示含义相同，只是维度顺序不同。

---

## 5. 为什么需要 Positional Encoding

只看 token 内容时，attention 本身不知道 token 的顺序和位置。

例如下面两个序列包含相同 token：

```text
[接近, 闭合, 抬起]
[抬起, 闭合, 接近]
```

如果没有位置信息，Transformer 很难区分它们的动作顺序。

图像也有同样的问题：

```text
物体在图像左上角
物体在图像右下角
```

即使局部视觉内容相似，空间位置的意义不同。

因此输入通常是：

$$
x_i^{input}=x_i^{content}+p_i
$$

其中：

- $x_i^{content}$：token 本身包含什么。
- $p_i$：token 位于哪里。

ACT 中主要有三类位置表示：

### 图像二维位置编码

告诉模型 visual token 来自 feature map 的哪个行列位置。

### Encoder 的一维 token 位置

区分 latent token、robot-state token、environment-state token。

### Decoder 的动作位置 embedding

区分：

```text
query 0   → 未来第 0 个动作
query 1   → 未来第 1 个动作
...
query H-1 → 未来第 H-1 个动作
```

当前 LeRobot ACT 使用长度为 `chunk_size` 的可学习 decoder position embedding：

```python
self.decoder_pos_embed = nn.Embedding(
    chunk_size,
    dim_model,
)
```

---

## 6. Attention 的直觉

Attention 要解决的问题是：

> 对当前这个 token，其他哪些 token 最值得关注？

在抓取任务中，可以想象某个未来夹爪动作 query 正在提问：

```text
为了决定这个时间位置的动作，
我应该关注哪块图像、哪个相机和什么关节状态？
```

Attention 使用三个角色：

- Query：我正在寻找什么。
- Key：我可以用什么特征被匹配。
- Value：如果匹配成功，我实际读取什么信息。

一个直观类比：

```text
Query：搜索关键词
Key：每份资料的索引标签
Value：资料的真正内容
```

Query 与 Key 决定相关程度，相关程度再决定从各个 Value 中读取多少信息。

---

## 7. Q、K、V 从哪里来

给定 token 矩阵 $X$，通过三个不同的线性投影得到：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

注意：

> 在 self-attention 中，Q、K、V 来源于同一组输入 token，但经过不同权重投影后，它们通常不是相同矩阵。

假设常规 batch-first 表示为：

```text
X shape = (B, S, D)
```

其中：

- $B$：batch size。
- $S$：token 数量。
- $D$：模型隐藏维度。

投影后：

```text
Q shape = (B, S, dₖ)
K shape = (B, S, dₖ)
V shape = (B, S, dᵥ)
```

---

## 8. Scaled Dot-Product Attention

Attention 的核心公式是：

$$
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

分四步理解。

### 第一步：Query 与 Key 相似度

$$
scores=QK^T
$$

如果有 $S_q$ 个 query 和 $S_k$ 个 key：

```text
Q shape = (S_q, dₖ)
K shape = (S_k, dₖ)
scores shape = (S_q, S_k)
```

每个 score 表示一个 query 对一个 key 的匹配程度。

### 第二步：缩放

$$
scores=\frac{QK^T}{\sqrt{d_k}}
$$

维度较大时，点积数值可能变大，使 softmax 过度饱和。除以 $\sqrt{d_k}$ 可以让训练更稳定。

### 第三步：Softmax

$$
A=softmax(scores)
$$

对每个 query 来说，它对所有 key 的权重和为 1。

### 第四步：对 Value 加权求和

$$
output=AV
$$

最终，每个 query 获得一个综合了相关 Value 的新表示。

---

## 9. 一个机器人 Attention 例子

假设 encoder 中有这些 token：

```text
token 0：robot state
token 1：第三人称图像中的物体区域
token 2：第三人称图像中的背景区域
token 3：腕部相机中的夹爪区域
token 4：腕部相机中的物体区域
```

未来“闭合夹爪”动作位置的 query 可能产生注意力权重：

```text
robot state             0.20
第三人称物体区域         0.15
第三人称背景区域         0.02
腕部相机夹爪区域         0.28
腕部相机物体区域         0.35
```

输出表示就是这些 Value 的加权组合。

这里的数字只是直觉示例，不代表模型一定会学习出可直接解释的人类语义。

Attention weight 较高也不自动证明该区域是模型决策的唯一原因。

---

## 10. Self-Attention

Self-attention 的 Q、K、V 都来自同一个 token 序列：

```text
Q = projection(X)
K = projection(X)
V = projection(X)
```

因此序列中的 token 可以互相交换信息。

### ACT Encoder 中的 self-attention

Encoder token 可能包括：

```text
latent token
robot-state token
camera 1 visual tokens
camera 2 visual tokens
```

Self-attention 可以学习：

- robot state 与视觉区域之间的关系。
- 两个相机之间的对应关系。
- 图像不同空间位置之间的关系。
- latent 信息如何影响场景表示。

### ACT Decoder 中的 self-attention

Decoder 中每个 token 对应未来动作块的一个时间位置。

它们之间进行 self-attention，可以交流：

```text
第 0 个未来动作
↔ 第 1 个未来动作
↔ 第 2 个未来动作
↔ ...
```

这有助于让整段动作具有时间关联。

---

## 11. Cross-Attention

Cross-attention 的 Q 与 K/V 来自不同序列。

ACT decoder 中：

```text
Query：decoder action queries
Key：encoder output
Value：encoder output
```

公式仍然是：

$$
softmax\left(\frac{Q_{action}K_{context}^T}{\sqrt{d_k}}\right)V_{context}
$$

含义是：

> 每个未来动作位置主动从编码后的视觉、机器人状态和 latent 中读取自己需要的信息。

如果：

```text
动作 query 数量 = H
encoder token 数量 = S
```

那么 cross-attention score 的形状是：

```text
(H, S)
```

每一行表示一个未来动作位置对所有 encoder context token 的关注程度。

---

## 12. Self-Attention 与 Cross-Attention 对比

| 类型 | Query 来源 | Key/Value 来源 | ACT 中的作用 |
|---|---|---|---|
| Encoder self-attention | encoder tokens | encoder tokens | 融合视觉、状态和 latent |
| Decoder self-attention | action queries | action queries | 建立动作时间位置之间的联系 |
| Decoder cross-attention | action queries | encoder output | 让动作位置读取场景信息 |

记忆方法：

```text
Self-attention：同一群 token 内部交流
Cross-attention：一群 token 向另一群 token 查询信息
```

---

## 13. Multi-Head Attention

单头 attention 只有一套 Q/K/V 投影。

Multi-head attention 将模型维度分成多个 head：

```text
dim_model = 512
n_heads = 8
head_dim = 512 / 8 = 64
```

每个 head 有自己的投影参数，可以学习不同的关系。

直觉上，不同 head 可能分别关注：

- 夹爪与物体的空间关系。
- 机器人当前关节姿态。
- 两个相机的对应区域。
- 动作块的短期连续性。
- 动作块中较远时间位置的关系。

各 head 的输出最后进行拼接并再次线性投影：

$$
MultiHead(Q,K,V)
=Concat(head_1,\dots,head_h)W_O
$$

不要把“一个 head 对应一种人类可命名功能”当作硬规则。这只是帮助理解的直觉，训练并不会为 head 指定固定语义。

---

## 14. Attention Mask 与 Padding Mask

Attention 中常见两类 mask。

### Padding Mask

用于忽略 padding token，例如 ACT 的 VAE encoder 处理 episode 尾部动作窗口时：

```text
action_is_pad = [False, False, False, True, True]
```

被标记为 padding 的 action token 不应提供正常时序信息。

### Causal Mask

语言模型生成下一个词时，不能偷看未来词，因此使用 causal mask：

```text
位置 0 只能看位置 0
位置 1 可以看位置 0～1
位置 2 可以看位置 0～2
```

但当前 ACT 的主 decoder 不是逐词式自回归生成。它使用一组动作位置 query 并行预测整个 action chunk，因此不要默认它和 GPT 一样使用 causal mask。

当前 LeRobot ACT decoder self-attention 调用中没有传入 causal attention mask。

---

## 15. Transformer Encoder Layer

一个典型 encoder layer 包含：

```text
输入 tokens
    ↓
Multi-Head Self-Attention
    ↓
Residual Connection + LayerNorm
    ↓
Feed-Forward Network
    ↓
Residual Connection + LayerNorm
```

### Residual Connection

如果某个子层为 $F(x)$：

$$
y=x+F(x)
$$

它让原始信息可以绕过子层，并帮助深层网络传播梯度。

### LayerNorm

对每个 token 的 feature 进行归一化，改善训练稳定性。

### Feed-Forward Network

对每个 token 独立使用同一组 MLP：

$$
FFN(x)=W_2\sigma(W_1x+b_1)+b_2
$$

Attention 负责 token 之间的信息交换；FFN 负责对每个 token 的特征进行非线性变换。

---

## 16. Transformer Decoder Layer

ACT 的一个 decoder layer 可以概括为：

```text
action queries
     ↓
decoder self-attention
     ↓
Residual + Norm
     ↓
cross-attention with encoder output
     ↓
Residual + Norm
     ↓
FFN
     ↓
Residual + Norm
```

这里最关键的是 cross-attention：

```text
动作位置提出问题
        ↓
从视觉、状态和 latent context 中读取信息
        ↓
形成该动作位置的隐藏表示
```

---

## 17. Encoder 与 Decoder 的职责

### Encoder

回答：

> 当前场景中有哪些信息，它们之间有什么关系？

输入示意：

```text
[latent]
[robot state]
[camera 1 token 1 ... token N]
[camera 2 token 1 ... token N]
```

输出仍是一组 context tokens，但每个 token 已经融合了其他 token 的信息。

### Decoder

回答：

> 对未来每个动作时间位置，应该从场景 context 中读取什么，并输出什么动作？

输入是一组动作位置 query，输出是一组动作位置表示。

最后通过 linear action head：

```text
decoder output
(B, chunk_size, dim_model)
        ↓ Linear(dim_model → action_dim)
predicted actions
(B, chunk_size, action_dim)
```

---

## 18. ACT 的 Main Transformer 数据流

暂时忽略 CVAE 如何得到 latent，只看 main Transformer：

```text
双相机图像
    ↓ CNN backbone
视觉 feature maps
    ↓ projection + flatten
visual tokens

robot state
    ↓ Linear
state token

latent
    ↓ Linear
latent token

[latent + state + visual tokens]
    ↓
Transformer Encoder
    ↓
context tokens

chunk_size 个 action position queries
    ↓ decoder self-attention
    ↓ cross-attention context tokens
Transformer Decoder
    ↓
chunk_size 个 action representations
    ↓ Linear action head
未来 action chunk
```

当前 LeRobot ACT 源码明确将 encoder tokens 组织为：

```text
[latent, (robot_state), (environment_state), image_feature_map_pixels]
```

并使用类似 DETR object query 的 learned decoder positional embeddings，为每个动作时间位置生成一个输出。

参考：[LeRobot ACT 当前实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)

---

## 19. 为什么提到 DETR

DETR 使用一组 learned object queries：

```text
object query 0 → 尝试输出一个物体
object query 1 → 尝试输出另一个物体
...
```

这些 query 通过 cross-attention 从图像 context 中读取信息。

ACT 借用了类似思想，但 query 的含义不同：

```text
DETR query 0 → 一个候选物体
ACT query 0  → 未来第 0 个动作位置

DETR query 1 → 另一个候选物体
ACT query 1  → 未来第 1 个动作位置
```

因此 ACT 可以并行产生固定长度的动作序列。

参考：[DETR 原论文](https://arxiv.org/abs/2005.12872)

---

## 20. ACT Decoder 不是语言模型式逐步生成

语言 decoder 常见流程：

```text
生成 token 1
→ 把 token 1 作为输入
→ 生成 token 2
→ 再生成 token 3
```

ACT 主 decoder 的核心方式是：

```text
一次准备 chunk_size 个 action queries
→ 并行经过 decoder
→ 一次输出 chunk_size 个动作表示
```

当前实现中，decoder content input 初始化为零：

```python
decoder_in = torch.zeros(
    chunk_size,
    batch_size,
    dim_model,
)
```

不同动作位置主要由 learned `decoder_pos_embed` 区分。

所以不要把 ACT action query 理解成“上一步真实 action”。它更像是：

> 一个可学习的输出槽位，代表未来动作块中的某个时间位置。

---

## 21. 一个完整张量形状例子

假设：

- batch size：$B=16$。
- `dim_model`：$D=512$。
- 两台相机。
- 每台相机 feature map：$15\times20$。
- 有一个 latent token。
- 有一个 robot-state token。
- `chunk_size=100`。
- `action_dim=6`。

### Encoder token 数

每台相机：

$$
15\times20=300
$$

两台相机：

$$
2\times300=600
$$

加上 latent 和 state：

$$
S=600+1+1=602
$$

源码 sequence-first 形状：

```text
encoder input  = (602, 16, 512)
encoder output = (602, 16, 512)
```

### Decoder

```text
decoder input  = (100, 16, 512)
decoder output = (100, 16, 512)
```

转置后：

```text
(16, 100, 512)
```

经过 action head：

```text
predicted action chunk = (16, 100, 6)
```

这正好对应阶段 1 学到的：

```text
(batch, action time, action dimension)
```

---

## 22. Attention 的计算量

Self-attention 需要形成 token 两两之间的 score matrix。

如果序列长度为 $S$：

```text
attention score shape = (S, S)
```

计算和显存通常随 $S^2$ 增长。

在视觉 Transformer 中，visual token 数量可能很大：

- 更高 feature-map 分辨率会增加 token。
- 更多相机会增加 token。
- token 增加会提高 attention 成本。

这也是为什么 ACT 先用 CNN 降低图像空间分辨率，再把 feature map 展开成 token，而不是直接把原始像素全部作为 token。

---

## 23. Transformer 为什么适合动作块

### 并行预测

可以一次预测整个 action chunk，而不是严格逐步生成。

### 时间位置之间可以交流

Decoder self-attention 可以建立未来动作之间的关系。

### 多模态融合

Encoder self-attention 可以融合多个相机、机器人状态和 latent。

### 每个动作位置主动读取 context

Decoder cross-attention 允许不同未来时间位置关注不同场景特征。

### 全局关系

Attention 可以直接建立相距较远 token 的联系，不需要像 RNN 那样逐步传递。

---

## 24. Transformer 不能自动解决什么

Transformer 很强，但它不能自动解决：

- 训练数据没有覆盖的失败状态。
- 相机看不到的关键信息。
- 错误的 action 标签。
- 图像与 action 时间错位。
- 标定和动作尺度错误。
- 真机延迟和安全限制。
- 动力学与接触的物理保证。

Attention 只是更强的信息关系建模工具，并不会自动产生任务常识或恢复能力。

---

## 25. 源码阅读地图

阅读当前 LeRobot ACT Transformer 时，建议按以下顺序：

```text
ACT.__init__()
├── state/action projections
├── CNN backbone
├── image feature projection
├── encoder positional embeddings
├── decoder_pos_embed
└── action_head

ACT.forward()
├── 准备 latent
├── 准备 state token
├── CNN 提取每台相机 feature map
├── flatten visual tokens
├── stack encoder tokens
├── self.encoder(...)
├── 创建 decoder queries
├── self.decoder(...)
└── action_head(...)

ACTEncoderLayer.forward()
├── self-attention
├── residual / norm
└── FFN

ACTDecoderLayer.forward()
├── decoder self-attention
├── cross-attention encoder_out
├── residual / norm
└── FFN
```

阅读时始终在纸上记录：

```text
变量名 | 物理含义 | shape | 属于 encoder 还是 decoder
```

---

## 26. 常见误区

### 误区 1：Token 就是单词

错误。ACT 的 token 可以是图像空间特征、机器人状态或动作位置。

### 误区 2：Self-attention 中 Q、K、V 完全相同

错误。它们来源于同一输入，但经过不同投影矩阵。

### 误区 3：Attention weight 就是模型的完整解释

错误。它可以提供线索，但不能单独证明因果关系。

### 误区 4：没有 position embedding 也能自然知道动作顺序

错误。Attention 本身对排列缺少顺序感知。

### 误区 5：ACT decoder 和 GPT 一样逐个生成动作

错误。ACT 使用动作位置 query 并行预测固定长度 action chunk。

### 误区 6：Encoder 负责生成动作，decoder 只做后处理

错误。Encoder 构建场景 context；decoder 用 action queries 读取 context 并形成动作表示。

### 误区 7：Transformer 能替代数据覆盖和底层控制

错误。模型结构不能消除 dataset 和真实系统接口问题。

---

## 27. 本阶段知识图

```text
不同模态
├── camera images
├── robot state
└── latent
      ↓ projection
统一 D 维 tokens
      ↓ + positional embeddings
Transformer Encoder
├── self-attention
├── residual + norm
└── FFN
      ↓
context tokens
      ↑ cross-attention
action position queries
      ↓ decoder self-attention
Transformer Decoder
      ↓
action representations
      ↓ Linear head
(B, chunk_size, action_dim)
```

---

## 28. 阶段 2 自测

建议先回答问题 1～3，我们再逐步进入形状计算。

### 问题 1：Token

在 ACT 中，下列信息分别如何变成 token？

1. 一张相机图像。
2. 当前机器人关节状态。
3. 未来动作块中的第 10 个时间位置。

### 问题 2：Q、K、V

请用自己的话解释 Query、Key、Value 的职责，并说明为什么 Q 与 K 决定“关注谁”，而 Value 决定“读取什么”。

### 问题 3：Self-Attention 与 Cross-Attention

请说明：

1. ACT encoder self-attention 中 Q/K/V 分别来自哪里？
2. ACT decoder cross-attention 中 Query 来自哪里，Key/Value 来自哪里？

### 问题 4：位置编码

如果 100 个 action query 没有任何位置编码，模型为什么难以区分“未来第 1 个动作”和“未来第 100 个动作”？

### 问题 5：Token 数量

假设两台相机的 CNN feature map 都是 `(B, 512, 15, 20)`，投影后的 `dim_model=512`，另外有一个 state token 和一个 latent token。

1. 每台相机产生多少 visual tokens？
2. Encoder 总 token 数是多少？
3. 如果 batch size 为 16，源码采用 `(sequence, batch, dim)`，encoder input shape 是什么？

### 问题 6：Multi-Head

`dim_model=512`、`n_heads=8` 时，每个 head 的维度是多少？

如果 encoder sequence length 为 602，一个 head 对单个样本产生的 self-attention score matrix 是什么形状？

### 问题 7：Decoder 输出

如果：

- batch size = 16。
- `chunk_size=100`。
- `dim_model=512`。
- `action_dim=6`。

请写出：

1. Decoder input 的 sequence-first shape。
2. Decoder output 转为 batch-first 后的 shape。
3. 经过 action head 后的 shape。

### 问题 8：ACT 与 GPT 的区别

为什么 ACT decoder 可以并行输出动作块，而不必像 GPT 一样逐 token 生成？

---

## 29. 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [DETR：End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [ACT：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [LeRobot ACT 当前实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)

