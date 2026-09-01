# 阶段 6：MuJoCo 与机器人仿真

## 0. 本阶段定位

前面我们已经学习了：

~~~text
关节空间与任务空间
FK / IK / Jacobian
质量 / 惯性 / 重力 / 力矩
Policy → 控制器 → 真实机械臂
~~~

现在把这些概念放进一个可计算、可观察、可重复的虚拟世界：MuJoCo。

本阶段只学习理论和源码阅读框架，不接管 Ubuntu 工程，也不要求现在部署 SO-101 仿真。

完成本阶段后，你应该能够：

1. 区分 physics engine、simulator 和 learning environment。
2. 理解 MuJoCo 模型与运行状态分别是什么。
3. 看懂 MJCF 中 body、joint、geom、actuator、sensor、camera 的作用。
4. 理解 qpos、qvel、ctrl 与 observation/action 的关系。
5. 解释一次 physics step 内发生了什么。
6. 区分 physics、control、policy 和 rendering frequency。
7. 理解接触、摩擦为什么是抓取仿真的核心。
8. 解释 sim-to-real gap、system identification 和 domain randomization。
9. 把 MuJoCo 环境循环连接到 ACT/LeRobot Policy 循环。

---

## 1. 为什么机器人学习需要仿真

真实机器人不可替代，但实验存在明显成本：

- 采集速度慢；
- 机械臂会磨损；
- 错误动作可能损坏硬件；
- 环境复位通常需要人工操作；
- 很难严格复现同一次实验；
- 一些真实状态无法直接测量；
- 同时运行大量真机成本很高。

仿真提供了可控实验场：

~~~text
设置初始状态
→ 执行 action
→ 物理引擎计算运动与接触
→ 得到新 observation
→ 重复
~~~

它适合：

- 验证 observation/action 接口；
- 理解控制频率和延迟；
- 测试碰撞与关节限制；
- 快速复位并重复实验；
- 批量评估 Policy；
- 在不损坏真机的情况下发现明显错误；
- 研究强化学习和 sim-to-real。

但必须记住：

> 仿真成功只说明 Policy 在这个仿真模型中成功，不代表真机一定成功。

---

## 2. Physics Engine、Simulator、Environment

### 2.1 Physics Engine

物理引擎负责计算：

~~~text
力与力矩
＋
质量、惯性、重力
＋
关节约束、碰撞、摩擦
↓
加速度、速度和位置如何变化
~~~

MuJoCo 的核心角色是高性能刚体动力学与接触计算引擎。

### 2.2 Simulator

模拟器通常进一步包含：

- 机器人与场景模型；
- 时间推进；
- 执行器；
- 传感器；
- 相机渲染；
- viewer 和交互工具。

所以 MuJoCo 也常被整体称为机器人模拟器。

### 2.3 Learning Environment

学习环境在模拟器外再定义：

~~~text
reset()
step(action)
observation
reward
terminated / truncated
任务成功条件
~~~

因此：

~~~text
MuJoCo physics
≠
完整机器人学习任务
~~~

同一个 MuJoCo 机器人模型可以被包装成抓取、推动、开抽屉等不同任务。

MuJoCo 也不是强化学习算法。ACT、Diffusion Policy 或 RL agent 才是产生动作的策略；MuJoCo 负责计算动作产生的物理后果。

---

## 3. 仿真的基本闭环

最基本的环境循环是：

~~~text
Simulator state s_t
        ↓ 传感器读取或图像渲染
Observation o_t
        ↓
Policy π
        ↓
Action a_t
        ↓ 动作映射、限幅、控制器
Control input u_t
        ↓
Physics engine
        ↓
Simulator state s_{t+1}
~~~

数学上可写成：

$$
\mathbf a_t=\pi(\mathbf o_t)
$$

$$
\mathbf s_{t+1}=F(\mathbf s_t,\mathbf u_t,\Delta t)
$$

$$
\mathbf o_{t+1}=G(\mathbf s_{t+1})
$$

其中：

- state 是模拟器内部完整状态；
- observation 是 Policy 实际得到的信息；
- action 是 Policy 输出；
- control 是最终写给 actuator 的输入；
- F 表示动力学、接触和数值积分；
- G 表示传感器与 observation 构造。

最重要的边界是：

> Simulator state 通常比 Policy observation 更完整。

例如模拟器知道杯子的精确三维位姿与速度，但视觉 ACT 可能只获得 RGB 图像和关节位置。

---

## 4. State、Observation、Action、Control

### 4.1 State

足以继续预测系统未来的内部信息，例如：

- 所有关节位置和速度；
- 自由物体的位置、姿态和速度；
- 执行器内部状态；
- 其他动力学状态。

### 4.2 Observation

从 state 中选择、加工或渲染后提供给 Policy：

- RGB camera；
- joint position；
- joint velocity；
- touch 或 force sensor；
- 可选任务信息。

Observation 可以是部分可观测的。杯子真实存在于 state 中，但被遮挡后可能没有出现在 RGB observation 中。

### 4.3 Action

Policy 输出的语义可能是：

- 目标关节位置；
- 关节位置增量；
- 目标速度；
- 关节力矩；
- 末端位姿增量。

### 4.4 Control

Control 是最终写给 MuJoCo actuator 的输入。

Action 与 control 不一定相等：

~~~text
Policy normalized joint target
→ inverse normalization
→ joint name mapping
→ safety clipping
→ actuator ctrl
~~~

也可能是：

~~~text
Policy end-effector delta
→ IK / Cartesian controller
→ joint target or torque
→ actuator ctrl
~~~

因此同为六维，并不代表 action 与 ctrl 的语义相同。

---

## 5. MuJoCo 的 mjModel 与 mjData

### 5.1 mjModel

mjModel 保存相对固定的世界结构和物理参数：

- body、joint、geom 的层级；
- 质量和惯性；
- 关节范围；
- actuator 参数；
- sensor 与 camera 定义；
- physics timestep；
- solver 等配置。

可以把它理解成：

~~~text
这个虚拟世界由什么组成，以及遵循什么物理规则
~~~

### 5.2 mjData

mjData 保存随仿真不断变化的状态和计算结果：

- qpos：广义位置；
- qvel：广义速度；
- ctrl：actuator 控制输入；
- 仿真时间；
- sensor 输出；
- 接触信息；
- 力、加速度和派生坐标。

可以把它理解成：

~~~text
这个虚拟世界此刻是什么状态
~~~

### 5.3 为什么分开

同一个 mjModel 可以创建多个 mjData：

~~~text
相同机器人与场景结构
＋
不同初始状态
→
多个独立 rollout
~~~

这对批量评估和并行仿真非常有用。

---

## 6. 从 MJCF/URDF 到运行时模型

MuJoCo 可以读取原生 MJCF，也能读取较受限制的 URDF，然后编译为运行时 mjModel：

~~~text
MJCF / URDF XML
→ parser
→ high-level specification
→ compiler
→ mjModel
~~~

官方文档强调，所有运行时计算最终基于编译后的 mjModel。

### 6.1 URDF

URDF 在 ROS 生态中常见，主要描述：

- link；
- joint；
- visual；
- collision；
- inertial。

### 6.2 MJCF

MJCF 是 MuJoCo 原生格式，更直接支持：

- actuator；
- sensor；
- tendon；
- contact 参数；
- defaults 继承；
- simulation option；
- 完整场景和任务元素。

即使机器人来自 URDF，完整任务通常还需要补充执行器、接触、场景、传感器和控制配置。

### 6.3 能加载不等于正确

XML 编译成功只说明结构和数值合法，不代表它准确复现真机。还要验证：

- 连杆长度；
- 关节轴与正负方向；
- 关节零位；
- 质量和惯性；
- 摩擦；
- actuator 增益；
- 关节范围；
- mesh 比例和原点；
- 相机外参。

---

## 7. MJCF 核心元素

### 7.1 worldbody

所有刚体层级的根节点，通常包含机械臂、地面、桌子、物体、灯光和相机。

### 7.2 body

一个刚体坐标系和层级节点。嵌套 body 表达串联机械臂：

~~~text
base
└── shoulder
    └── upper_arm
        └── forearm
            └── wrist
~~~

它与阶段5中的齐次变换链对应。

### 7.3 joint

定义 body 相对于父 body 如何运动：

- hinge：绕一根轴旋转；
- slide：沿一根轴平移；
- ball：三维旋转；
- free：三维平移加三维旋转。

### 7.4 geom

几何体可参与可视化、碰撞和质量/惯性推断。

视觉 mesh 很精细，不代表适合直接作为碰撞几何。碰撞常使用更简单、稳定的近似形状。

### 7.5 site

无质量的标记点或局部坐标系，常用于：

- 标记末端执行器；
- 放置 sensor；
- 表示目标点；
- 读取局部位姿；
- 定义 tendon 路径。

### 7.6 actuator

定义控制如何作用到 joint 或 tendon。它可能表示力矩型 motor、position servo、velocity servo 或更一般的增益模型。

### 7.7 sensor

可模拟 joint position、velocity、touch、force/torque、accelerometer、gyro 或 frame pose 等传感器。

### 7.8 camera

定义渲染相机的位置、方向和视场角。相机图像由 rendering 生成，不是 qpos 数组本身。

---

## 8. qpos 不是简单的关节角列表

qpos 保存广义位置：

~~~text
hinge joint → 1个角度
slide joint → 1个位移
ball joint  → 4维 quaternion
free joint  → 3维位置 + 4维 quaternion
~~~

因此：

> len(qpos) 不一定等于 joint 数量。

自由刚体在 qpos 中占 7 维，但它的速度是：

~~~text
3维线速度 + 3维角速度
~~~

所以在 qvel 中占 6 维。

这意味着：

$$
n_q=\operatorname{len}(\mathbf{qpos})
$$

$$
n_v=\operatorname{len}(\mathbf{qvel})
$$

并不一定相等。

对于固定基座、全部由单自由度 hinge 构成的机械臂，手臂部分的 qpos 很像关节角列表。但场景中的杯子如果有 free joint，就会引入额外的 7 维 qpos 和 6 维 qvel。

因此读取状态时不应盲目假设固定数组位置，最好根据 joint name 和 address 映射。

---

## 9. ctrl 也不一定是关节角

mjData.ctrl 是 actuator 输入：

$$
\mathbf{ctrl}\in\mathbb R^{n_u}
$$

每个值的物理意义由 actuator 定义决定。

### Motor actuator

ctrl 可能映射为力或力矩相关输入。

### Position actuator

ctrl 通常表示目标位置，actuator 根据位置误差产生作用：

~~~text
error = target_position - current_position
→ actuator force
~~~

### Velocity actuator

ctrl 通常表示目标速度。

所以看到：

~~~python
data.ctrl[i] = value
~~~

不能立即断言 value 是角度、速度还是力矩，必须检查对应 actuator 配置。

还要区分：

- nq：广义位置维度；
- nv：广义速度维度；
- nu：actuator control 维度。

一个 joint 可能没有 actuator，也可能通过 tendon 或耦合结构驱动，所以三者不必相同。

---

## 10. 一次 physics step 发生什么

高层直觉如下：

~~~text
读取当前 qpos / qvel
＋
读取 ctrl
＋
计算 actuator force
＋
计算重力、惯性、摩擦和约束
＋
检测并求解接触
↓
得到 acceleration
↓ 数值积分
更新 qvel 和 qpos
↓
更新时间与派生状态
~~~

可以粗略连接到动力学方程：

$$
\mathbf M(\mathbf q)\ddot{\mathbf q}
+\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}
+\mathbf g(\mathbf q)
=
\boldsymbol\tau+\mathbf J_c^T\boldsymbol\lambda
$$

其中：

- τ 表示 actuator 等产生的广义力；
- Jc 表示接触约束 Jacobian；
- λ 表示接触约束作用。

不用推导公式，只要知道碰撞后物体不会简单穿透，而是通过接触约束求解影响运动。

---

## 11. mj_forward 与 mj_step

### mj_forward

根据当前 qpos、qvel 等状态重新计算运动学、动力学、传感器等派生量，但不把时间推进一个完整仿真步。

典型情况：

~~~text
手动修改 qpos
→ mj_forward
→ body/site/camera 等结果与新状态一致
~~~

### mj_step

执行一次完整动力学时间推进和积分，使系统进入下一时刻。

简单记忆：

~~~text
mj_forward：根据“现在”重新计算
mj_step：把物理时间推进到“下一刻”
~~~

它们也与神经网络的 policy.forward() 不同。policy.forward() 是模型计算；mj_forward 是物理模型派生量计算。

---

## 12. 四种频率

### 12.1 Physics Frequency

由 physics timestep 决定：

$$
f_{physics}=\frac{1}{\Delta t_{physics}}
$$

若 timestep 为 0.002 s：

$$
f_{physics}=500\text{ Hz}
$$

### 12.2 Control Frequency

同一个 control 可以保持多个 physics steps。

若每个 action 保持10个 physics steps：

$$
\Delta t_{control}=10\times0.002=0.02\text{ s}
$$

$$
f_{control}=50\text{ Hz}
$$

这类参数常称 action repeat、frame skip 或 control decimation。

### 12.3 Policy Inference Frequency

若 ACT 一次预测20个动作，并按50 Hz发送：

$$
f_{policy}=\frac{50}{20}=2.5\text{ Hz}
$$

### 12.4 Rendering Frequency

渲染频率也可独立设置。因此一个系统可能同时是：

~~~text
Physics：500 Hz
Control：50 Hz
ACT forward：2.5 Hz
Rendering：30 Hz
~~~

### 12.5 Simulation Time 与 Wall Time

仿真推进1秒物理时间，不一定占用现实中的1秒：

- 无渲染、模型简单时可以快于实时；
- 高分辨率视觉或复杂接触时可能慢于实时；
- viewer 按实时速度播放，不代表物理引擎只能实时计算。

---

## 13. Rendering 不等于 Physics

MuJoCo 不显示窗口也能进行物理计算。

~~~text
Physics：状态如何变化
Rendering：根据状态生成图像
Viewer：供人观察和交互
~~~

状态型 Policy 直接读取数值时，可以不渲染 RGB。视觉 Policy 则必须考虑：

- 相机外参；
- 视场角；
- 分辨率；
- 光照和阴影；
- 材质纹理；
- 遮挡；
- 图像噪声与延迟。

物理运动一致但图像差异很大，仍可能使视觉 Policy 失败。

---

## 14. 碰撞、接触与摩擦

抓取不是“夹爪到达物体位置”就结束，还依赖接触作用。

### Collision Detection

引擎判断哪些 geom 接近或发生接触。

### Contact Constraint

接触求解器计算阻止过度穿透并满足接触约束所需的作用。

### Friction

摩擦影响：

- 夹爪能否夹住物体；
- 物体会不会滑落；
- 推动物体时如何滑动；
- 桌面上的物体如何停止。

摩擦太大，仿真抓取可能异常容易；摩擦太小，物体可能总是滑落。

接触结果还受到以下因素影响：

- geom 形状和尺寸；
- contact 参数；
- solver；
- physics timestep；
- actuator force；
- 物体质量和惯性；
- collision mesh 质量。

所以“动作正确但抓不住”不一定是 Policy 问题。

---

## 15. Actuator 与真实舵机

真实 SO-101 链路大致是：

~~~text
Goal_Position
→ 舵机内部位置控制
→ 电机与齿轮传动
→ 关节运动
~~~

MuJoCo 需要用 actuator 模型近似这条链。

如果仿真关节可以瞬间、无限力地到达目标，Policy 可能依赖真机不存在的控制能力。

真实系统还有：

- 最大力矩与速度；
- 通信和控制延迟；
- 死区；
- 摩擦与齿隙；
- 跟踪误差；
- safety clipping。

更接近真机的模型可能需要匹配：

- actuator gain；
- control/force range；
- joint damping 与 friction；
- 控制延迟；
- 目标变化限制；
- 真机 control frequency。

但模型越复杂不一定越准确。错误的复杂参数可能不如简单、经过验证的模型。

---

## 16. Reset、随机性与复现

Episode reset 通常需要：

~~~text
恢复机器人初始 qpos
→ 放置物体
→ 清除 qvel 与控制缓存
→ 重置时间
→ 刷新派生状态
~~~

如果只修改 qpos 却保留旧 qvel，新 episode 可能继承上一轮速度。

随机种子可能影响：

- 物体初始位姿；
- 光照与纹理；
- sensor noise；
- Policy 自身采样。

调试时固定 seed，有助于区分代码变化和场景随机变化。

但不同硬件、版本、线程或浮点路径仍可能产生细微差异，所以确定性需要在实验中实际验证。

---

## 17. Reward、Success 与 Episode 结束

强化学习常使用 reward：

$$
r_t=R(s_t,a_t)
$$

Behavior Cloning 和 ACT 的训练主要依赖 demonstration action 标签，并不要求 reward 直接参与梯度更新。

但仿真评估仍需 success metric，例如：

- 是否抓住物体；
- 是否抬高超过阈值；
- 是否放到目标区域；
- 是否在规定时间内完成。

还要区分：

- terminated：任务自然到达成功或失败终态；
- truncated：因为时间上限等外部限制停止。

超时结束与进入真实失败终态，语义并不完全相同。

---

## 18. MuJoCo 与 LeRobot 的接口连接

MuJoCo 提供底层 state、sensor 和 rendering，LeRobot Policy 希望看到统一的 observation/action 接口。中间需要适配层：

~~~text
MuJoCo qpos / qvel / camera
→ observation dictionary
→ LeRobot preprocessor
→ ACT Policy
→ normalized action
→ postprocessor
→ MuJoCo action adapter
→ data.ctrl
→ physics steps
~~~

### Observation mapping

~~~text
rendered RGB
→ observation.images.front

selected arm qpos
→ observation.state
~~~

### Action mapping

~~~text
ACT 6维目标位置
→ inverse normalization
→ actuator name mapping
→ safety clipping
→ data.ctrl
~~~

最危险的问题是“维度相同，但语义不同”。必须对齐：

- joint order；
- 正负方向；
- 单位；
- absolute/delta；
- position/velocity/torque；
- gripper 开合方向；
- control frequency。

---

## 19. 仿真模仿学习数据

仿真数据流程与真机概念一致：

~~~text
仿真遥操作或专家控制器
→ 记录 episodes
→ observation/action Dataset
→ DataLoader
→ ACT training
→ checkpoint
→ simulation rollout evaluation
~~~

### 优势

- 快速复位；
- 可精确读取状态；
- 可控制随机化范围；
- 可大量并行；
- 可自动标记成功。

### 风险

- 图像太干净；
- actuator 太理想；
- contact 参数不真实；
- 专家使用了真机不可获得的信息；
- 轨迹分布与真实示教不同。

### Privileged Information

仿真器可以知道杯子精确位置，它可用于：

- 专家控制器；
- reward；
- success 判断；
- teacher 模型。

但如果真机部署只有 RGB，最终 student Policy 的 observation 不应依赖该精确坐标。

---

## 20. Sim-to-Real Gap

Sim-to-real gap 由多个差异叠加而成。

### 20.1 Visual Gap

- 光照、纹理、阴影；
- 曝光、白平衡、镜头畸变；
- 背景杂物；
- 图像噪声；
- camera pose 与视场角误差。

### 20.2 Dynamics Gap

- 质量和惯性；
- 摩擦和阻尼；
- actuator force；
- 齿轮间隙；
- contact 软硬程度；
- 抓取负载。

### 20.3 Control Gap

- 真机通信延迟；
- action 时间抖动；
- 舵机内部 PID；
- 最大速度和力矩；
- safety clipping；
- control frequency。

### 20.4 Geometry / Calibration Gap

- 连杆长度；
- 关节零位；
- joint axis；
- mesh 原点与比例；
- 相机外参；
- 桌面高度和 gripper 尺寸。

### 20.5 Task Distribution Gap

- 物体位置范围；
- 物体种类和背景；
- 人工复位分布；
- 仿真 demonstration 过于理想。

---

## 21. System Identification

System identification 根据真机数据估计仿真参数，使模拟响应更接近真实响应：

~~~text
真机与仿真发送相同目标
→ 比较 joint response
→ 调整 gain、damping、friction、delay
→ 减少轨迹误差
~~~

抽象写成：

$$
\theta^*
=
\arg\min_\theta
\sum_t
\left\|
\mathbf y_t^{real}
-
\mathbf y_t^{sim}(\theta)
\right\|^2
$$

需要注意：

- 只拟合一条轨迹容易过拟合；
- 多种动作、姿态和负载更有辨识价值；
- 某些不同参数组合可能产生相似结果；
- 视觉参数和动力学参数需要不同的测量方式。

---

## 22. Domain Randomization

Domain randomization 不追求唯一完美仿真，而是在训练中随机改变不确定参数：

~~~text
每个 episode：
随机光照和纹理
随机 camera 小偏移
随机物体质量和摩擦
随机控制延迟
随机初始位置
~~~

目标是让 Policy 不依赖某个精确仿真假象。

但随机范围不是越大越好。过大可能：

- 产生不合理世界；
- 使学习问题过难；
- 降低动作精度；
- 让 Policy 学不到稳定规律。

合理做法是先用测量和 system identification 得到大致范围，再围绕真实不确定性随机化。

---

## 23. 常见“仿真假成功”

### 接触假成功

物体看似被夹住，实际依赖了 geom 穿透或异常接触参数。

### 执行器过强

机械臂无视负载、瞬间到达目标，真机无法复现。

### 使用 Privileged State

Policy 直接读取杯子精确坐标，而真机只有 RGB。

### Reset 泄漏答案

物体总在固定位置，Policy 只记住固定轨迹。

### 指标太宽松

只检测“夹爪接近”，却将其当作“稳定抓取”。

### 忽略延迟

仿真 action 立即生效，真机却有相机、推理和通信延迟。

---

## 24. 从现象定位问题

### 机器人加载后爆炸或飞走

可能原因：

- 初始 geom 穿透；
- 质量/惯性不合理；
- actuator gain 太大；
- physics timestep 太大；
- 初始关节状态违反约束。

### Policy action 正常但机器人不动

检查：

~~~text
action mapping
→ ctrl 是否写入
→ actuator 是否连接正确 joint
→ control range 是否裁剪
→ actuator 是否有足够输出
~~~

### 夹爪包住物体但物体滑落

检查：

- gripper action 方向和范围；
- gripper collision geom；
- friction；
- actuator force；
- object mass；
- contact/solver/timestep。

### State Policy 成功但 Vision Policy 失败

优先检查 camera pose、rendering、图像 shape/normalization、遮挡和视觉数据量。

### 仿真成功但真机关节方向相反

优先检查 joint sign、motor order、calibration、action semantics 和坐标系约定。

---

## 25. 最小循环的理论阅读

下面仅用于理解对象关系：

~~~python
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

while data.time < 1.0:
    observation = make_observation(model, data)
    action = policy(observation)
    data.ctrl[:] = action_to_control(action)
    mujoco.mj_step(model, data)
~~~

逐行对应：

~~~text
MjModel.from_xml_path
→ 解析并编译场景

MjData(model)
→ 创建一个动态状态实例

make_observation
→ 从 state/sensor/rendering 构造 Policy 输入

policy
→ observation 到 action

action_to_control
→ 反归一化、映射、控制和限幅

data.ctrl
→ actuator 输入

mj_step
→ 推进一个 physics timestep
~~~

实际环境常在一次 control 后连续调用多次 mj_step，形成 action repeat。

---

## 26. 与 ACT Action Chunk 的连接

假设：

~~~text
physics timestep = 0.002 s
action repeat = 10
n_action_steps = 20
~~~

Physics frequency：

$$
f_{physics}=\frac{1}{0.002}=500\text{ Hz}
$$

Control frequency：

$$
f_{control}=\frac{500}{10}=50\text{ Hz}
$$

一个 ACT chunk 的执行时长：

$$
T_{chunk}=\frac{20}{50}=0.4\text{ s}
$$

理想 Policy forward frequency：

$$
f_{policy}=\frac{50}{20}=2.5\text{ Hz}
$$

嵌套循环是：

~~~text
ACT forward一次
→ 得到20个actions

对每个action：
    写入ctrl
    连续运行10次physics step

20个action执行完
→ 读取新observation
→ ACT再次forward
~~~

---

## 27. 常见误区

1. **MuJoCo 是强化学习算法。**  
   错。它是物理仿真与建模工具。

2. **Simulator state 就是 Policy observation。**  
   错。Observation 通常只是 state 的一部分或其渲染结果。

3. **qpos 长度一定等于 joint 数量。**  
   错。ball/free joint 使用 quaternion。

4. **data.ctrl 一定是关节角。**  
   错。其语义取决于 actuator 和环境映射。

5. **一次 mj_step 等于一次 Policy forward。**  
   错。一个 control 常保持多个 physics steps，一个 ACT forward 又可产生多个 control actions。

6. **画面正确说明动力学正确。**  
   错。视觉 mesh 正常时，质量、碰撞和 actuator 仍可能错误。

7. **仿真成功说明真机一定成功。**  
   错。还存在 visual、dynamics、control、calibration 和 task gap。

8. **Domain randomization 越大越好。**  
   错。过宽随机化会产生不合理世界并增加学习难度。

9. **ACT 训练一定需要 reward。**  
   错。BC/ACT 训练主要使用 demonstration action；reward 或 success 常用于专家、RL 和评估。

10. **MuJoCo 替代了运动学与动力学理论。**  
    错。MuJoCo 数值计算这些关系，而理论是正确建模和诊断的基础。

---

## 28. 本阶段知识图

~~~text
MJCF / URDF
     │ parser + compiler
     ▼
  mjModel ─────────────────────────────┐
  结构、质量、关节、actuator            │
                                      ▼
                                  Physics Engine
                                      │
  mjData                              │
  qpos/qvel/ctrl/sensor ◄─────────────┘
     │                                 ▲
     │ observation adapter             │ control adapter
     ▼                                 │
Camera + Robot State              Policy Action
     │                                 ▲
     ▼                                 │
   ACT Policy ─────────────────────────┘

嵌套频率：
Physics step > Control step > Policy forward

Simulation
  ├── visual gap
  ├── dynamics gap
  ├── control/latency gap
  ├── geometry/calibration gap
  └── task distribution gap
             ↓
        Sim-to-Real
~~~

---

## 29. 阶段 6 自测

建议先回答问题1～3，我们再继续后面的计算与诊断题。

### 问题 1：概念边界

请分别解释：

1. Physics engine 负责什么？
2. Simulator 在 physics engine 之外还包含什么？
3. Learning environment 又额外定义什么？
4. MuJoCo 是否等于强化学习算法？

### 问题 2：State 与 Observation

模拟器知道杯子的精确三维位置、姿态和速度，但 ACT 只收到两路 RGB 图像与机器人关节位置。

1. 哪些属于 simulator state？
2. 哪些属于 Policy observation？
3. 为什么不能把杯子精确坐标加入训练 observation，然后期待同一模型直接部署到只有 RGB 的真机？

### 问题 3：qpos、qvel 与 ctrl

一个场景包含：

- 5个单自由度 hinge 手臂关节；
- 1个单自由度 gripper joint；
- 1个带 free joint 的杯子。

假设没有其他 joint：

1. nq = len(qpos) 是多少？
2. nv = len(qvel) 是多少？
3. 如果只给6个机械臂关节配置 actuator，nu = len(ctrl) 是多少？
4. 为什么 nq、nv、nu 不相等？

提示：free joint 在 qpos 占7维，在 qvel 占6维。

### 问题 4：频率

配置为：

~~~text
physics timestep = 0.001 s
action repeat = 20
n_action_steps = 10
~~~

请计算：

1. physics frequency；
2. control frequency；
3. 一个 action chunk 的执行时长；
4. 理想 Policy forward frequency。

### 问题 5：Actuator 语义

代码中出现：

~~~python
data.ctrl[2] = 0.5
~~~

为什么不能只看这一行就断言“第三个关节目标角度是0.5 rad”？至少还需检查哪些定义？

### 问题 6：mj_forward 与 mj_step

1. 手动修改 qpos 后，为刷新 body/site 位置但不推进时间，应使用哪个函数？
2. 让系统按照动力学进入下一时刻，应使用哪个函数？
3. 二者与神经网络 policy.forward() 有什么区别？

### 问题 7：抓取失败诊断

ACT 的夹爪轨迹看起来正确，夹爪也包住了方块，但方块总是滑落。请提出至少五个仿真层面的检查方向。

### 问题 8：Sim-to-Real

仿真成功率95%，真机只有20%。请从以下五类分别提出可能差异：

1. visual；
2. dynamics；
3. control/latency；
4. geometry/calibration；
5. task distribution。

### 问题 9：System Identification 与 Domain Randomization

1. System identification 的目标是什么？
2. Domain randomization 的目标是什么？
3. 为什么二者可以共同使用？
4. 为什么不能把所有参数在极大范围内随意随机？

### 问题 10：综合判断

判断正误并解释：

1. 仿真器能访问杯子精确坐标，所以视觉 ACT 在真机也一定能访问。
2. RGB rendering 正确，说明物体质量和摩擦也一定正确。
3. 固定基座的单自由度 hinge joint 在 qpos 中通常各占1维。
4. free joint 在 qpos 和 qvel 中都占6维。
5. 同一个 control 可以保持多个 physics steps。
6. Physics、control、Policy inference frequency 可以不同。
7. ACT 训练不一定需要 reward，但仿真评估仍可使用 success metric。
8. Domain randomization 可减少对某个精确仿真假象的依赖，但不能保证完全消除 sim-to-real gap。

---

## 30. 本阶段完成标准

如果你能不看讲义解释下面这段话，就完成了阶段6：

> MuJoCo 使用由 MJCF/URDF 编译得到的 mjModel 描述世界结构，用 mjData 保存 qpos、qvel、ctrl 和 sensor 等运行状态。Policy observation 只是 simulator state 的一部分，Policy action 也可能经过映射才成为 actuator control。一次 mj_step 根据控制、动力学、接触和积分推进状态；多个 physics steps 可以共享一个 control，多个 control actions 又可以来自一个 ACT chunk。仿真与真机间还存在视觉、动力学、控制、几何标定和任务分布差异，需要通过模型验证、system identification、合理随机化与真机评估逐步缩小。

---

## 31. 参考资料

- [MuJoCo 官方概览](https://mujoco.readthedocs.io/en/stable/overview.html)
- [MuJoCo 官方建模指南](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [MuJoCo 官方编程指南](https://mujoco.readthedocs.io/en/stable/programming/)
- [MuJoCo 官方 GitHub](https://github.com/google-deepmind/mujoco)
- [LeRobot Robot API](https://github.com/huggingface/lerobot/blob/main/docs/source/api/robots.mdx)

