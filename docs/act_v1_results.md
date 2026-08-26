# ACT v1 结果

## 数据与训练

| 项目 | 值 |
|---|---|
| 数据集 | `so101_pick_place_v1` |
| episodes / frames | 50 / 16,881 |
| 视觉 | handeye + environment，640×480 |
| FPS | 20 |
| steps / batch size | 50,000 / 16 |
| eval split / interval | 0.1 / 1,000 steps |
| 最佳保存点 | 45,000 |
| 最佳 eval loss | 0.3095 |
| 保留模型 | 45k、50k |

完整验证曲线见 [`act_v1_eval.csv`](../results/training_curves/act_v1_eval.csv)。

## 实机评测

正式部署使用 `n_action_steps=100`。10 次评测成功 8 次，成功率 80%。失败集中于训练覆盖不足的较远物体位置，表现为夹爪未能抓到线束。

该结果促成 v2 的数据策略：保持线束朝向不变，针对近、中、远距离补采20条示教。逐次记录见 [`act_v1_trials.csv`](../results/evaluation_csv/act_v1_trials.csv)。
