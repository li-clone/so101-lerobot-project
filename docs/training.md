# ACT 训练

## 环境变量

```bash
export PROJECT_ROOT="$PWD"
export DATA_ROOT="$PROJECT_ROOT/data"
export OUTPUT_ROOT="$PROJECT_ROOT/outputs"
```

运行 v1 或 v2：

```bash
bash scripts/training/train_act.sh v1
bash scripts/training/train_act.sh v2
```

公共脚本固定了本次实验参数：50,000 steps、batch size 16、seed 1000、10% 验证集、每 1,000 step 验证、每 5,000 step 保存、AMP 关闭、W&B 关闭。

服务器训练建议使用 `tmux`、`screen` 或平台后台任务；单纯 SSH `exit` 不应终止后台会话。训练结束必须同时检查 `End of training`、目标 checkpoint 文件和日志中的异常，而不能只看进程消失。

## 选择 checkpoint

```bash
python scripts/training/extract_eval_curve.py \
  "$OUTPUT_ROOT/<run>_console.log" \
  results/training_curves/<run>_eval.csv \
  --run <run>
```

只在实际保存的 checkpoint 中按 `eval_loss` 选择候选，再进行实机评测。v1 选择 45k，v2 选择 30k；同时保留 50k 最终模型用于追溯。

训练产生的 `train_config.json` 中旧摘要字段可能将验证频率显示为 `null`，但实际命令使用 `--eval_steps=1000`，完整日志也每 1,000 step 记录一次验证。本仓库以可执行脚本和提取出的日志曲线共同作为证据。

## Loss 的解释

不同数据集的 loss 不应直接比较。v2 的 loss 低于 v1 并不矛盾：样本数量、动作分布和验证切分都变了。模型是否更好最终由固定协议的实机成功率和失败模式决定。
