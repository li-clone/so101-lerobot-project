# 数据、模型与公开材料上传清单

## GitHub 公共仓库

上传本仓库中已经跟踪的文档、代码、配置模板、CSV、SVG和 SHA-256 清单。不要上传：

- `data/` 原始数据和评测视频
- `outputs/`、checkpoint、`.safetensors`、完整 `.log`
- calibration JSON 和本地硬件配置
- USB序列号、SSH地址、密码、token
- 未脱敏人员/家庭画面
- 厂商 PDF 原文件

## Hugging Face 私有数据仓库

建议 `kerong111/so101-pick-place-data-private` 保存：

| artifact | 内容 |
|---|---|
| `so101_pick_place_v1` | v1 训练集50条 |
| `so101_pick_place_v2_70` | v2 合并训练集70条 |
| `rollout_act_v1_fixed_eval_10` | v1 实机评测10条 |
| `rollout_act_v2_n50_balanced_eval_15` | v2 平衡评测15条 |

20条补数不重复保存完整副本；`manifests/datasets/v2_provenance.csv` 说明其对应 v2 episodes 50–69。上传前在数据仓库根目录附 README、元信息和 SHA-256，并保持仓库为 private。

## Hugging Face 私有模型仓库

建议 `kerong111/so101-act-models-private` 保存：

- v1：45k最佳模型、50k最终模型
- v2：30k最佳模型、50k最终模型
- 每个模型的 config、train config、preprocessor/postprocessor和权重
- `selected_models.sha256`、完整训练日志、模型卡和评测摘要

GitHub 的 `manifests/models/` 只公开逻辑 artifact ID 与 SHA-256，不包含 Hugging Face token。

## 上传前检查

```bash
git status --short
git grep -nE '(/home/[a-z][a-z0-9_-]+|USB_Single_Serial_[A-Za-z0-9]+)'
git ls-files | grep -E '\.(safetensors|mp4|log|pdf)$' && exit 1 || true
```

只有敏感扫描为空、清单校验成功、视频完成脱敏后，才创建 GitHub remote 或推送。
