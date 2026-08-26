---
pretty_name: SO-101 Pick-and-Place ACT Private Artifacts
task_categories:
- robotics
tags:
- lerobot
- so101
- imitation-learning
- act
---

# SO-101 Pick-and-Place datasets

Private artifact backup for the public project
[`li-clone/so101-lerobot-project`](https://github.com/li-clone/so101-lerobot-project).

## Contents

| directory | purpose | episodes | frames |
|---|---|---:|---:|
| `training/so101_pick_place_v1` | ACT v1 training | 50 | 16,881 |
| `training/so101_pick_place_v2_70` | ACT v2 merged training | 70 | 23,868 |
| `evaluation/rollout_act_v1_fixed_eval_10` | ACT v1 hardware evaluation | 10 | 2,965 |
| `evaluation/rollout_act_v2_n50_balanced_eval_15` | ACT v2 balanced hardware evaluation | 15 | 5,119 |

All datasets use LeRobot dataset codebase version `v3.0`, 20 FPS, 6-dimensional
state/action, and two 640×480 RGB views named `handeye` and `environment`.

The 20 targeted v2 demonstrations are represented by merged v2 episodes 50–69
and are not uploaded as a duplicate standalone dataset. See `manifests/v2_provenance.csv`.

This repository is private because the raw third-person evaluation recordings
may contain household context. Do not make it public without frame-by-frame privacy review.
