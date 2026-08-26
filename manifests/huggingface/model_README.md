---
library_name: lerobot
tags:
- robotics
- lerobot
- so101
- act
- imitation-learning
---

# SO-101 ACT v1/v2 checkpoints

Private model backup for the public project
[`li-clone/so101-lerobot-project`](https://github.com/li-clone/so101-lerobot-project).

## Retained checkpoints

| directory | role | eval loss | deployment |
|---|---|---:|---|
| `act_v1/checkpoints/045000/pretrained_model` | v1 best saved checkpoint | 0.3095 | `n_action_steps=100` |
| `act_v1/checkpoints/050000/pretrained_model` | v1 final checkpoint | 0.3150 | reference |
| `act_v2/checkpoints/030000/pretrained_model` | v2 best saved checkpoint | 0.2643 | `n_action_steps=50` |
| `act_v2/checkpoints/050000/pretrained_model` | v2 final checkpoint | 0.2724 | reference |

Each directory contains model weights, policy config, train config, and the
normalization preprocessor/postprocessor artifacts required by LeRobot `v0.6.1`.
The exact upstream commit is `7e241bd630a3719a56157a497ce5d08f244784f1`.

Full console logs and SHA-256 inventories are stored alongside the models.
The v1 and v2 offline losses are not a controlled cross-dataset comparison;
refer to the public repository for hardware protocols and failure analysis.
