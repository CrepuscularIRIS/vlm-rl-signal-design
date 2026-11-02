# DTPO step32 E1 结果（2026-11-03）

另一条线的 DTPO（outcome-only 分段损失，A-SFT 起点，32 步，GPU 2/3 训练）固定 step32 的 E1：552 题 × 4 采样，原 `run_eval.sh caption` 口径，`summarize.py` 统一评分（失败/截断留在分母）。

| 指标 | 值 |
|---|---:|
| **acc** | **0.6553**（1,447/2,208） |
| 格式有效率 | 0.9805 |
| mme_realworld | 0.6799 |
| vstar | 0.8638 |
| worldbench | 0.3548 |
| mmmu_pro | 0.3205 |

对照：k8 DAPO 0.6635、k8 GRPO 0.6390、GSPO 0.6395、GiGPO 0.6209（同 E1）。DTPO 未超过 DAPO（差 −0.8 个点），但高于其余三个；vstar 0.8638 为各条件最高。训练侧：32 步 exit0、tracker32、best_val .5995。E1 为开发评测，非盲测。

产物：`summary_dtpo32.json`、`header_dtpo32.json`（模型/题池/代码哈希见 header）；轨迹在云端 `evidence/dtpo_e1_20261103/traj_dtpo32.jsonl`（2,208 条）。
