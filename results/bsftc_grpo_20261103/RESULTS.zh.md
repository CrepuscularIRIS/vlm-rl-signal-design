# bsftc（因果 SFT + GRPO）训练报告（三部分格式）

2026-11-03 | run: `outputs/v52mc_bsftc_grpo` | GPU 0,1 | 步 1–20 + 恢复 21–32

## 一、设定

| 项 | 值 |
|---|---|
| SFT 起点 | **B-SFT causal**（`sft_bsft_causal_merged`，E1 .4543）——因果叙事格式语料 4,040 条（用户指示的另一种 SFT 协议） |
| RL | 与 k8_grpo 逐项相同：无过滤 GRPO，clip .2/.3，lr 1e-6 const，无 KL，seed 42，4k×8，32 步 |
| 奖励 | 纯终答 V·Y（无过程项） |
| **唯一变量 vs k8_grpo** | **SFT 起点：A-SFT(turnwise) → B-SFT(causal)** |
| 事故与恢复 | ① 启动崩 `rope_scaling=None`（LLaMA-Factory 导出损伤，对照修复）② 20 步处因双训练内存减载暂停于完整 ckpt_20 ③ 并行恢复（用户指示）step21 backward CUDA OOM（GPU0 需 6.49G 仅剩 6.33G）④ U1 恢复（update micro 2→1，梯度累积等价）跑完 21–32。步 1–20 的 stdout 熵等指标被 log 覆盖丢失（标缺测），rollouts/updates 完整 |

## 二、定性 + 定量

### Case（行为差异）

bsftc 裁图/条 **1.254**（A-SFT+GRPO 1.905）：因果 SFT 起点的策略显著少裁图。典型轨迹：多数题直接原图 commit 或 1 次裁图即作答；相对地 A-SFT 起点平均多一次探索性裁图。分源上两者共同弱项相同（mmmu .2308 vs k8 .3077，worldbench .3226 vs .3266）。

### 定量

| | B-SFT 起点 | +GRPO (bsftc) | A-SFT 起点 | +GRPO (k8) |
|---|---:|---:|---:|---:|
| **E1** | .4543 | **.6033** (1332/2208) | .5670 | **.6390** |

- **GRPO 把弱 SFT 起点拉升 +14.9pp**（.4543→.6033）；A-SFT 侧 +7.2pp（.5670→.6390）
- 两种 SFT 协议的 E1 差距 **11.3pp → 3.6pp**（RL 收窄约 2/3）
- 逐题配对 vs k8_grpo：共同对 1076 / 仅 bsftc 对 256 / 仅 k8 对 335 / 共同错 541（**net −79**）
- E1 分源：mme .6291 / vstar .8103 / wb .3226 / mmmu .2308
- val（4k 池）reward：step5 .5095 → step32 .5853

**判读**：SFT 协议（turnwise vs causal）决定 RL 前的起点高低；RL 之后差距大幅收窄但未消除。终答上限仍由 A-SFT+GRPO 占优。

## 三、训练 log 曲线

图：`curves/family_curves.png`（左：训练曲线全家福含 bsftc；右：E1 终点条形）
- train accuracy：0.589 → ~0.61（rollouts 重建，32 步完整；步 1–20 熵/grad_norm 缺测不画）
- 更新账目：步 13–15 ledger all_correct 28→42（升）；步 21 后 U1 恢复，correct 均值与 U2 段连续（.62→.61）
