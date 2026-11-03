# R-read 离线准入结果（2026-11-03）

依据 [R_READ_FREEZE](../../design/R_READ_FREEZE.zh.md) §离线准入。全部通过，G1 进入完整训练。

## 探针设定

- 策略：A-SFT（G1 起点，`sft_v22_turnwise_merged`），temp 0.7（同训练 rollout），caption v5.1
- 样本：256 题（从 2,699 个 m=1 题按 seed 42 采样；mme_realworld 223 / worldbench 17 / mmmu_pro 16）× 8 rollouts = 2,048 条
- 池 sha16=57617e77a9f7e1c3；工具环与 E1 评测器同源（eval_v4）

## 评分器准入

| 检查 | 结果 |
|---|---|
| 预登记成对反例（14 组：对错数字/枚举/否定/重复/子串守卫/会计括号/百分号/缺失） | **14/14** |
| commit 提取规则用例（grounding 后文本/裁掉先前读数） | **2/2**（合计 16/16） |
| 实数据冒烟：评分器（final_text）vs 训练奖励路径（turns 拼接→commit 提取）在 2,048 条真实轨迹 | **0 偏差** |

## 信号准入（冻结线 ≥10% 完整训练 / 5–10% 有限预算 / <5% 暂停）

| 统计量 | 数值 |
|---|---|
| m=1 覆盖率 | 2,699/4,096 = 65.9% |
| mean Φ_read | 0.5518（K̄=2.86 字段/题） |
| 明确写出 ≥1 个正确读数的轨迹 | 81.1% |
| **有效读取差异率**（116 个无任务信号组中 Φ_read 有方差） | **0.509（59/116）→ 完整训练** |
| ν_allwrong（全错组内 Φ 有方差） | 0.714（10/14） |
| ν_allcorrect（全对组内 Φ 有方差） | 0.480（49/102） |
| ν_fixedview（字段级部分正确 0<Σz<8） | 0.438（320/731；全零字段 186/731） |

组构成：allwrong 14 / allcorrect 102 / mixed 140。

## 信号方向核验

mean Φ_read：allcorrect **0.609** > mixed **0.530** > allwrong **0.356**。
Φ_read 与终答正确同向但不冗余（读对答错保留信号；答对读错扣信号）——与冻结设计一致。

## 判定

有效读取差异率 0.509 远超 10% 准入线 → **G1（R-read）完整训练**。

## 复现

- 评分器/统计：`scripts/rread/`（rread_scorer.py 16 例自检 / admission_stats.py / run_admission.py）
- G1 奖励：服务器 `grpo_r1/`（拷贝自 grpo_d2，换 reward_rread.py；caption_contracts sha16=c84d3dfa 含 phi 日志）
- 原始分数：`scores.jsonl`（2,048 行，uid/sample_idx/vy/K/phi/z）；轨迹在本地 obs/runs（含图像路径，不入公开库）
