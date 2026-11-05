# S 奖励的 psi 审计与 v2 改写（2026-11-05）

适用对象：A+B、A+B+C、GRPO+A 的过程奖励 S，即 R = V·(Y + 0.1·m·S)。DAPO-SB、Hτ 不用这个 S。只列事实。

## 1. v1 实现（`code/sg_scale_gated/reward_sg.py::_psi`，A+B / A+B+C / GRPO+A 实际训练用的版本）
```
z_b   = max_v 1[cover(b,v) ≥ 0.9] · h_kind(detail_b · s(v))     视图 v 含原图；h 为未校准的像素尺度坡
phi   = max_{S∈evidence_sets} mean_{b∈S} z_b
gate  = 任一关键框 z_b ≥ 0.5
psi   = 0.25 + 0.75·phi                               若 gate 开
      = 0.25 · |∪{裁图 v : s(v) ≥ 2 s(v0)}| / (W·H)   若 gate 关（“探索分”）
S     = clip(psi_T − psi_0, 0, 1)
```
- gate 关的分支：只奖励“放大看”这个动作，不看位置，上限 0.25。这是 Design_New 冻结的设计，不是写错。

## 2. v1 实测（GRPO+A / A+B 训练账本，可判题 m=1，不含续答行）

| | gate 关的轨迹占比 | S 总量中来自 gate 关分支的份额 | gate 关分支的 S 均值 |
|---|---:|---:|---:|
| GRPO+A 第 1–2 步 | 18.1% | 1.1% | .006 |
| GRPO+A 第 25–32 步 | 4.5% | 0.6% | .042 |
| A+B 第 25–32 步 | 6.4% | 0.8% | .033 |

- gate 关分支内部，S 随裁图数上升：1 张 .014 → 6 张 .052。乘上 λ=0.1 后，奖励为 0.001–0.005。
- GRPO+A 后期 gate 开的轨迹：平均 2.95 张裁图，S 均值 .31。v1 中 S 均值随裁图数上升：1 张 .24 → 6 张 .38。

## 3. 导师 11-05 的方向（原话见 CONTROL）
- 不复刻 PAPO、DVRP 这类方法；先把机制做 work，optimizer 是可以迁移的。
- 他前面说的一项暂时不改。（本记录里没有对应的上下文，原话照录。）
- 改 explore：explore = 全部历史裁图的并集面积 / 原图面积；去掉 `s ≥ 2·s0` 的筛选。
- psi = (α + (1−α)·proximity) 若 gate，否则 α·explore；返回 psi、proximity、gate、explore。
- proximity：对每个目标 bbox，取“历史最大 IoU + 历史裁图并集对目标的覆盖率”，归一化到 [0, 1]。
- 用户补充：保留 S = psi_t − psi_0；分别记录 explore、proximity、gate；原图不算执行过的裁图，初始的裁图集合是空集。

## 4. v2 实现（`code/sg_scale_gated_v2/reward_sg.py`，从 grpo_sg 复制，只改这一个文件）
```
C        = 执行过的裁图（不含原图）；psi_0 时 C = ∅
prox_b   = 0.5 · ( max_{c∈C} IoU(b, c) + |b ∩ ∪C| / |b| )          ∈ [0, 1]
prox     = max_{S∈evidence_sets} mean_{b∈S} prox_b                  （聚合方式沿用 v1）
gate     = 与 v1 相同（任一关键框在某个视图中覆盖 ≥ 0.9 且尺度 h ≥ 0.5；视图含原图）
explore  = |∪C| / (W·H)
psi      = 0.25 + 0.75·prox  若 gate，否则 0.25·explore
S        = clip(psi_T − psi_0, 0, 1)；账本记录 prox、prox0、gate、gate0、explore、psi
```
- **两处由我沿用 v1、需要确认的选择**：
  1. 多个目标框的聚合方式：取 L2 evidence_sets 上“组内均值的最大值”；
  2. gate 的定义不变，仍包含原图视图。
- 自检 10/10 通过（`python reward_sg.py`）。

**相对 v1 的行为变化**（自检里打印，没有隐藏）：

| 情形 | v1 | v2 |
|---|---:|---:|
| gate 关时裁一张整图 | S = 0 | S = 0.25（explore = 1，满探索分） |
| 原图已看得清（gate0 = 1）时裁到目标上 | S = 0 | S = 0.75·prox（例中 .60） |

## 5. v1 vs v2 离线对比（同一批 GRPO+A rollout 的执行框重算，`psi_v1_v2_offline.py`）

| | 第 1–2 步（A-SFT 策略） | 第 25–32 步 |
|---|---:|---:|
| S 均值 v1 / v2 | .091 / .135 | .299 / .449 |
| S 按裁图数（1 → 6 张）v1 | .26 → .12 | .24 → .38 |
| S 按裁图数（1 → 6 张）v2 | .39 → .17 | .44 → .45 |
| 组内 AUC（S vs Y）v1 / v2 | .574 / .599 | .514 / .532 |
| v2 S 均值：答对 / 答错 | .147 / .110 | .459 / .420 |
| v2 中 gate 关分支的 S 份额 | 1.4% | 1.0% |
| gate0 = 1 的比例（原图已打开门） | 73.6% | 70.3% |
| 含近整图裁图（面积 ≥ 90%）的轨迹 | 0.07% | 4.05% |

- 这些 rollout 来自用 v1 训练的策略，离线重算不能代表 v2 训练后的策略行为。
