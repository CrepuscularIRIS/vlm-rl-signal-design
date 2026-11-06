# VLM 多轮裁图 RL：阶段性汇报（2026-11-05）

Qwen3-VL-4B 在 caption v5.1 协议下多轮裁图后作答选择题。本页汇总到目前为止**同一起点、同一题池、同一评测**下的全部 RL 训练，按三部分组织：① 设定 ② 定性 + 定量 ③ 训练 Log。

> 11-06 新增 **进入更新拆解（GRPO 底座，advantage 不动）**：**R：回放补满**（本步全对/全错组换成上一步的混合组，rollout 1×，时长 1.02×）E1 **1450/2208（65.67%）**、四 bench 不加权 **55.66**；**P：混合题追加一组**（rollout 1.56×，时长 1.29×）1438（65.13%）、55.44。GRPO 1411 / 53.92，DAPO 1465 / 56.50。
> - R − GRPO：合并 +1.77 [−0.23, +3.81]，不加权 +1.74 [−0.97, +4.36]；R − DAPO 合并 −0.68 [−2.72, +1.27]；P − R 合并 −0.54 [−2.45, +1.40]。
> - R、P 每次更新的混合组 126 / 124（GRPO 72，DAPO 128）；P 多采 56% 新轨迹未高于 R。R 的 dev（.583@32）低于 GRPO（.602）而 E1 更高。
> - [三部分报告、训练图、长表与进入规则代码](results/entry_ab_20261106/REPORT.zh.md) · [设计与预登记](results/entry_ab_20261106/DESIGN.zh.md) · [entry_patches.py](results/entry_ab_20261106/entry_patches.py)
>
> ![R/P E1](results/entry_ab_20261106/entry_e1.png)
> ![R/P training](results/entry_ab_20261106/entry_training.png)

## E1 按 benchmark 分开看（2026-11-06）

导师 11-05：“算平均分（评估的时候）可以不用加权”；“dapo只是一个optimizer……假设我们的optimizer用的是grpo（或者基于grpo的），那我们超过grpo就可以，对于dapo也是同理。”

所以本节**不再按题数加权合并四个来源**（原 E1 总分中 MME-RealWorld 占 1,356/2,208 = 61%），改为每个 benchmark 单独给分，再取四个 benchmark 的**不加权平均**；每个方法只与它自己的优化器底座比较（GRPO 底座的方法比 GRPO，DAPO 底座的方法比 DAPO）。行按底座分组，不按分数排序。

| 方法 | 类别 | 对照底座 | MME-RealWorld（339 题） | V\*（112） | WorldBench（62） | MMMU-Pro（39） | **四个 bench 不加权平均** | 平均相对底座 [95% 区间] |
|---|---|---|---:|---:|---:|---:|---:|---|
| A-SFT 起点 | 起点 | — | 57.1 | 81.2 | 27.8 | 28.8 | 48.75 | — |
| GRPO | 优化器 | — | 66.4 | 84.4 | 33.5 | 31.4 | 53.92 | — |
| **R：回放补满（新）** | GRPO 底座（进入更新） | GRPO | 67.8（+1.4） | 86.6（+2.2） | 38.7（+5.2） | 29.5（-1.9） | 55.66 | +1.74 [-0.97, +4.36] |
| **P：混合题追加一组（新）** | GRPO 底座（进入更新，1.56× rollout） | GRPO | 67.5（+1.0） | 86.4（+2.0） | 32.7（-0.8） | **35.3**（+3.8） | 55.44 | +1.52 [-1.51, +4.48] |
| DAPO | 优化器 | GRPO | **68.8**（+2.4） | **87.3**（+2.9） | 34.7（+1.2） | **35.3**（+3.8） | **56.50** | +2.58 [-0.33, +5.52] |
| GSPO | 优化器 | GRPO | 66.2（-0.2） | 83.5（-0.9） | 35.1（+1.6） | 34.0（+2.6） | 54.69 | +0.77 [-2.04, +3.68] |
| GiGPO | 优化器 | GRPO | 63.4（-3.0\*） | 83.0（-1.3） | 37.5（+4.0） | 29.5（-1.9） | 53.36 | -0.56 [-3.20, +2.00] |
| RLOO | 优化器 | GRPO | 65.6（-0.9） | 84.6（+0.2） | 30.2（-3.2） | 33.3（+1.9） | 53.43 | -0.49 [-3.39, +2.51] |
| SAPO | 优化器 | GRPO | 58.2（-8.3\*） | 81.2（-3.1） | 29.4（-4.0） | 30.8（-0.6） | 49.91 | -4.01 [-7.29, -0.84]\* |
| DTPO | GRPO 底座 | GRPO | 68.0（+1.5） | 86.4（+2.0） | 35.5（+2.0） | 32.1（+0.6） | 55.48 | +1.55 [-1.48, +4.57] |
| G1 | GRPO 底座 | GRPO | 66.4（+0.0） | 85.3（+0.9） | 32.7（-0.8） | 30.8（-0.6） | 53.79 | -0.14 [-3.06, +2.84] |
| A+B | GRPO 底座 | GRPO | 66.6（+0.1） | 83.7（-0.7） | **39.5**（+6.0） | 34.0（+2.6） | 55.95 | +2.02 [-0.84, +4.81] |
| A+B+C | GRPO 底座 | GRPO | 65.6（-0.8） | **87.3**（+2.9） | 33.9（+0.4） | 30.1（-1.3） | 54.23 | +0.30 [-2.46, +3.04] |
| GRPO+A | GRPO 底座 | GRPO | 66.9（+0.4） | 85.3（+0.9） | 33.5（+0.0） | 31.4（+0.0） | 54.26 | +0.33 [-2.49, +3.14] |
| strat | GRPO 底座 | GRPO | 66.5（+0.1） | 82.6（-1.8） | 31.9（-1.6） | 30.8（-0.6） | 52.93 | -0.99 [-3.93, +1.85] |
| max2 | GRPO 底座 | GRPO | 65.6（-0.9） | 85.5（+1.1） | 32.7（-0.8） | 32.7（+1.3） | 54.10 | +0.18 [-2.76, +3.04] |
| Hτ | DAPO 底座 | DAPO | 67.9（-0.9） | 85.7（-1.6） | 33.1（-1.6） | 32.1（-3.2） | 54.69 | -1.82 [-4.37, +0.70] |
| DAPO-SB | DAPO 底座 | DAPO | 68.0（-0.8） | 83.3（-4.0\*） | 29.0（-5.6） | 31.4（-3.8） | 52.92 | -3.58 [-6.60, -0.58]\* |
| RLOO + 首动作分叉 | RLOO 底座 | RLOO | 60.8（-4.7\*） | 81.9（-2.7） | 30.6（+0.4） | 25.0（-8.3） | 49.60 | -3.83 [-7.13, -0.71]\* |

- 单元格：准确率（%），括号内为相对“对照底座”的差（百分点）；`*` = 按题重采样 95% 区间不含 0。每列最高准确率加粗（不含 A-SFT 起点）。
- 区间：同一 (uid, sample_idx) 配对；每个来源内按题重采样（10,000 次，seed 20261105），不加权平均的差用同一组重采样。各来源题数不同：WorldBench 62 题、MMMU-Pro 39 题，单个 bench 的区间宽。
- 与旧口径的差别：按题数加权时 DAPO−GRPO 为 +2.45pp [+0.54, +4.44]（区间不含 0）；按四个 bench 不加权平均为 +2.58pp [−0.33, +5.52]（区间含 0）。
- GRPO 底座的方法里，A+B 的不加权平均 +2.02pp [−0.84, +4.81]，主要来自 WorldBench +6.0（该列最高 39.5）与 MMMU-Pro +2.6；DTPO +1.55pp，四个 bench 都为正；其余在 ±1pp 内。
- 所有数字由统一评分器从原始 E1 轨迹重算，总正确数与下文 §2.1 已发布值逐一核对一致。脚本：[e1_by_bench.py](results/e1_by_bench_20261106/e1_by_bench.py)（计算）、[make_table.py](results/e1_by_bench_20261106/make_table.py)（本表）、[plot_by_bench.py](results/e1_by_bench_20261106/plot_by_bench.py)（下图）；数据：[e1_by_bench.json](results/e1_by_bench_20261106/e1_by_bench.json)（仅聚合数）。

![E1 by benchmark vs own base](results/e1_by_bench_20261106/e1_by_bench_vs_base.png)

> 11-05 最新终点 **max2（GRPO + 终点式 S v2，psi = max(prox, 0.25·explore)）**：E1 **1404/2208（63.59%）**。
> - 与直接对照 **GRPO（1411）**：净差 −7 条，−0.32pp，按题 95% 区间 [−2.54, +1.90]，未分辨；裁图 3.54 vs 1.91 张/条，处理 token 14,727 vs 8,125/条（+81%）。差距按裁图数与来源的拆分见 §2.3。
> - 与 DAPO（1465）：−2.76pp [−4.98, −0.54]；与 GRPO+A、A+B、strat：区间均含 0。
> - 已并入下文 §1.4（公式与 [psi 参考实现](#max2grpo--psi-v2max-模式)）、§2.1–2.4 与 §三 的全部表图；[三部分报告](results/sgv2_max2_20261105/REPORT.zh.md) · [复算数据与指纹](results/sgv2_max2_20261105/FILES.json)。
> - 相对 GRPO+A 同时更换 psi 公式与索引口径（L1 3,077 题 vs L2 3,014 题），不是单变量消融。原训练退出码和逐 rank 更新事件仍缺。
>
> ![max2 与 GRPO 的差距拆分](results/sgv2_max2_20261105/max2_vs_grpo_gap.png)
> ![含 max2 的 E1 family](results/sgv2_max2_20261105/family_e1.png)

> 11-05 新增 **strat（同结果层中心化的过程 advantage）**：E1 **1399/2208，63.36%**；GRPO 63.90%、GRPO+A 64.36%、DAPO 66.35%。裁图2.473张/条，处理token10,792.76/条。已并入总表、分源、配对和family图；[三部分报告、两例逐图记录与复算数据](results/sgv3_20261105/REPORT.zh.md)。单训练seed；原训练独立退出码及每rank更新事件仍缺。

> 11-05 较早版本新增两组训练：DAPO-SB（DAPO + 停止相对 advantage）和 GRPO+A（GRPO + 只加 S 奖励），并入下文各表与全部曲线图。11-04 版见 [docs/README_stage_20261104.md](docs/README_stage_20261104.md)，更早的过程记录见 [docs/README_before_stage_20261104.md](docs/README_before_stage_20261104.md)。11-03 发布的“读取层 43%”诊断已撤回，见 [results/rereview_20261104](results/rereview_20261104/RESULTS.zh.md)。

## 阶段小结：结果与预估是否一致

| 预估（来源） | 观测 | 是否一致 |
|---|---|---|
| 尺度门控获取分 S 在训练中能被抬高（离线准入：无任务信号组 40.4% 有 S 方差） | S 从 0.11（第 1 步）升到 0.27（第 25–32 步均值）；门控开启率 0.86 → 0.93 | 一致 |
| 预登记风险：S 可以靠多裁图抬高（8 格平铺离线 ΔS +0.164） | 训练后期裁图数 2.18 张/条，GRPO 为 1.65；E1 裁图 2.40 张/条，GRPO 为 1.91 | 一致（风险兑现） |
| V\* 上答错时“覆盖了但目标太小”的情况会减少（尺度门的设计目的） | 答错且单裁覆盖 GT 的轨迹中，目标 <5% 裁图边长的条数：GRPO 14 → A+B 7；A+B+C 15 | A+B 一致，A+B+C 不一致 |
| 端到端门槛：总体 E1 ≥ +1pp（R-bind 草案沿用的预登记线） | A+B +0.82pp（配对净增 +18/2,208）；A+B+C +0.05pp（净增 +1） | 均未达到 |
| DAPO-SB 在 DAPO 上带来增益（Advantage.md 把“纯 DAPO 对比 DAPO-SB”定为第一对实验，“有收益后”再做根状态基线消融） | E1 .6413 vs DAPO .6635（−2.22pp，配对净增 DAPO +49/2,208）；四个分源都低于 DAPO，V\* −4.0、WorldBench −5.6 | 不一致 |
| 只加 A（S 奖励、标准 GRPO 归一）会改变取证偏好 | 训练后期裁图 3.39 张/条（GRPO 1.65）；E1 裁图 4.02 张/条，总处理 token 18,034（GRPO 8,125）；E1 +0.46pp（净增 +10） | 取证偏好一致；端到端未达 +1pp |
| 证据续答行（C）提升“看到之后”的读取与推理（A+B+C 对 A+B 不只差续答行：被选作续答的 384 题没有普通探索行，第 31–32 步题目构成也不同） | 续答行训练正确率约 0.8，同步原题行约 0.6；E1 上 A+B+C 对 A+B 净增 −17：V\* +16、WorldBench −14、MME −13 | 未观测到整体迁移；第 31–32 步题目构成有偏差（见 §1.6） |

Design_New 给出的定位是：A 改变的是取证偏好，不证明感知能力提高。分层探针（给 GT 证据图、控制尺度、单独统计选项匹配错误）还没有跑。

---

## L2 事实落到轨迹上的准入，以及给 Astra 的设计任务（2026-11-05）

- [results/l2fact_20261105/ADMISSION.zh.md](results/l2fact_20261105/ADMISSION.zh.md)：L2 成对事实比较（只覆盖数值表格：b=0 抽检 29/30、b=1 30/30；混合组里 P_i≠0 只有 2.3%）和轻量版（抽检 x=0 的样本，47% 是规则没识别的正确读数）。
- [prompts/ASTRA_ADVANTAGE_DESIGN_20261105.md](prompts/ASTRA_ADVANTAGE_DESIGN_20261105.md)：把过程信号放进 advantage 而不是 reward 的设计任务，汇总了设置、导师构造、全部结果、审计事实、标注使用情况和审查过的轨迹。

## 组内对比诊断与 psi v2（2026-11-05）

- **D0 组内对比**（[results/diag_d0_20261105/D0_REPORT.zh.md](results/diag_d0_20261105/D0_REPORT.zh.md)）：
  - DAPO 实际更新的组里，22.7% 来自 L0 标为不可答（主要是 needs_temporal）或 GT 有问题的题，题池里这类题占 19.1%；
  - 可判题的混合组配对中，取证差异占 14.8%（纯几何口径 23.6%），原图已足够占 34.2%；
  - E1 V\* 上，RL 之后组内差异以“取证之后”为主（GRPO 51%、DAPO 53%）；
  - DAPO 丢掉的全错组中，76% 取到过证据。
- **psi 审计与 v2**（[results/diag_d0_20261105/AUDIT_PSI.zh.md](results/diag_d0_20261105/AUDIT_PSI.zh.md)，代码 [code/sg_scale_gated_v2](code/methods/sg/v2/reward_sg.py)）：
  - v1 中 gate 关的“探索分”只占 S 总量约 1%；
  - 按导师规格改写为 proximity（最大 IoU 与裁图并集覆盖的均值）和 explore（全部历史裁图的并集面积），保留 S = psi_t − psi_0；
  - 离线重算：S 不再随裁图数单调上升。

## 学习信号审计（2026-11-05）

把全部方法按“进入更新的数据 / 奖励 / 组内 advantage / 信用分配”四列对齐，并从训练账本复算每个方法实际送进更新的组和 advantage，见 [results/audit_signal_20261105/AUDIT_SIGNAL.zh.md](results/audit_signal_20261105/AUDIT_SIGNAL.zh.md)。只列事实：

- GRPO 每步 128 组中平均 72.2 组是混合组，其余 A=0；DAPO 每步 128 组都是混合组，为此每次更新平均生成 7.7 轮 × 32 组，丢掉约 80 个全对组、30 个全错组。
- 两者混合组的难度分布（k=1…7 的占比）几乎相同；DAPO 混合组总数是 GRPO 的 1.77 倍，并且对 1,447 道题更新了两次（k 均值 3.81 → 4.16）。
- 正 / 负 advantage 质量比（token 加权）：所有减组均值的方法在 0.91–0.97 之间，DAPO-SB 为 2.23。
- E1 按题重采样的 95% 区间：只有 DAPO 相对 GRPO 的 +2.45 [+0.54, +4.35] 不含 0；相对 DAPO，DTPO、Hτ、A+B、GRPO+A 的区间含 0，其余在 0 以下。
- DAPO 相对 GRPO 的净增按 A-SFT 难度分档（0/4…4/4）：+17、+5、+2、+15、+15。

![update composition](results/audit_signal_20261105/figs/a1_update_composition.png)

---

## 一、设定

### 1.1 SFT

- **A-SFT**（所有 RL 的起点）：4,040 条 caption 轨迹，turnwise 格式，mask_history；E1 .5670。**本阶段无 SFT 语料改动。**
- 因果叙事格式的新 SFT（B-SFT causal）：同样 4,040 条，改写成因果叙事、不做 mask_history。E1 只有 .4543；在其上再做 GRPO 得 .6033，仍低于 A-SFT 起点的 GRPO（.6390）。该线已停。

### 1.2 协议（caption v5.1，训练与评测相同）

- 每轮输出一个 `<think>`（首行 `Step N | WORD`，WORD ∈ locate / re_examine / fallback / commit / abstain），后接**恰好一个**动作：
  - `<grounding>{"bbox_2d":[x1,y1,x2,y2],"source":"VIEW_ID"}</grounding>`：坐标是所选源视图上的 [0,1] 相对值；
  - 或 `<answer>X</answer>`。
- 每题最多 8 次裁图，外加 1 次最终作答回合。裁图从全分辨率原图重新切；任一边小于 56 px 时，绕中心扩到 56 px。
- 呈现尺寸规则：
  - 短边小于 448 时放大到 448，放大倍数最多 4×；
  - 面积超过 1400² 时等比缩小；
  - 边长对齐到 28 的倍数；
  - 之后再经 `max_pixels = 589,824` 缩放。
- **奖励 V·Y**：
  - V = 1 当且仅当轨迹以合法 answer 结束（无硬协议错误、未截断）；
  - Y = 1 当且仅当输出恰好一个 `<answer>X</answer>` 且 X 等于 gt；
  - 奖励放在最后一个 assistant token 上；回执和图像 token 的 loss mask 为 0。
- 模型的思考文本里**不写坐标**。下文凡是用到 bbox 的地方，都只在**奖励计算一侧**使用 L1 标注框。

### 1.3 公共 RL 设置（所有 run 以此为准，差异在 §1.4 逐条列出）

| 项 | 值 |
|---|---|
| 起点 | A-SFT（Qwen3-VL-4B） |
| 题池 | rl4k：4,096 道题，原图互不相同。进入梯度更新的始终是 32 步 × 128 组 = 4,096 组。无过滤的方法（GRPO、GSPO、GiGPO、RLOO、DTPO、G1、A+B、A+B+C、GRPO+A）每题只采一遍；**DAPO、Hτ、DAPO-SB 带过滤和补采**，采样量不同，见下表 |
| rollout | 每题 n = 8；temperature 0.7；top_p 1.0 |
| 更新 | 每步 128 题，actor mini-batch 32 题（每步 4 次更新）；ppo_epochs 1 |
| 优化 | lr 1e-6，constant，warmup ratio 0.05；grad norm 1.0；无 KL |
| clip | low 0.2 / high 0.3（dual clip 3.0）；token 平均 |
| seed | 42；RandomSampler 题序（各 run 第 1 步的题目集合相同） |
| 评测 E1 | 552 题 × 4 次采样 = 2,208；step 32 终点权重；统一评分器；截断、错误、无答案都保留在分母里 |

**采样预算并不相等**：所有方法的梯度更新次数相同（32 步 × 128 组），但带过滤的方法生成的 rollout 更多。

| | GRPO | DAPO | DAPO-SB |
|---|---:|---:|---:|
| 实际生成的组（每组 8 条） | 4,096 | 7,872 | 7,968 |
| rollout 条数 | 32,768 | 62,976 | 63,744 |
| 题池遍历 | 每题 1 遍 | 4,096 题全部采过，其中 3,776 题采了 2 遍 | 4,096 题全部采过，其中 3,872 题采了 2 遍 |
| 进入更新的不同题 / 被更新 ≥2 次的题 | 4,096 / 0 | 2,648 / 1,447 | 2,650 / 1,446 |
| 累计处理 token | 2.11 亿 | 4.79 亿（2.27 倍） | 4.00 亿（不含探针前向） |
| 训练墙钟 | 6.8 h | 11.2 h | 13.25 h（与 GRPO+A 同机并行；探针每个不同状态 2 次单 token 前向，共 ≥182,334 次） |

DAPO 累计生成到 4,096 组是在第 18 步，该步的权重没有保留。Hτ、DAPO-SB 用同样的过滤，同样多采了样。因此 DAPO / Hτ / DAPO-SB 与其他方法的比较属于**等更新次数、不等采样量**；DAPO-SB 与 DAPO 之间采样量接近。

### 1.4 各方法：公式与取值

记号：每题 $G=8$ 条 rollout，组记作 $g$。第 $i$ 条轨迹的奖励为 $r_i$；$V_i, Y_i\in\{0,1\}$ 的定义见 §1.2；$N_i$ 为裁图次数；$\rho_{i,t}=\pi_\theta/\pi_{\text{old}}$ 为 token 级重要性比。

**参数一览**（未列出的项都与 §1.3 相同；各方法的源码见 [code/](code/README.md)）

| 方法 | advantage | 奖励 | clip (low / high) | 过滤 | 其他差异 |
|---|---|---|---|---|---|
| GRPO | 组内 z-score | V·Y | 0.2 / 0.3 | 无 | — |
| DAPO | 组内 z-score | V·Y | 0.2 / 0.28 | 组均值 ∈ (0.01, 0.99)，补采上限 100 | update micro 1 |
| GSPO | 组内 z-score | V·Y | 3e-4 / 4e-4（序列级） | 无 | 序列内先平均 |
| GiGPO | episode + 1.0 × step | V·Y | 0.2 / 0.3 | 无 | 锚点分组 |
| RLOO | 留一基线 | Y(1 − 0.01N) | 0.2 / 0.3 | 无 | 训练协议 v5.2i，warmup 1 步 |
| RLOO + 首动作分叉 | 同 RLOO | 同 RLOO | 0.2 / 0.3 | 无 | β = 1/256，每边 K = 4 分叉 |
| DTPO | 组内 z-score | V·Y | 0.2 / 0.24 | 无 | 中间段权重 0.5 |
| SAPO | turn 级 GAE（γ = λ = 0.95） | Y | 0.2 / 0.2 | 无 | n = 1；V/Q 损失系数各 0.1 |
| Hτ | 组内 z-score | V·(Y + 0.25·m·Φ_τ) | 0.2 / 0.28 | 同 DAPO（filter_key = accuracy） | τ = 0.005 |
| G1 | 组内 z-score | V·(Y + 0.25·m·Φ_read) | 0.2 / 0.3 | 无 | m = 1 的题共 2,699 道 |
| A+B | 任务尺度归一 | V·(Y + 0.1·m·S) | 0.2 / 0.3 | 无 | m = 1 的题共 3,014 道 |
| A+B+C | 任务尺度归一 | 同 A+B；续答行 V·Y | 0.2 / 0.3 | 无 | 384 行续答行 |
| GRPO+A | 组内 z-score | V·(Y + 0.1·m·S) | 0.2 / 0.3 | 无 | S 与 A+B 相同 |
| DAPO-SB | 停止相对：(Y − b_k)/σ_q，λ = 1 | V·Y | 0.2 / 0.28 | 同 DAPO | 每回合前的停止探针 b_k；update micro 1 |
| strat | GRPO + 0.1·同结果层中心化 S | V·Y | 0.2 / 0.3 | 无 | psi v2；32步，n=8，seed42 |
| **max2（新）** | 组内 z-score | V·(Y + 0.1·m·S)，S 用 psi v2（max 模式） | 0.2 / 0.3 | 无 | L1 索引，m = 1 的题共 3,077 道 |

strat：$A_i=(r_i-\bar r_g)/(\operatorname{std}_{sample,g}(r)+10^{-6})+0.1(S_i-\operatorname{mean}_{j\in H_i}S_j)$，$H_i$ 为同题、同二值训练结果 $r=VY$、可评分的轨迹；不可评分或层内少于两条时第二项为0。第二项只中心化、不再除标准差，广播到全部有效response token。S=clip(psi_T−psi_0,0,1)，psi=max(prox,.25·explore)，prox为最大IoU与目标并集覆盖均值，explore为全部历史裁图并集面积比。原图不算crop。与GRPO只改advantage；GRPO+A使用psi v1且改奖励，二者不是单变量消融。[实现快照](results/sgv3_20261105/stratified_adv.py)与[完整设定](results/sgv3_20261105/REPORT.zh.md)。

#### GRPO（对照）

$$
A_i=\frac{r_i-\operatorname{mean}_{g}(r)}{\operatorname{std}_{g}(r)+10^{-6}},\qquad r_i=V_iY_i
$$

$$
\mathcal{L}=-\frac{1}{T}\sum_{i,t}\min\Big(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t},\,1-0.2,\,1+0.3)\,A_i\Big)
$$

$A_i$ 铺到该轨迹的全部 assistant token；8 条全对或全错时 $A_i=0$。$A<0$ 时另有 dual clip $3.0$。

#### DAPO

advantage 与 GRPO 相同，差别在进入更新的题组：

$$
\text{keep group } g \iff 0.01<\operatorname{mean}_g(Y)<0.99
$$

保留的组不足 128 个时继续补采（最多 100 次）。本 run 共生成 7,872 个题组，进入更新 4,096 组。clip high 为 0.28。

#### GSPO

$$
s_i=\frac{1}{\lvert y_i\rvert}\sum_t\big(\log\pi_\theta(y_{i,t})-\log\pi_{\text{old}}(y_{i,t})\big),\qquad
\mathcal{L}_i=-\min\Big(e^{s_i}A_i,\ \operatorname{clip}(e^{s_i},\,1-3\times10^{-4},\,1+4\times10^{-4})\,A_i\Big)
$$

同一序列的所有 token 共用 $e^{s_i}$；loss 先在序列内平均，再跨序列平均。

#### GiGPO

$$
A_{i,k}=A^{\text{episode}}_i+1.0\cdot\frac{r_i-\operatorname{mean}_{a}(r)}{\operatorname{std}_{a}(r)+10^{-6}}
$$

$a$ 是第 $k$ 个回合的锚点组。同一题中“已收视图、已执行裁图框、已用裁图数、步序号”都相同的回合属于同一个锚点组；单成员组的 step 项为 0。step 项只加在该回合的生成 token 上。

#### RLOO 与 RLOO + 首动作分叉

$$
A_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j,\qquad r_i=Y_i\,(1-0.01\,N_i)
$$

分叉监督：在首个动作之前的前缀 $h$ 处，强制 answer 续写 4 条、强制 crop 续写 4 条，各自续到终止。

$$
D=\operatorname{sg}\big(\bar R_{\text{answer}}-\bar R_{\text{crop}}\big),\qquad
z=\log\pi(\text{answer tag}\mid h)-\log\pi(\text{grounding tag}\mid h)
$$

$$
\ell=\operatorname{relu}(D)\,\operatorname{softplus}(-z)+\operatorname{relu}(-D)\,\operatorname{softplus}(z),\qquad
\mathcal{L}=\mathcal{L}_{\text{RLOO}}+\frac{\beta}{H_0}\sum_q\ell_q,\quad \beta=\tfrac{1}{256},\ H_0=32
$$

#### DTPO

advantage 同 GRPO。把非最终回合的 assistant token 合并为中间段 $I$，最终回合为 $F$：

$$
\mathcal{L}=0.5\cdot\frac{1}{\lvert I\rvert}\sum_{t\in I}\ell_t+\frac{1}{\lvert F\rvert}\sum_{t\in F}\ell_t,\qquad
\ell_t=-\min\big(\rho_tA,\ \operatorname{clip}(\rho_t,0.8,1.24)\,A\big)
$$

#### SAPO

$n=1$。用共享 LM head 读出每轮的 $V$（动作前）和 $Q$（本轮末），不新增 value head。

$$
\delta_k=r_k+\gamma\,c_k\,V_{k+1}-V_k,\qquad \hat A_k=\delta_k+\gamma\lambda\,c_k\,\hat A_{k+1},\qquad \gamma=\lambda=0.95
$$

$\hat A_k$ 在有效 turn 之间做方差归一，再广播到该轮生成 token。总损失为 PPO 损失（clip 0.2 / 0.2）加 $0.1\,\mathcal{L}_V+0.1\,\mathcal{L}_Q$。

#### Hτ：bbox 覆盖过程奖励（DAPO 基座）

$$
r_i=V_i\Big(Y_i+0.25\,m_q\,\Phi_\tau\Big),\qquad
\Phi_\tau=\frac{1}{J}\sum_{j=1}^{J}\min\Big(1,\ \tfrac12\big(\max_{v\in\mathcal{C}_j}\operatorname{IoU}(b_j,v)+\operatorname{cover}(b_j,\textstyle\bigcup\mathcal{C}_j)\big)\Big)
$$

$b_j$ 为第 $j$ 个 L1 证据框。$\mathcal{C}_j=\{v:\lvert b_j\cap v\rvert/\lvert v\rvert\ge\tau\}$，$\tau=0.005$，即只计入目标框在裁图中占比足够的裁图。原图不计入。

#### G1：读值一致性过程奖励（GRPO 基座）

$$
r_i=V_i\Big(Y_i+0.25\,m_q\,\Phi_{\text{read}}\Big),\qquad
\Phi_{\text{read}}=\frac{1}{K_q}\sum_{k=1}^{K_q}\mathbb{1}\big[\text{作答文本中字段 }k\text{ 的读数}=\text{L2 值}_k\big]
$$

作答文本 = 最后一次 `</grounding>` 之后的文本。读数经规范化后要求精确相等；枚举多个值、否定、并列都记 0；每个字段只算一次。

#### A+B：尺度门控证据奖励 + 任务尺度归一的 advantage（GRPO 基座）

**A（奖励）**

$$
r_i=V_i\big(Y_i+\lambda\,m_q\,S_i\big),\qquad \lambda=0.1
$$

对每个 L1 证据框 $b$ 和模型看到的每个视图 $v$（含原图 $v_0$）：

$$
c(b,v)=\frac{\lvert b\cap v\rvert}{\lvert b\rvert},\qquad
p_b(v)=d_b\cdot s(v),\qquad
a_b(v)=\mathbb{1}\big[c(b,v)\ge0.9\big]\cdot h\big(p_b(v)\big)
$$

- $d_b$：细节尺寸。文字取框高 / 行数，物体取框的短边。
- $s(v)$：原图像素到最终模型输入像素的倍率（依次经过呈现规则和 `max_pixels`）。
- $h$：线性上升后饱和于 1，文字 $6\to14$ px，物体 $16\to48$ px（**未校准**）。

$$
z_b=\max_{v}\,a_b(v),\qquad
\Phi=\max_{\mathcal{S}\in\mathcal{E}_q}\ \frac{1}{\lvert\mathcal{S}\rvert}\sum_{b\in\mathcal{S}}z_b
$$

$\mathcal{E}_q$ 为 L2 的 evidence_sets（合法的充分证据组合）。

$$
\Psi=\begin{cases}0.25\,E' & g=0\\[2pt] 0.25+0.75\,\Phi & g=1\end{cases},\qquad
S_i=\Psi_T-\Psi_0\in[0,1]
$$

- 门 $g=1$ 的条件：任一关键框（cell / answer_text / value_label / instance / target / data_mark / subject）满足 $a_b\ge0.5$。
- $E'$：放大倍率 $\ge2$ 倍原图倍率的裁图，其并集面积占全图的比例（探索分，上限 0.25）。
- $\Psi_0$ 只用原图计算，所以原图上已能看清的题 $S=0$。

**B（advantage）**

$$
y_i=\mathbb{1}[r_i\ge1],\qquad s'_i=r_i-y_i,\qquad
A_i=\frac{(y_i-\bar y)+(s'_i-\bar s')}{\max\big(\operatorname{std}(y),\ 1/\sqrt{G}\big)}
$$

因为 $\lambda S\le0.1<1$，$r$ 可以精确拆成 $y$ 和 $s'$。全错或全对组中 $\lvert A_i\rvert\le\lambda\sqrt{G}=0.28$；混合组里，过程项不会翻转任务项的符号（$\lambda<1/(G-1)$）。

**资格 $m_q=1$**：可回答、无 GT 分歧、L1 不是 absent / region_global、L1 置信度不为 low、至少一组 evidence_set 成员齐全且没有 `readable=false`。共 3,014/4,096 题。

**离线准入**（A-SFT 真实轨迹 256 题 × 8）：无任务信号组中 S 有方差的占 40.4%（准入线 10%）；追加一张整图裁剪 ΔS = 0；8 格平铺 ΔS +0.164。

#### A+B+C：再加证据续答行

- 在 A+B 基础上，把 384 行训练题换成续答行：第 1–8 步每步 32 行，第 9–16 步每步 16 行，之后为 0。每道题仍只出现一次。
- 续答行把该题第一组 evidence_set 的并集框（每边外扩 10%）作为合成的第 1 步 locate 回合注入 prompt；这部分是 prompt token，不进 loss。若并集框中有框达不到可用尺度，就改为每个框一个视图（最多 3 个）。模型从第 2 步接着答。
- 续答行只拿 $r=V\cdot Y$，注入的前缀不得过程分；它们与普通行同在一个 batch，但各自按题组计算 advantage。

#### GRPO+A：只加 A（GRPO 基座）

$$
r_i=V_i\big(Y_i+0.1\,m_q\,S_i\big),\qquad
A_i=\frac{r_i-\operatorname{mean}_{g}(r)}{\operatorname{std}_{g}(r)+10^{-6}}
$$

S、资格 $m_q$ 的定义与 A+B 完全相同；advantage 为标准 GRPO（不用 B 的任务尺度归一）。对 GRPO 的唯一变量是奖励中的 $0.1\,m_q S_i$；对 A+B 的唯一变量是 advantage 归一方式。

#### max2：GRPO + psi v2（max 模式）

$$
r_i=V_i\big(Y_i+0.1\,m_q\,S_i\big),\qquad
A_i=\frac{r_i-\operatorname{mean}_{g}(r)}{\operatorname{std}_{g}(r)+10^{-6}}
$$

$$
\operatorname{prox}_b=\tfrac12\Big(\max_{c\in C}\operatorname{IoU}(b,c)+\frac{\lvert b\cap\bigcup C\rvert}{\lvert b\rvert}\Big),\quad
\operatorname{prox}=\max_{\mathcal S\in\mathcal E_q}\frac{1}{\lvert\mathcal S\rvert}\sum_{b\in\mathcal S}\operatorname{prox}_b,\quad
\operatorname{explore}=\frac{\lvert\bigcup C\rvert}{WH}
$$

$$
\psi=\max\big(\operatorname{prox},\ 0.25\cdot\operatorname{explore}\big),\qquad S_i=\operatorname{clip}(\psi_T-\psi_0,\,0,\,1)
$$

- $C$ 为已执行的裁图框（原图不算裁图），$\psi_0$ 只有原图，因此 $\psi_0=0$；$b$ 为 L1 证据框。max2 用 `--sets l1` 索引：每题 $\mathcal E_q$ 只有一组，即该题全部 L1 框（3,077 道可判题均如此，已从索引核对），所以 prox 就是全部 L1 框 prox_b 的平均。
- 与 GRPO+A 的差别：psi 公式（v1 的尺度门控 → v2 的逼近/探索上包络）和索引（L2 evidence_sets 3,014 题 → L1 3,077 题）同时改变。
- 训练代码：[code/methods/sg/v2/reward_sg.py](code/methods/sg/v2/reward_sg.py)（`CAPTION_SG_PSI=max`，启动器 `grpo_sgv2.sh` 默认值）。下面是只保留 max 模式的参考实现 [results/sgv2_max2_20261105/psi_v2.py](results/sgv2_max2_20261105/psi_v2.py)；`python psi_v2.py` 在 2,000 个随机样例上与训练代码逐数值对照（输出 `2000/2000 match`）。

```python
Box = tuple[float, float, float, float]
ALPHA = 0.25


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a: Box, b: Box) -> Box | None:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[2] > box[0] and box[3] > box[1] else None


def union_area(boxes: Sequence[Box]) -> float:
    """矩形并集的精确面积：按 x 坐标切条，条内合并 y 区间。"""
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        covered = top = 0.0
        for lo, hi in spans:
            covered += max(0.0, hi - max(lo, top))
            top = max(top, hi)
        total += (right - left) * covered
    return total


def iou(a: Box, b: Box) -> float:
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    shared = area(inter)
    return shared / (area(a) + area(b) - shared)


def prox_one(target: Box, crops: Sequence[Box]) -> float:
    """单个目标框：最大 IoU 与被裁图并集覆盖比例的均值。"""
    if not crops:
        return 0.0
    pieces = [p for p in (intersect(target, c) for c in crops) if p]
    cover = min(1.0, union_area(pieces) / area(target)) if pieces and area(target) > 0 else 0.0
    return 0.5 * (max(iou(target, c) for c in crops) + cover)


def psi(targets: Mapping[str, Box], evidence_sets: Sequence[Sequence[str]],
        crops: Sequence[Box], image_wh: tuple[float, float]) -> float:
    """psi = max(prox, ALPHA × explore)。"""
    prox = {key: prox_one(box, crops) for key, box in targets.items()}
    approach = max(sum(prox[k] for k in s) / len(s) for s in evidence_sets)
    width, height = image_wh
    explore = min(1.0, union_area(crops) / (width * height)) if crops else 0.0
    return max(approach, ALPHA * explore)


def process_score(targets: Mapping[str, Box], evidence_sets: Sequence[Sequence[str]],
                  crops: Sequence[Box], image_wh: tuple[float, float]) -> float:
    """S = clip(psi_T − psi_0, 0, 1)。"""
    gain = psi(targets, evidence_sets, crops, image_wh) - psi(targets, evidence_sets, [], image_wh)
    return min(1.0, max(0.0, gain))
```

#### DAPO-SB：停止相对 advantage（DAPO 基座）

**停止探针**。第 $i$ 条轨迹在第 $k$ 个 assistant 回合之前的状态 $h_{i,k}$ = 题面与选项 + 该回合前实际呈现过的图像（原图 + 前 $k$ 张裁图），**不含模型自己的文本**。用 rollout 引擎的当前权重 $\theta_{\text{old}}$ 一次前向、不采样，强制作答：

$$
b_{i,k}=P_{\theta_{\text{old}}}(\text{gt 字母}\mid h_{i,k}),\qquad
P(X)=P(\texttt{>X}\mid\ldots\texttt{<answer})+P(\texttt{>}\mid\ldots\texttt{<answer})\cdot P(X\mid\ldots\texttt{<answer>})
$$

训练文本中 `<answer>D` 被切成 `'<' 'answer' '>D'`，所以探针停在 `<answer` 读 `>A`…`>J`，单独的 `>` 再用第二次请求（以 `<answer>` 结尾）读 `A`…`J`。top-20 logprob；system prompt 为 caption v5.1 的 SYSTEM，末尾加 “Answer with the option's letter from the given choices directly.”。GT 只在 driver 一侧用来取 $b=P(\text{gt})$。

**advantage**（Pro 参考实现 `stop_relative_advantage.py`，原样使用）：

$$
\delta_{i,k}=b_{i,k+1}-b_{i,k}\ (k<K_i),\qquad \delta_{i,K_i}=Y_i-b_{i,K_i},\qquad
\hat A_{i,k}=\sum_{j\ge k}\lambda^{\,j-k}\delta_{i,j}
$$

$$
A_{i,t}=\frac{\hat A_{i,k(t)}}{\sigma_q+\epsilon},\qquad \sigma_q=\operatorname{std}_g(Y)
$$

$\lambda=1$ 时 $\hat A_{i,k}=Y_i-b_{i,k}$：第 $k$ 回合的 token 用“该回合之前强制作答的成功率”作基线，代替 DAPO 的组均值 $\bar Y_q$。$A_{i,t}$ 铺到第 $k(t)$ 回合的生成 token；回执和图像 token 为 0。题组仍由 DAPO 的过滤决定（只有混合组进入更新）。

**探针冒烟**（A-SFT，40 道训练题，只看原图）：P(gt) 均值 .371，取最大概率选项的正确率 .525，字母概率总量均值 .778（最低 .293）；缺失的质量流向 `>` 之后的 `UNC` 或数字。Advantage.md §七要求的三层离线验收（探针扰动与真实停止对照、同状态分叉的四种信用方向、固定根状态基线的同批梯度对照）**没有做**。

### 1.5 rl4k 标注

**思路**：对 RL 题池的每道题逐级回答三个问题。
- L0：这题能不能答、gt 对不对。
- L1：答题要看图里哪里。
- L2：那里写的或画的是什么，以及它们怎样组合成答案。

做不到常规框的题，另走线索层 LC。原则：逐题看图，不参考模型轨迹，不改 gt。

| 层 | 内容 | 规模 |
|---|---|---|
| L0 定性 | 标注单位（single_box / keyed_set / multi_set / pair_relation / region_global / absent / unanswerable …）、可回答性、gt 核对（agree 3,369 / cannot_tell 575 / disagree 152）、task_tags | 4,096 |
| L1 证据框 | 带 id 和 role 的框（cell 1,794、answer_text 1,433、row_header 1,279、col_header 1,121、instance 918 …），原图归一化 [x1,y1,x2,y2]；负候选框；`target_too_small`（主目标短边 < 56 px）；置信度 | 3,559 题，10,142 个框 |
| L2 事实与绑定 | 每个框的读值（text / unit / readable）；绑定元组（table 1,685、entity 671、chart 429、compare 220、count 200、relation 135、chain 13）；**evidence_sets**（充分证据组合，3,212 题只有 1 组）；区分条件（为什么负候选不对） | 3,255 题 |
| LC 线索层 | 没有常规证据框的题：可见线索 → 可推测 → 未解决。类别：needs_temporal 406、disagree 152、ambiguous 38、external 36、not_found 31、low_res 26 | 681 题 |

- relation 绑定中 `computable_from_boxes = true` 的有 48 条：表示该关系是图像坐标系下的左右、上下，能直接由两框坐标算出。11-04 已把它们规范成 L/R/U/D：看图核对 18 条，其中 2 条标注方向错误，5 条不是方向关系。
- 流程：
  - gpt-6-astra 两轮标注：先出草稿，再叠框并对照原图裁图核对后写 L2；校验报错时回灌重做，最多 2 次。
  - Claude Opus 5.5 审查 1,857 题：pass 1,499、小改 90、大改 268。
  - needs_external_knowledge 的 215 题重标。
  - 全量校验 0 error。

### 1.6 hint 的构造与使用

- **13 类 hint**：count、table、chart、map、instance、text、relation、color、search、aerial、screen、compute、puzzle。每类是一句操作指引，例如 table 类提示先确认行键和列路径再读单元格。原文见 [protocol/CAPTION_V51.md](protocol/CAPTION_V51.md)。
- **生成规则 `task_hints`**：按题目元数据（material、category）、题面正则关键词、图像长边 ≥ 3000（触发 search）依次判断，每题最多 2 条，**不使用 gold**。
- **在哪里出现**：
  - 教师生成 SFT 轨迹时，按题在首轮 `Question:` 后加 `Hint: …`；
  - 导出给学生的 SFT 数据去掉了 hint；
  - RL 题池和 E1 的 prompt 中都**没有** hint。学生模型在 SFT、RL、评测中都没见过 hint 原文。
- **标注阶段怎么用**：
  - 规则 `task_hints` 用于生成单位草案；
  - L0 的 task_tags 沿用 13 类名称，但按看图结果选，不按题面关键词选；
  - 外部知识题重标时，以 compute hint 原文作为“该找什么证据”的指引。
- 本阶段的 A / B / C 都没有把 hint 放进 prompt。C 的续答行也只注入证据视图，不注入 hint 或答案事实。

### 1.7 本阶段的记录事项

- A+B+C 第 31–32 步的原题集合与其他 run 不同：构造续答行时从题序尾部挑题，跳过的不合格题积在了最后两步。m = 0 的题分别占 49/128 和 109/128；GRPO 在同一批题上的正确率为 .566 / .318，A+B+C 为 .560 / .311。
- 11-03 的 oracle 探针计分有误（正则把冠词 “A” 当成选项），依赖它的“读取层 43%”结论已撤回。
- DAPO-SB 与 GRPO+A 同机并行（GPU 2,3 / 0,1），cgroup 内存最高 347G，两者都是 32/32 步、0 Traceback。DAPO-SB 的 mm cache 设置与 k8 DAPO 不同（只影响速度与内存）。
- DAPO-SB 的第 1 批与 k8 DAPO 逐项相同（score .482、entropy .448；训练账本正确率 .571、裁图 .677、生成 token 134.3），第 1 步两者 lr 都是 0（预热）。
- 11-05 初稿的 DAPO-SB 报告把“保留组回合数 2.81”与 E1 裁图数并列为未弄清的问题，口径不同，已更正：同口径（全部生成的 rollout）训练裁图第 32 步 DAPO-SB 1.57、DAPO 1.99。

---

## 二、定性 + 定量

### 2.1 E1 总分与成本（step 32；行按底座分组，不按分数排序）

| 方法 | 类别 | **四个 bench 不加权平均** | 合并（按题数加权，旧口径，仅参考） | 正确 / 2,208 | 裁图 / 条 | 生成 token / 条 | 总处理 token / 条 |
|---|---|---:|---:|---:|---:|---:|---:|
| A-SFT 起点（无 RL） | 起点 | 48.75 | .5670 | — | 0.994 | — | 5,015 |
| GRPO | 优化器 | 53.92 | .6390 | 1411 | 1.905 | 261 | 8,125 |
| **R：回放补满（新）** | GRPO 底座（进入更新，1× rollout） | 55.66 | .6567 | 1450 | 2.161 | 264 | 9,100 |
| **P：混合题追加一组（新）** | GRPO 底座（进入更新，1.56× rollout） | 55.44 | .6513 | 1438 | 1.927 | 248 | 8,136 |
| DAPO | 优化器 | **56.50** | .6635 | 1465 | 2.092 | 287 | 8,930 |
| GSPO | 优化器 | 54.69 | .6395 | 1412 | 2.263 | 278 | 9,756 |
| GiGPO | 优化器 | 53.36 | .6209 | 1371 | 1.998 | 275 | 8,506 |
| RLOO | 优化器 | 53.43 | .6318 | 1395 | 1.992 | 278 | 8,535 |
| SAPO | 优化器 | 49.91 | .5770 | 1274 | 1.063 | 171 | 5,292 |
| DTPO | GRPO 底座 | 55.48 | .6553 | 1447 | 3.357 | 395 | 15,579 |
| G1（读值一致性） | GRPO 底座 | 53.79 | .6395 | 1412 | 2.308 | 374 | 10,063 |
| A+B | GRPO 底座 | 55.95 | .6472 | 1429 | 2.404 | 342 | 10,047 |
| A+B+C | GRPO 底座 | 54.23 | .6395 | 1412 | 2.337 | 326 | 10,181 |
| GRPO+A | GRPO 底座 | 54.26 | .6436 | 1421 | 4.020 | 440 | 18,034 |
| strat | GRPO 底座 | 52.93 | .6336 | 1399 | 2.473 | 321 | 10,793 |
| max2（新） | GRPO 底座 | 54.10 | .6359 | 1404 | 3.540 | 431 | 14,727 |
| Hτ（bbox 覆盖奖励） | DAPO 底座 | 54.69 | .6508 | 1437 | 2.546 | 305 | 10,872 |
| DAPO-SB | DAPO 底座 | 52.92 | .6413 | 1416 | 1.631 | 222 | 7,055 |
| RLOO + 首动作分叉 | RLOO 底座 | 49.60 | .5919 | 1307 | 1.372 | 203 | 6,266 |

标注：粗体 = 不加权平均列的最高值；合并分数按题数加权（MME-RealWorld 占 61%），只作旧口径参考，不用于排序；成本列不标最优；“（新）”= 最近新增条件。分 benchmark 的分数与相对底座差见上方“E1 按 benchmark 分开看”。

### 2.2 分源

| 方法 | MME-RealWorld (1,356) | V\* (448) | WorldBench (248) | MMMU-Pro (156) |
|---|---:|---:|---:|---:|
| GRPO | .6645 | .8438 | .3347 | .3141 |
| **max2（新）** | .6556 | .8549 | .3266 | .3269 |
| strat | .6652 | .8259 | .3185 | .3077 |
| DAPO | **.6881** | **.8728** | .3468 | **.3526** |
| A+B | .6659 | .8371 | **.3952** | .3397 |
| A+B+C | .6563 | **.8728** | .3387 | .3013 |
| GRPO+A | .6689 | .8527 | .3347 | .3141 |
| DAPO-SB | .6799 | .8326 | .2903 | .3141 |
| Hτ | .6792 | .8571 | .3306 | .3205 |
| G1 | .6645 | .8527 | .3266 | .3077 |
| GSPO | .6622 | .8348 | .3508 | .3397 |
| GiGPO | .6342 | .8304 | .3750 | .2949 |
| DTPO | .6799 | .8638 | .3548 | .3205 |

粗体 = 该列最优（V\* 上 DAPO 与 A+B+C 并列）。

### 2.3 同题配对（uid + sample_idx，2,208 对）

| 对比 | 两者都对 | 仅前者对 | 仅后者对 | 都错 | 后者净增 |
|---|---:|---:|---:|---:|---:|
| GRPO vs A+B | 1157 | 254 | 272 | 525 | +18（WorldBench 39 / 54） |
| GRPO vs A+B+C | 1152 | 259 | 260 | 537 | +1（V\* 33 / 46） |
| A+B vs A+B+C | 1146 | 283 | 266 | 513 | −17 |
| GRPO vs GRPO+A | 1143 | 268 | 278 | 519 | +10 |
| DAPO vs DAPO-SB | 1187 | 278 | 229 | 514 | **−49**（MME 157 / 146、V\* 51 / 33、WorldBench 49 / 35、MMMU-Pro 21 / 15） |
| A+B vs GRPO+A | 1159 | 270 | 262 | 517 | −8 |
| GRPO vs strat | 1147 | 264 | 252 | 545 | −12 |
| GRPO+A vs strat | 1147 | 274 | 252 | 535 | −22 |
| DAPO vs strat | 1188 | 277 | 211 | 532 | **−66** |
| GRPO vs max2 | 1125 | 286 | 279 | 518 | −7 |
| GRPO+A vs max2 | 1139 | 282 | 265 | 522 | −17 |
| A+B vs max2 | 1148 | 281 | 256 | 523 | −25 |
| DAPO vs max2 | 1172 | 293 | 232 | 511 | **−61** |
| strat vs max2 | 1129 | 270 | 275 | 534 | +5 |

净增粗体 = 按题 95% 区间不含 0（只对附有区间的比较标注；DAPO vs DAPO-SB 的区间见[学习信号审计](results/audit_signal_20261105/AUDIT_SIGNAL.zh.md)）。

max2−对照的按题 bootstrap 95% 区间（pp）：GRPO −0.32 [−2.54, +1.90]；GRPO+A −0.77 [−2.90, +1.45]；A+B −1.13 [−3.26, +1.09]；DAPO −2.76 [−4.98, −0.54]；strat +0.23 [−1.90, +2.40]。552 题整体重采样 10,000 次，seed 20261105；[匿名逐题数据](results/sgv2_max2_20261105/paired_questions.json)。

**max2 与 GRPO 的差距拆分**（[grpo_gap.py](results/sgv2_max2_20261105/grpo_gap.py) 按 uid + sample_idx 对齐两份轨迹，只输出聚合数 [grpo_gap.json](results/sgv2_max2_20261105/grpo_gap.json)；分档为描述性计数，裁图数与题目难度相关，不作因果归因）：

| 按 GRPO 在同键上的裁图数 | 条数 | GRPO 对 | max2 对 | 净差 | MME | V\* | WorldBench | MMMU-Pro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 96 | 51 | 53 | +2 | +1 | 0 | −1 | +2 |
| 1 | 1024 | 791 | 778 | −13 | −7 | −3 | −2 | −1 |
| 2 | 574 | 321 | 314 | −7 | 0 | −3 | −2 | −2 |
| 3–4 | 398 | 195 | 198 | +3 | −8 | +7 | +1 | +3 |
| 5+ | 116 | 53 | 61 | +8 | +2 | +4 | +2 | 0 |
| 合计 | 2208 | 1411 | 1404 | −7 | −12 | +5 | −2 | +2 |

- 裁图数分布：GRPO 只裁 1 次的 1,024 条，max2 只有 8 条；max2 裁 5 次及以上 541 条，GRPO 116 条；零裁图 GRPO 96 条，max2 3 条。
- 按自身裁图数的正确率：GRPO 1 次 .772、2 次 .559、3–4 次 .490、5+ 次 .457；max2 2 次 .784、3–4 次 .616、5+ 次 .436。
- 按题（4 次采样平均）max2 比 GRPO 多裁：≤0 的 19 题净差 +13；(0,1] 163 题 −1；(1,3] 321 题 −16；>3 49 题 −3。
- 缺答案 max2 28 条、GRPO 19 条；max2 缺答案的 28 条中 GRPO 同键答对 11 条。

![max2 vs GRPO](results/sgv2_max2_20261105/max2_vs_grpo_gap.png)

strat−对照的按题bootstrap95%区间（pp）：GRPO −0.54 [−2.63,+1.49]；GRPO+A −1.00 [−3.04,+1.09]；DAPO −2.99 [−5.03,−1.04]。552题整体重采样10,000次，固定seed20261105；不是多训练seed，不含训练seed不确定性，未校正多重比较。[匿名逐题数据与完整四格](results/sgv3_20261105/comparison.json)。

- DAPO-SB 的零裁图直答 4.80%（106 条，正确率 .500），DAPO 1.00%（22 条，.591）；有裁图时正确率 DAPO-SB .648、DAPO .664。
- 按题计（每题 4 次采样，DAPO-SB 答对数 − DAPO 答对数）：−3：8 题，−2：34，−1：95，0：314，+1：69，+2：27，+3：5。
- mmmu_pro 上 GRPO、GRPO+A、DAPO-SB 都是 49/156。逐键核对是不同轨迹：DAPO-SB 与 GRPO 同 pred 64/156，GRPO+A 与 GRPO 同 pred 67/156。

本次strat直接比较附按题bootstrap区间；其余历史比较的范围与统计口径保留原记录。strat零裁15条、截断3条、error0条、无答案32条（计数可重叠）；首裁指示均值.9932、后续裁图1.4796/条。[两例正反向逐图记录](results/sgv3_20261105/CASE_REVIEW.zh.md)：已审2/2对、8/8张实际crop；本地环境与远端header哈希不同，不声称远端逐像素重放。

### 2.4 案例（同题 4 次采样；V\* 原图不在公开库中，以下用文字描述）

**案例 1：vstar_82“汽水罐是什么颜色”（GT 绿色，目标仅 8×12 px）。** GRPO 1/4，A+B 4/4，A+B+C 3/4。
- GRPO s2：执行了 7 张裁图，覆盖目标的裁图中，目标都不到裁图边长的 2%。最后一张宽 1011 px，作答 “The soda can has a reddish tint” → B，错。
- A+B s0：第 2 张裁图 56×56 px，目标约占裁图边长的 1/5，作答 “The enlarged view of the soda can shows it is a distinct green” → D，对。

**案例 2：vstar_15“行李箱是什么颜色”（GT 棕色，被栏杆部分遮挡，由一名穿黑裤的人拖着）。** GRPO 3/4，A+B 0/4。
- A+B 有 2 条采样的裁图都没覆盖到这只行李箱，作答 “clearly black”。
- A+B s0：第 4 张裁图紧框覆盖 GT（目标占裁图边长 0.73），仍答 “clearly dark, consistent with black” → C，错。
- GRPO s0：68×56 px 裁图覆盖 GT，答 “brown suitcase carried by the person” → D，对。

**案例 3：vstar_148“铲子在房子的哪一侧”（GT 右侧；GT 框 [1487,939,1612,1210]，图 2250×1500）。** DAPO 4/4，DAPO-SB 1/4。
- DAPO-SB s1、s2、s3 各 1 张裁图，都在图像左下部（如 [0,900,1012,1500]），覆盖 GT 框的比例为 0.0，答 “the shovel is on the left side of the house” → 错。s0 的第 1 张裁图 [1350,750,2250,1500] 完整覆盖 GT，答右侧 → 对。
- DAPO s1、s2、s3 都有完整覆盖 GT 的裁图（如 s2 的 [1440,840,1642,1260]），答 “leaning against the wall at the right side of the house” → 对；s0 的裁图对 GT 最多覆盖 .31，也答对。

**案例 4：MMMU-Pro validation_Music_5（判断“该音符是 G♭ 大调的上主音”，GT 为“错误”）。** DAPO 3/4，DAPO-SB 0/4。
- DAPO-SB 4/4 都没有裁图，都答“正确”。s0 写 “a note on the third line (A above middle C) … the second note of G♭ major is A♭”，随后写 “A♭ = A4” → 错。
- DAPO s0 同样没有裁图，前半句相同，结尾 “The note shown is A, not A♭, so the claim is false” → 对。DAPO s1、s3 有裁图 → 对；s2 无裁图 → 错。

**案例 5（反向）：mme_7685“2020 年 Qtr2 的 Rent Trend 数值”（GT 2.1）。** DAPO-SB 4/4，DAPO 1/4。
- DAPO-SB 4/4 各 1 张横条裁图，读 “the Qtr2 bar labeled 2.1” → 对。
- DAPO s1、s2、s3 读 “the 2020 Qtr2 bar is labeled 2.2” → 错；其中 s1 的裁图 [21,195,999,374] 与 DAPO-SB s3 的 [0,229,1030,348] 位置相近。

**案例 6–7（max2 vs GRPO，V\* 同一道救生圈颜色题，GT 为 B；原 uid 不公开；同题不同采样；按 E1 完成前冻结的字典序规则选出）。** [逐图记录](results/sgv2_max2_20261105/REPORT.zh.md)
- 案例 6（max2 对、GRPO 错）：max2 裁 7 次答 B，所引 observation_4/7 写 “a red top section and a white lower section”，但这两张视图只有水面橙红球体，不含官方目标框——答案对、依据不支持。GRPO 裁 1 次答 C，裁图同样不含目标。
- 案例 7（max2 错、GRPO 对）：max2 4 张裁图都不含目标，写 “orange/red and shows no white band” 后答 D。GRPO 第 2 张裁图含目标，视图可见岸边红白环状物，写 “small ring-shaped object” 后答 B。

### 2.5 V\* 答错轨迹的尺度复算

口径：目标框长边 / 覆盖它的最小裁图长边。

| | V\* 错答 | 其中单裁覆盖 GT | 覆盖时尺度中位数 | 尺度 <5% 的条数 | 答对且覆盖时尺度中位数 |
|---|---:|---:|---:|---:|---:|
| GRPO | 70 | 38 | 0.066 | 14 | 0.193 |
| A+B | 73 | 29 | 0.087 | 7 | 0.218 |
| A+B+C | 57 | 30 | 0.053 | 15 | 0.153 |
| DAPO | 57 | 31 | 0.053 | 14 | 0.171 |
| GRPO+A | 66 | 47 | 0.064 | 18 | 0.213 |
| DAPO-SB | 75 | 31 | 0.051 | 15 | 0.177 |

DAPO-SB 的 V\* 错答比 DAPO 多 18 条，多出的部分落在“没有任何单张裁图覆盖 GT”的一类（44 vs 26）。

---

## 三、训练 Log

- 每张图一个指标；max2 红色粗线，GRPO 蓝、GRPO+A 绿、DAPO 橙、strat 紫为加粗的直接对照，其余条件为淡线背景；每个条件颜色唯一。
- 逐步原值，无平滑、无插值；少于 10 个点的序列（如 dev）以点加虚线绘出。纵轴范围只由 5 个直接对照决定；背景条件超出范围的点不画，图左下角逐条写明条件、极值和步数（如梯度范数图中 DTPO 第 3 步 614、SAPO 第 9 步 209；训练正确率图中 A+B+C 第 32 步 .311）；完全重合的恒定序列（adv_nz 中 DAPO、DAPO-SB、Hτ 恒为 1）在图内注明。A+B+C 第 31–32 步题目构成不同（见 §1.7）；S/gate 与 SB 专有诊断仍为历史快照，不含 strat、max2。
- DAPO 系（DAPO、Hτ、DAPO-SB）的训练正确率、裁图、生成 token 取全部生成的 rollout（含被过滤组），按保留该组的更新步归位；reward 与采样 surprisal 取日志（只含保留组）。不同算法的奖励/归约差异保留。
- 缺测不画并标注。max2 第 32 步：reward .658、训练正确率 .617、采样 surprisal .477、ppo_kl 9.6e-5、裁剪前 grad_norm 7.171、上侧 clip 比例 .006；dev 0.533（第 0 步）→ 0.602（25）→ 0.621（32）。
- 数据：[family_long.csv](results/sgv2_max2_20261105/family_long.csv)（max2 355 行）；复画：[plot_results.py](results/sgv2_max2_20261105/plot_results.py)。max2 训练生成 11,126,332 token、处理 371,949,079 token；32 个 step 计时合计 43,355.981 秒，双卡 × 步时 24.09 GPU·小时（含 step 内验证与保存，不含合并与 E1）。

![max2 训练诊断](results/sgv2_max2_20261105/max2_training_diagnostics.png)

**Reward**（日志里的 rollout 平均奖励；S 系列含 0.1·m·S，G1 含 0.25·m·Φ；DAPO 系只含保留的混合组）。第 32 步：max2 .658、DAPO-SB .542、DAPO .556、GRPO+A .616、GRPO .601、A+B .618

![reward](results/sgv2_max2_20261105/family_01_reward.png)

**训练正确率 Y**（全部生成的 rollout，不含续答行）。第 25–32 步均值：DAPO-SB .636、DAPO .626、GRPO+A .616、GRPO .613、max2 .613、A+B .608、A+B+C .577

![acc](results/sgv2_max2_20261105/family_02_acc.png)

**采样 token surprisal**（actor/entropy_loss，temperature=.7；不是全词表熵）。第 25–32 步均值：max2 .465（第 32 步 .477）、DAPO-SB .359（第 1 步 .448、第 32 步 .341）、DAPO .447、GRPO+A .436、GRPO .448、A+B .469、A+B+C .417

![entropy](results/sgv2_max2_20261105/family_03_entropy.png)

**Diff**（actor/ppo_kl，更新策略相对rollout策略的日志估计，允许负值，逐步原始值）。除 A+B 第 5 步、GRPO+A 第 3 步各 1.0e-3 外，各 run 都在 ±1e-4 以内（max2 最大 9.9e-5，第 17 步）

![kl](results/sgv2_max2_20261105/family_04_ppo_kl.png)

**梯度范数**（裁剪前；max_grad_norm 1.0；记录了梯度范数的 run 每步都 >1，最小 3.05）。第 25–32 步均值：DAPO-SB 12.04、DAPO 7.34、max2 6.41、GRPO+A 5.66、GRPO 5.99、A+B 6.65、A+B+C 4.91

![grad](results/sgv2_max2_20261105/family_05_grad_norm.png)

**非零 advantage 的 token 占比**。全程均值：DAPO-SB 1.00、DAPO 1.00、max2 .89、GRPO+A .74、GRPO .64、A+B .75

![adv](results/sgv2_max2_20261105/family_08_adv_nz.png)

**裁图数 / 条**。第 25–32 步均值：max2 3.66、GRPO+A 3.39、A+B 2.18、DAPO 2.04、A+B+C 2.04、GRPO 1.65、DAPO-SB 1.59

![crops](results/sgv2_max2_20261105/family_06_crops.png)

**生成 token / 条**。第 25–32 步均值：max2 445、GRPO+A 394、A+B 317、DAPO 282、GRPO 247、DAPO-SB 224

![gen](results/sgv2_max2_20261105/family_07_gen_tok.png)

**S 与门控开启率**（历史图，旧配色；只有 S 系列计算这两项，不含 max2）。第 25–32 步均值：S GRPO+A .30、A+B .27；门控开启 GRPO+A .95、A+B .94

![S](results/stage_report_20261105/figs/fig09_S.png)
![gate](results/stage_report_20261105/figs/fig10_gate.png)

**DAPO-SB 的停止探针**（历史图，旧配色；只在保留的混合组上记录）。b0 均值 .257 → .306；组正确率 − b0 在 .16–.27 之间（32 步均值 +.220）；原始 advantage > 0 的回合占比 .46 → .53；探针字母概率总量 .73 → .87；保留组裁图 0.77 → 1.81

![probe](results/stage_report_20261105/figs/fig14_sb_probe.png)

**E1 汇总图**

![E1 family](results/sgv2_max2_20261105/family_e1.png)
![First and later crop costs](results/sgv2_max2_20261105/direct_e1_cost.png)
![max2 vs GRPO](results/sgv2_max2_20261105/max2_vs_grpo_gap.png)

分源（六条件统一评分；每源最优标 best）：
![E1 by source](results/sgv2_max2_20261105/e1_by_source.png)

较早条件（Hτ、DAPO-SB、A+B+C）的分源历史图见 [fig13_e1_by_source.png](results/stage_report_20261105/figs/fig13_e1_by_source.png)（旧配色：该图中红色为 DAPO-SB，不是 max2）。

### GRPO 与 DAPO 的差异

两者的 advantage 公式相同，差别在下面三点：
- **过滤**：DAPO 丢掉全对和全错的题组，不足 128 组就补采。本 run 共生成 7,872 组，GRPO 是 4,096 组。所以 DAPO 进入更新的 token 全部带非零 advantage（GRPO 约 .64），代价是多出约 92% 的 rollout。
- **clip high**：DAPO 0.28，GRPO 0.3。
- **update micro batch**：DAPO 为 1，GRPO 为 2，属于显存工程差异。

E1 上 DAPO 比 GRPO 高 2.45pp。A+B、GRPO+A 是在 GRPO（无过滤）上做的改动，处在两者之间（.6472、.6436）；strat、max2 也是 GRPO 基座上的改动，低于 GRPO（.6336、.6359），与 GRPO 的按题区间都含 0；DAPO-SB 是在 DAPO 上只换 advantage 的改动，低于 DAPO（.6413）。

---

## 行为审计（2026-11-04）

按外部评审提出的检查点，对 DAPO、A+B、A+B+C（GRPO 作参照）做了离线审计，见 [results/sg_20261104/audit/AUDIT.zh.md](results/sg_20261104/audit/AUDIT.zh.md)。要点：

- DAPO 的 4,095 个更新组只覆盖 2,648 道不同的题，其中 1,447 道题被更新 2 次以上，累计 token 是 GRPO 的 2.27 倍。
- A+B 中拿到正 advantage 的错答 rollout 占 2.6%，平均 +0.035，全部来自全错组；混合组没有出现符号翻转。
- A+B 把大量单裁轨迹变成双裁（单裁 1,024 → 615 条，双裁 574 → 892 条）。在 V\* 上，A+B 首次覆盖 GT 后平均再裁 0.91 张（GRPO 0.44）。
- 目标不存在的题（72 条）：正确率 GRPO .403、DAPO .361、A+B .236、A+B+C .167。
- A+B 对 DAPO 的差距集中在 MME（−30）和 V\*（−16）；在 WorldBench 上 A+B 多对 12 条。

## 设计主线：预算化势能信用（草案，2026-11-04）

不再设计新的过程奖励，只改结果信用怎样分配到轨迹和回合：每条轨迹的信用总额固定、末轮保障 2/3；中间工具回合按“看图前后强制作答答对概率的差”（势能差）分配信用，对应导师的四种信用情况和“势能方程”。含依据的实测数（梯度 32/32 步被裁剪、末轮只占约 30% 的 token、按 token 平均时负 / 正信用比为 1.05）、类比来源、与已有方法的区别、风险和验证顺序。见 [design/CREDIT_MAINLINE.zh.md](design/CREDIT_MAINLINE.zh.md)。尚未实现和训练。

## 期待行为清单的实测表（2026-11-04）

用 DAPO step 32 在 1,572 道 rl4k 训练题上各采 8 条（共 12,576 条轨迹），逐条测量 A1–C3 共 12 条期待行为：适用题数、达标率、组内差异率、同题内“做到 vs 没做到”的正确率差、与答对的一致率、全错组中是否仍有差异。见 [results/behavior_20261104/BEHAVIOR_TABLE.zh.md](results/behavior_20261104/BEHAVIOR_TABLE.zh.md)。

Design_PH 执行结果见 [BIND_AUDIT.zh.md](results/behavior_20261104/BIND_AUDIT.zh.md)：剥离协议编号后，B2 达标率从 .572 降到 .273，C2 从 .940 降到 .786。Φ_bind（最终采用的表格绑定）60 条人工抽检：v1 精确率 90%、召回率 63%，v2 精确率 26/26、召回率 60%。被抽到的 30 条答对轨迹全部写对了绑定，所以混合组里“答对的轨迹之间”的 Φ 差异在抽检中全部来自评分器能否识别其写法，而不是绑定对错。

同一批数据上，DAPO 失败组可核验证据辅助线的第二道关结果：可教的全错组只占可核验题的 22/1191，该线按判据停止。

## 后续实验（已列入但未开始）

1. Design_New §3 的分层探针：同一批错题，依次给“模型自己尺度最合适的裁图”“GT 证据图”“GT 三档尺度”“正确转录”“正确绑定”，按选项计分；同时用来校准尺度函数 h。
2. 修正 A+B+C 的题序构造，让每一步的原题集合与 GRPO 一致后重跑，单独得到 C 相对 A+B 的增量。
3. 拆分 A+B 多出的约 0.5 张裁图 / 条、约 80 个生成 token / 条分别落在哪类题上；逐图审查 A+B 在 WorldBench 上的 +15 题。
4. 7B 阶段（Qwen3-VL-8B、Qwen2.5-VL-7B）：模型规模、轨迹、RL 方法不同时更换。
5. DAPO-SB 未跑的检查（Advantage.md §七）：探针扰动与真实停止对照、同状态分叉的四种信用方向、固定根状态基线 $b(h_0)$ 的消融、λ = 0 / 0.5。

DAPO-SB 训练中 advantage 的实际分配（正负信用总量、按对错 × 是否裁图、裁图前后 b 的变化）见 [results/sb_20261104/CREDIT_FACTS.zh.md](results/sb_20261104/CREDIT_FACTS.zh.md)；DAPO-SB 的设计原文见 [design/ADVANTAGE_DAPO_SB_PRO.zh.md](design/ADVANTAGE_DAPO_SB_PRO.zh.md)；给 Pro 的复盘与重设计提示词见 [prompts/PRO_DAPO_SB_20261105.md](prompts/PRO_DAPO_SB_20261105.md)。

## 目录

完整源码（每个方法一个目录，含公式所在函数与行号）见 [code/README.md](code/README.md)。

| 路径 | 内容 |
|---|---|
| [code/](code/README.md) | 13 个方法、E1 评测、rl4k 标注的实际运行代码快照 + MANIFEST（原始 sha256） |
| [results/sgv2_max2_20261105/](results/sgv2_max2_20261105/REPORT.zh.md) | **最新**：含 max2 与 strat 的 family 图、max2 三部分报告、六条件配对、与 GRPO 的差距拆分、psi 参考实现、验收脚本 |
| [results/sgv3_20261105/](results/sgv3_20261105/REPORT.zh.md) | strat 三部分报告、逐题配对、两例审查与验收脚本 |
| [results/stage_report_20261105/](results/stage_report_20261105/) | 较早family图及S/gate/SB专有诊断，保留历史快照 |
| [results/stage_report_20261104/](results/stage_report_20261104/) | 11-04 版的图（不含 DAPO-SB、GRPO+A） |
| [results/sb_20261104/](results/sb_20261104/) | DAPO-SB 冻结记录、三部分报告、曲线、E1 配对表 |
| [results/sg_20261104/](results/sg_20261104/) | A+B / A+B+C / GRPO+A 冻结记录、三部分报告、代码 |
| [results/rereview_20261104/](results/rereview_20261104/) | Probe A/B 撤回与重审 |
| [protocol/CAPTION_V51.md](protocol/CAPTION_V51.md) | 协议全文、13 类 hint |
| [design/ANNOTATION_SPEC.md](design/ANNOTATION_SPEC.md) | L0 / L1 / L2 标注规范 |
| [results/RESULTS.md](results/RESULTS.md) | 结果入口 |
| [docs/README_stage_20261104.md](docs/README_stage_20261104.md) | 11-04 版阶段 README |
| [docs/README_before_stage_20261104.md](docs/README_before_stage_20261104.md) | 更早的 README |
