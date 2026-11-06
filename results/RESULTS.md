# 结果索引

每个实验一个目录。E1 分数与相对底座的配对差集中在 [README §2 总表](../README.md#2-结果总表e1step-32)；本页只说明每个目录是什么、主文件在哪。行按类别排列，类内按时间。

> **更正（2026-11-04）**：11-03 的 oracle Probe A 计分有误（正则把冠词 “A” 当作选项字母），依赖 Probe A/B 的结论（读取层 43%、单图读不对 8.3%、G1 回填实锤）作废；复核见 [rereview_20261104](rereview_20261104/RESULTS.zh.md)。相关报告正文已移至 [`_superseded_20261106/`](_superseded_20261106/)，原始探针数据仍在原目录。

## 基线与评测协议

| 目录 | 内容 |
|---|---|
| [baseline_classes_20261031](baseline_classes_20261031/REPORT.md) | 三类基础 baseline：Base / A-SFT（直答、CoT、原生工具）/ outcome-only GRPO |
| [native_zoom_normalized_20261031](native_zoom_normalized_20261031/REPORT.md) | A-SFT 走原生 zoom 工具：坐标换算修正后 32.16% → 44.88%；取消累计生成上限后不变 |
| [CONDITIONS.md](CONDITIONS.md) · [PAIRED_RESULTS.md](PAIRED_RESULTS.md) · [paired/](paired/) · [data/](data/) | 25 个已发布条件的目录、同题得失表与原始汇总数据 |
| [TRAINING_SIGNALS.md](TRAINING_SIGNALS.md) · [ALGORITHM_CURVE_AUDIT.md](ALGORITHM_CURVE_AUDIT.md) | 早期训练信号与六算法曲线核对 |

## 优化器对照

| 目录 | 内容 |
|---|---|
| [po_review_20261030](po_review_20261030/REPORT.md) · [po_diagnostic_20261031](po_diagnostic_20261031/RESULTS.md) · [po_final32_20261031](po_final32_20261031/REPORT.md) | PO 两方法（SP / CA）：中期审查、轨迹对照、32 步终点 E1 与逐图审查 |
| [dtpo_e1_20261103](dtpo_e1_20261103/RESULTS.zh.md) | DTPO（中间 / 最终回合 loss 权重 .5:1）32 步 E1、配对与训练记录 |
| [sapo_vlm_20261102](sapo_vlm_20261102/REPORT.md) | SAPO（单 rollout、turn 级 GAE）32 步 E1 57.70% |

## GRPO 底座的方法

| 目录 | 内容 |
|---|---|
| [rread_admission_20261103](rread_admission_20261103/RESULTS.zh.md) → [g1_rread_20261103](g1_rread_20261103/RESULTS.zh.md) | G1 读值一致性奖励：离线准入与训练报告（E1 .6395，与 GRPO 持平） |
| [bsftc_grpo_20261103](bsftc_grpo_20261103/RESULTS.zh.md) | 因果叙事 SFT + GRPO（起点 .4543 → .6033；该线已停） |
| [sg_20261104](sg_20261104/RESULTS.zh.md) | A+B、A+B+C（[结果](sg_20261104/RESULTS.zh.md)）与 GRPO+A（[报告](sg_20261104/REPORT_sg_a.zh.md)）；冻结记录 [FREEZE](sg_20261104/FREEZE.zh.md) |
| [sgv3_20261105](sgv3_20261105/REPORT.zh.md) | strat：同结果层中心化的过程 advantage；[两例逐图审查](sgv3_20261105/CASE_REVIEW.zh.md) |
| [sgv2_max2_20261105](sgv2_max2_20261105/REPORT.zh.md) | max2：奖励中加 psi v2 的 S；含 psi 参考实现与对 GRPO 的差距拆分 |
| [entry_ab_20261106](entry_ab_20261106/REPORT.zh.md) | 进入更新拆解：R（回放补满，1× rollout）与 P（混合题追加一组，1.56×）；[设计与预登记](entry_ab_20261106/DESIGN.zh.md) |

## DAPO 底座的方法

| 目录 | 内容 |
|---|---|
| [htau_training_20261102](htau_training_20261102/PROTOCOL.zh.md) | Hτ：DAPO + τ 门控的 bbox 覆盖奖励（冻结与启动记录） |
| [sb_20261104](sb_20261104/REPORT.zh.md) | DAPO-SB：停止相对 advantage；[冻结](sb_20261104/FREEZE.zh.md)、[advantage 实际分配](sb_20261104/CREDIT_FACTS.zh.md) |

## 离线准入、诊断与审计

| 目录 | 内容 |
|---|---|
| [d1_b_offline_20261102](d1_b_offline_20261102/RESULTS.zh.md) · [d2_offline_20261102](d2_offline_20261102/RESULTS.zh.md) | 按题几何奖励（D1-B）与覆盖 + 引用（D2）的冻结协议与离线结果 |
| [rereview_20261104](rereview_20261104/RESULTS.zh.md) | 11-03 oracle 探针 / V\* 分解 / G1 audit 链的重审（含更正） |
| [behavior_20261104](behavior_20261104/BEHAVIOR_TABLE.zh.md) | 期待行为清单实测表（DAPO step 32，rl4k）；[绑定审计](behavior_20261104/BIND_AUDIT.zh.md) |
| [audit_signal_20261105](audit_signal_20261105/AUDIT_SIGNAL.zh.md) | 各方法学习信号审计：进入更新的数据 / 奖励 / 组内 advantage / 信用分配 |
| [diag_d0_20261105](diag_d0_20261105/D0_REPORT.zh.md) | 组内对比诊断（D0）与 [psi 审计及 v2 改写](diag_d0_20261105/AUDIT_PSI.zh.md) |
| [l2fact_20261105](l2fact_20261105/ADMISSION.zh.md) | L2 事实抽取器离线准入 |
| oracle_g1_20261103 · oracle_probes_20261103 | 11-03 oracle 探针原始数据（报告正文已撤回，见顶部更正） |

## 跨实验汇总

| 目录 | 内容 |
|---|---|
| [main_table](main_table/main_table.md) | README 总表：生成脚本 [make_main_table.py](main_table/make_main_table.py) 与成本表 [costs.csv](main_table/costs.csv) |
| [e1_by_bench_20261106](e1_by_bench_20261106/table.md) | E1 按 benchmark 与四 bench 不加权平均、相对各自底座的配对差（脚本与图） |
| stage_report_20261104 · stage_report_20261105 | 阶段报告用图（family 曲线、S / gate、DAPO-SB 探针、分源） |
| [_superseded_20261106](_superseded_20261106/) | 已撤回的报告正文（G1 后 audit、oracle probes、感知差距简报） |
