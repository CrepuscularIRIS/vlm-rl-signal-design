# 当前结果

> **更正（2026-11-04）**：本文所依据的 Probe A 计分有误——探针不给题面，正则把模型描述里的冠词 “A” 当作选项字母，DAPO 15/15、G1 14/14 命中均如此；Probe B 167/180 复述模型原错答。依赖 Probe A/B 的结论（读取层 43%、单图读不对 8.3%、G1 回填实锤）作废。复核与仍成立的测量见 [results/rereview_20261104](../results/rereview_20261104/RESULTS.zh.md)。

**[DAPO vs DAPO-SB（停止相对 advantage，2026-11-05）](sb_20261104/REPORT.zh.md)**：E1 DAPO-SB .6413 vs DAPO .6635（配对净增 DAPO +49/2208；vstar 391→373、worldbench 86→72）。裁图 1.63 vs 2.09 张/条，零裁图 4.8% vs 1.0%，总处理 token 7,055 vs 8,930。训练 entropy .448→.341（DAPO .446）。Advantage.md §七的离线验收与根状态基线消融未跑；一个 seed。曲线 `data/sb_curves.png`。

**[GRPO vs GRPO+A（单 A：只加 S 奖励，2026-11-05）](sg_20261104/REPORT_sg_a.zh.md)**：E1 .6436 vs GRPO .6390（配对净增 +10/2208），裁图 4.02 vs 1.91 张/条，总处理 token 18,034 vs 8,125。曲线 `data/sg_a_curves.png`。

**[GRPO vs A+B vs A+B+C（Design_New，2026-11-04）](sg_20261104/RESULTS.zh.md)**：E1 GRPO .6390 / **A+B .6472**（配对净增 +18/2208，worldbench .335→.395）/ A+B+C .6395（净增 +1；vstar .844→.873）。A+B 的裁图从 1.905 增至 2.404 张/条，生成 token 从 261 增至 342。A+B+C 第 31–32 步因构造副作用，原题集合与另两臂不同。曲线 `sg_curves.png`；分层探针与尺度校准未跑。

**主实验 [G1 (R-read)](g1_rread_20261103/RESULTS.zh.md)（2026-11-03）**：E1 .6395 vs G0 .6390 持平；训练池同题读取一致性 .5518→.6911（配对 +0.139，750↑/140↓）但未转化到端到端——落在冻结协议「读取改善未转化」行；去选项内容变化测试未跑。主图 `g1_main.png`，全家福曲线（vs DAPO/DTPO/GSPO/GIGPO/GRPO）`family_curves.png`。附属：因果 SFT+GRPO 对照 .6033（起点 .4543，SFT 差距 11.3pp 被 RL 收窄至 3.6pp，该线已停，见 [bsftc_grpo_20261103](bsftc_grpo_20261103/RESULTS.zh.md)）。

**[G1 后 audit（2026-11-04）](oracle_g1_20261103/RESULTS.zh.md)**：判别链闭合——Φ .55→.69 但 Probe A 7.8%≈8.3% 不动 = 表达适配/答案回填实锤，感知与推理两层未移动；on-policy 奖励必须接到不可回填侧（看见什么/看哪里）。**[R-bind 方向冻结草案](../design/R_BIND_DIRECTION.zh.md)**：L1+L2+LC 三层标注 → primitives 信号位（空间基元/绑定关系/线索框），可行性判据已实测（未饱和 11.5%、真方差 19.1%、几何端不可回填），探索行为搁置。

前置准入 [R-read 离线准入（2026-11-03）](rread_admission_20261103/RESULTS.zh.md)——评分器 16/16 反例通过、2,048 条真实轨迹双路径零偏差、有效读取差异率 0.509 ≫ 10% 准入线。

其余新增入口：[三类基础baseline：Base / A-SFT / outcome-only GRPO](baseline_classes_20261031/REPORT.md)。原先的八方法表保留为历史算法对照；不再代替第一、二类展示。

事实快照；本次没有新增GPU评测。每方法2208条，失败保留分母。上游已发布25条件均快照到data/rollouts；完整目录见 [catalog](CONDITIONS.md)。

| 方法 | E1 | 裁图/条 | 总处理token/条 | 生成token/条 |
|---|---:|---:|---:|---:|
| grpo | 0.6390 | 1.905 | 8125.4 | 261.2 |
| dapo | 0.6635 | 2.092 | 8929.8 | 287.0 |
| gspo | 0.6395 | 2.263 | 9756.2 | 277.8 |
| gigpo | 0.6209 | 1.998 | 8506.3 | 275.3 |
| rloo | 0.6318 | 1.992 | 8535.4 | 278.0 |
| rloo_decision | 0.5919 | 1.372 | 6266.1 | 203.3 |
| minio3 | 0.6395 | 2.703 | 17279.8 | 679.7 |
| deepeyes | 0.5589 | 0.852 | 13799.6 | 3291.5 |

相同总分可能来自不同题目，见 [同题得失](PAIRED_RESULTS.md)。条件裁图数下的题目组成不同，不能把层间准确率解释成多裁一次的因果收益。没有参考状态/区域标签，暂不报告“必要观察”或“有效裁图”的precision/recall；pass@4也不是区域召回率。

## 图表与数据

![八组性能成本](../figures/eight_method_comparison.png)
![分来源](../figures/source_accuracy.png)
![按实际裁图数分层](../figures/crop_stratified_accuracy.png)
![训练验证](../figures/six_training_curves.png)

每张图有同名PDF。数据在data/summary_eight.json、data/strata.csv、data/validation_points.json和data/training_summary.json。GSPO step25 .600>DAPO .595；step32 .621<.637；这里只能记录已观测点的交叉，不能定位具体反超时刻或原因。

## v53结果与预算

control墙钟9h42m09，decision11h32m34；主rollout各32768条。decision额外分叉32744条，成本单列。开发验证末步 .595/.585是422题开发结果，不能与E1 .6318/.5919混用。

E1 decision少裁图、少token同时掉分；协议标记390/2208，其中160条仍按E1答对。标记不是评分器自动判错，不能将分差直接归给协议。外部模型caption标记必须N/A，不能填0。
