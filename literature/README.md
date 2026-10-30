# 文献包：13篇来源，3份详细阅读稿

来源为reading_20261028已有归档。本次是整理发布，不是重新精读或复现。完整PDF不重发；链接与原归档SHA在SOURCES.json。未钉版本的链接可能指向更新版本，应以归档SHA区分。

| 文献 | 与本讨论的关系 | 本包阅读状态 |
|---|---|---|
| [adaptvision](https://arxiv.org/abs/2512.03794v2) | 工具/答案信号分配及视觉token口径 | [阅读稿](readings/adaptvision/精读.zh.md) / [逐项笔记](readings/adaptvision/facts.md) / [来源](readings/adaptvision/SOURCE.md) |
| [countercredit](https://arxiv.org/abs/2609.22910v1) | needed/used两个探针对比与成本信用 | [阅读稿](readings/countercredit/精读.zh.md) / [逐项笔记](readings/countercredit/facts.md) / [来源](readings/countercredit/SOURCE.md) |
| [mini-o3](https://arxiv.org/abs/2509.07969v1) | 多轮工具协议、SFT/RL阶段和超轮次处理 | [阅读稿](readings/mini-o3/精读.zh.md) / [逐项笔记](readings/mini-o3/facts.md) / [来源](readings/mini-o3/SOURCE.md) |
| [pixel-reasoner](https://arxiv.org/abs/2505.15966) | 工具探索、冷启动与恢复 | 仅来源索引；未提供本批精读稿 |
| [deepeyes](https://arxiv.org/abs/2505.14362) | 直接RL的模型/协议前提 | 仅来源索引；未提供本批精读稿 |
| [gigpo](https://arxiv.org/abs/2505.10978v1) | 同状态分组与过程advantage | 仅来源索引；未提供本批精读稿 |
| [visualprm](https://arxiv.org/abs/2503.10291v1) | 过程标签到底估计什么 | 仅来源索引；未提供本批精读稿 |
| [med](https://arxiv.org/abs/2602.01334) | 工具带来的gain/harm与能力测量 | 仅来源索引；未提供本批精读稿 |
| [deepseekmath](https://arxiv.org/abs/2402.03300v3) | GRPO组相对优势基础 | 仅来源索引；未提供本批精读稿 |
| [rloo](https://arxiv.org/abs/2402.14740v2) | leave-one-out基线基础 | 仅来源索引；未提供本批精读稿 |
| [dr-grpo](https://arxiv.org/abs/2503.20783v1) | std、长度归一化与训练偏差 | 仅来源索引；未提供本批精读稿 |
| [dapo](https://arxiv.org/abs/2503.14476v1) | 动态采样、clip与实际预算 | 仅来源索引；未提供本批精读稿 |
| [gspo](https://arxiv.org/abs/2507.18071v2) | sequence ratio和更新稳定性 | 仅来源索引；未提供本批精读稿 |

## 状态与使用

- AdaptVision：原SOURCE记载首轮10项核查/修正；本次未重做核查。
- CounterCredit：原核查中断，不能标记完整独立复核通过。
- Mini-o3：详细稿已交付，独立核查未完成；arXiv v1与会议稿差异保留。
- 其余10篇本批只有PDF归档，不把“文件存在”写成“已精读”。

先用DeepSeekMath/RLOO/Dr.GRPO/DAPO/GSPO核对训练量与advantage，再按当前问题读CounterCredit/AdaptVision/GiGPO。论文有效不证明可直接迁移到我们的多轮状态。VisualPRM的过程标签、终答成功率与动作相对收益不是同一个对象。

读论文时必须核对：可得信息、信号估计量、信用接收位置、额外成本、关键消融、与当前协议的区别。不要引用本文献表证明新颖性或“没有算法能解决幻觉”。
