# Visual Primitives 桥接与 bbox 过程奖励（设计讨论稿）

2026-10-31（纽约时间）。用户要求：把论文《Thinking with Visual Primitives》发现的问题和方法，逐条对照到我们的 caption 协议下；导师已经给了 bbox 过程信号的思路，先按这个方向设计 reward。本文汇总全部上下文。

**各节性质不同，阅读时注意区分：**
- §1–§5 是事实：论文原文、导师原话、协议代码、数据实测，都有出处。
- §6–§7 是讨论稿：对照关系和公式化写法是 CC 的读法。
- §8 记录用户 10-31 的想法。
- §9 是候选想法清单，全部未验证。

所有决定由用户做。

---

## 1. 来源

| 内容 | 位置 |
|---|---|
| 论文 | DeepSeek-AI，*Thinking with Visual Primitives*（2026）。PDF 不在本库重发，见 [literature/readings/visual_primitives/SOURCE.md](../literature/readings/visual_primitives/SOURCE.md) |
| 论文精读 | [literature/readings/visual_primitives/精读.zh.md](../literature/readings/visual_primitives/精读.zh.md) |
| 导师 10-31 要求（匿名技术整理） | [context/MENTOR_20261031_PROCESS_SIGNAL.md](../context/MENTOR_20261031_PROCESS_SIGNAL.md) |
| 协议定义 | [protocol/CAPTION_V51.md](../protocol/CAPTION_V51.md)；分数口径见 [protocol/CURRENT_PROTOCOL.md](../protocol/CURRENT_PROTOCOL.md) |
| Hint 转奖励 | [design/HINTS_AS_REWARD.md](HINTS_AS_REWARD.md) |

---

## 2. 我们的 caption 协议 v5.1

完整定义（SYSTEM 原文、模板、13 类 hint、环境步骤、硬检查、软标记、奖励与 advantage）见 [protocol/CAPTION_V51.md](../protocol/CAPTION_V51.md)。下面讨论要用到的几点：

- 每轮由“think（`Step N | WORD` + 引用视图的短 caption）”和“一个动作”组成。动作是 `<grounding>{"bbox_2d":[x1,y1,x2,y2],"source":VIEW}</grounding>`（[0,1] 坐标，相对 source 视图）或 `<answer>`。
- **坐标只能写在动作里，不能写进 caption。**
- 裁图总是从全分辨率原图重新切，最小边 56 px，最多 8 次。日志 `boxes_abs` 按原图像素记录每次实际执行的裁图。
- 奖励只有终答 0/1；任一轮违反硬检查，整条轨迹终止、记 0；GRPO 整条轨迹共用一个 advantage。

---

## 3. 数据现状（事实，实测）

| 项 | 数值 | 出处 |
|---|---|---|
| RL 题池 `grpo_caption_v51_4k` | 4,096 题：mme_realworld 2,987 / worldbench 610 / mmmu_pro 499；字段里**没有任何框** | 训练题池 parquet |
| 题池中带“不存在”类选项的题（does not feature / contain / show 等） | 725 题有这类选项；其中 14 题的正确答案就是它 | 同上，用正则统计 |
| 带真值框的数据 | V\* 191（目标物体框，xywh）、TreeBench 405（`target_instances`，可以有多个）、O3Bench 345（`layout_infos` 版面区域框、`target_layouts`，以及按 `[layout k]` 引用区域的逐步解释） | 各 benchmark 官方标注字段 |
| 上面这三份数据的角色 | obsbench 评测主池（V\* 191 + TreeBench 399 + O3Bench 345）；A-SFT 已用 V\* 67 题 | 研究者整理的 SFT 统计 |
| A-SFT 行为分布 | 直答 / 1 裁 / ≥2 裁：2,093 / 1,141 / 749 条；locate / re_examine / fallback：1,943 / 907 / 120 轮；HOLD 88 轮、REVISE 2 轮、abstain 0 | 同上 |
| caption 里出现坐标的轮次比例（E1） | SFT 2/4,403；k8 GRPO / DAPO / GSPO / GiGPO：15/6,415、17/6,827、23/7,205、17/6,620 | `coord_status` 统计 |
| 裁图与真值框的 IoU（V\*，E1 第 32 步） | 蓝帆 73×139 px：IoU 0.137–0.155，覆盖比例 1.0；泳圈 42×44 px：IoU 0.005–0.128，覆盖比例 0.95–1.0；四个算法都答对 | [精读](../literature/readings/visual_primitives/精读.zh.md) §8.3 |
| 整图裁剪 | T07 GRPO 的裁图框是 [0,0,0.99,0.99] | E1 轨迹 worldbench_1341, k8_grpo, sample 0 |

### 3.1 E1 样例里和“记忆”有关的现象（只记现象，样本小，不代表频率）

样例编号对应 E1 题号（均为 4k×8 第 32 步、sample 0）：T01 = mme_18681，T02 = mme_17580，T03 = test_Music_60，T04 = test_Basic_Medical_Science_222，T05 = vstar_21，T06 = vstar_108，T07 = worldbench_1341，T08 = worldbench_1378。题面和原图不公开，模型原句可在本库的脱敏轨迹中查到。

- **T07 GSPO**：第 5 步写右下角 “only schedule text, not a green icon”，commit 时又写右下角 “clock icon” 是绿色。
- **T04 GSPO**：第 3 步说要 “bind this branch to the vertebral artery”，第 4–5 步换成 posterior choroidal / PCA，没有写 REVISE。
- **T03 DAPO**：CALC 得出 unison，最终选了 A（minor third）。
- **T01 GSPO**：用户判读为“识别对了，但不记得选项”。

---

## 4. 论文要点（事实，详见精读笔记）

- **问题**：Reference Gap，即 "seeing is not reasoning"，推理时文字跟丢了要指的实体；Perception Gap 不是论文的贡献。
- **方法**：在思考里穿插框和点（坐标为 0–999 的整数，用特殊 token 包起来）；4,000 万级 grounding 预训练；冷启动数据少而精、严格校验；框和点分别做 SFT 和 RL，再做统一 RFT 和 OPD。
- **冷启动的写法**：
  - 计数：一次框出全部候选，再汇总；
  - 细粒度：逐个候选核对条件；
  - 迷宫：DFS，每一步给点，碰到死路就回溯；
  - 线路追踪：途经点的疏密随曲率变化；
  - 负样本：要求依据视觉证据 faithful refusal。
- **RL**：一般 VQA 不监督思考里的 primitive；Format RM 惩罚重复框和无限输出；Quality RM 是 LLM 评审（只看文字，查一致性、自相矛盾、hacking）；迷宫奖励包括因果截断后的探索进度、不可解迷宫的探索完整度、穿墙惩罚、最终路径合法、判断正确；路径奖励包括双向轨迹距离、端点距离、连续性惩罚、答案正确；只用 1 ≤ k < N 的题。
- **证据边界**：没有“有 primitive 对比无 primitive”的消融；拓扑任务的 benchmark 是论文自建的；前沿模型的 thinking budget 统一设为 low。
- **局限**：受分辨率限制、要靠触发词启用、拓扑任务泛化有限。

## 5. 导师的设计（原话摘要，全文见 [context/MENTOR_20261031_PROCESS_SIGNAL.md](../context/MENTOR_20261031_PROCESS_SIGNAL.md)）

- **工具**：crop / zoom in / zoom out / verify / stop / recover，OCR 也算工具；不需要把图转成 OCR 文字。
- **奖励主次**：任务奖励是 primary，过程奖励另加；reward hacking 无法被平衡掉，但可以尝试 reward shaping。
- **目标逼近**：maxIoU(目标框, 到某一步为止所有 crop 的框) + 目标框被覆盖的比例。
- **探索程度**：到某一步为止所有 crop 的框并起来覆盖的区域。
- **gate**：前期用探索、中后期用逼近；两者不要做加权和，否则找到目标后继续扫图仍能拿探索分。
- **前提**：给语料补 bbox，坐标要归一化，然后纳入 RL。
- **风险**：对 bbox 做监督、约束工具使用都有 hacking 风险。
- **原理**：改 reward 本质上是在改 advantage；梯度由 advantage、重要性比率和 clip 决定；只改 reward 往往还需要配套优化。
- **报告格式**：设定；1–2 个 case 加定量分数；reward / entropy / diff 曲线。

---

## 6. 论文问题与方法 → 我们协议下的对应（讨论稿）

**总的对应关系**：论文“在 CoT 里写框”，对应我们**会被执行的裁图框**（区域指针）加上 **`[observation_k]` 视图引用**（视图句柄）。论文“用标注校验 primitive”，对应我们**用真值 bbox 校验裁图动作**。按这个对应，协议和 SFT 都不用改。

| # | 论文的问题 | 论文的方法 | 我们协议里的对应 | 按导师思路的奖励 | 前提 |
|---|---|---|---|---|---|
| 1 | Reference Gap：推理一长就跟丢实体 | 在 CoT 里写框 | 裁图框 + 视图引用 | 目标逼近：裁图有没有落到目标上 | 真值 bbox |
| 2 | 冷启动 primitive 要和标注一致 | 严格校验后才入库 | 不改 SFT；在 RL 里用真值校验裁图 | 同 #1 | 同上 |
| 3 | 标注有漏标、截断、整图大框 | 语义审查 + 几何审查 | 作为我们补 bbox 的标注规范（坐标归一化） | — | 标注流程 |
| 4 | 迷宫需要 DFS 探索和回溯 | 奖励探索进度、探索完整度 | `locate` → `fallback`；fallback 只有 120 轮 | 探索程度：历史裁图并集的覆盖面积 | 不需要真值 |
| 5 | 迷宫：第一次穿墙之后的探索无效 | 因果截断 | **已有同构**：任一轮违反硬检查，整条轨迹终止、记 0 | — | — |
| 6 | 不可解迷宫：要证明“没有路”就得搜完 | 探索完整度 | “The image does not feature …”类题：证明不存在要看完全图 | 没有目标框时只用探索 | 725 题有这类选项，14 题的答案就是它 |
| 7 | 路径：只算单向会被钻空子 | 双向轨迹距离 | maxIoU 惩罚框太大（偏 precision），覆盖比例惩罚没框到（偏 recall） | 导师公式里的两项 | 真值 bbox |
| 8 | 路径：画一半就跳到猜的终点 | 连续性惩罚 | 裁图还没碰到目标就 commit | 看 commit 那一轮的逼近值 | 同上 |
| 9 | 计数和多实例 | 一次框出全部 / 逐个核对 | 一题多个目标框（TreeBench `target_instances`）；O3Bench 有有序的多区域证据链 | 多框时逼近值怎么聚合：导师未说；用户 §8 认为多框是探索和单框行为的交叉 | 多框标注 |
| 10 | 重复框、无限输出 | Format RM | 重复的等价裁图（T03 GiGPO） | 探索覆盖对重复裁图天然不加分 | — |
| 11 | 只用 Normal 难度题 | 离线按 k/N 分难度 | DAPO 在线过滤 | — | 已有 |
| 12 | 要靠触发词启用 | 未解决（局限 2） | 首动作分叉监督（v53） | — | 已在做 |
| 13 | Perception Gap | 局限 1：可以和裁图结合 | 裁图就是我们的核心 | — | — |
| 14 | 思考和答案不一致、自相矛盾 | Quality RM（LLM 评审，只看文字） | §3.1 的几个现象 | 可以用规则检查（选项一致性）；第一版放不放由用户定 | — |

不搬过来的：特殊 token、点格式、预训练、OPD、LLM 评审。

## 7. 导师信号写成协议变量（讨论稿）

- **记号**：
  - `B_t` = `boxes_abs[1..t]`，即到第 t 步为止的裁图框，不含 view 0；
  - `G` = 归一化的真值框（可能是一组框）；
  - `I` = 整张图。
- **目标逼近** A_t = max_i IoU(G, b_i) + |G ∩ ∪B_t| / |G|
- **探索程度** E_t = |∪B_t| / |I|
- **组合方式**：R = Y + 过程项。过程项由 gate 在 A 和 E 之间二选一，不能对 A 和 E 加权。
- **未定，由用户决定或问导师**：
  1. gate 依据什么切换：训练步数、单条轨迹的 A 是否超过阈值，还是组内统计？
  2. 过程项乘多大的系数？
  3. 多个目标框时，A 怎么聚合？
  4. 按步给还是只在终点给：现在整条轨迹共用一个 advantage，按步的势能差加起来等于 A_T − A_0，和只看终点没有区别。要让信号真正落到每一步，需要步级 advantage；GiGPO 代码里有 step advantage。
- **已算出来的风险**：
  - **覆盖项占了大头**。裁整张图也能拿到约 1.0：蓝帆 IoU 0.0025 + 1.0，泳圈 0.0005 + 1.0。好的裁图也只有 1.05–1.14。这个行为出现过（T07 GRPO）。
  - **小目标 IoU 有上限**：裁图最小 56 px，所以 42×44 的目标 IoU 最高只有 0.59。
  - **探索项和成本目标方向相反**：gate 只管“找到之后”，找到之前扫得越多，探索分越高。

---

## 8. 用户 10-31 的想法（记录，原意转述）

1. **拓扑视角的标注**：DeepSeek 用点和线标注轨迹；我们能否从拓扑的角度设计标注？迷宫用点可能可行；探索顺序可能是 DFS 和 BFS 之间的思路。
2. **记忆**：之前看到模型的记忆能力很弱（§3.1），这部分最好也考虑标注。
3. **边界**：点和拓扑写法属于 CoT 的部分，我们的轨迹里没有对照；导师的 bbox 方案已经很直观。
4. **多框标注**：可以理解为“探索”和“单框行为”的交叉。
5. **推进方式**：论文发现的问题和方法，后续逐条对照处理（§6 的表就是逐条清单）。
6. **下一步**：先找出很难标注的题；看能否借其他领域的思路做代理学习（类似 surrogate model），甚至借鉴 BPTT 一类的建模方式。先把这些想法列出来，见 §9。

## 9. 难标注题与代理信号：候选清单（未验证；只列出来，不排序，不作推荐）

### 9.1 先要摸清的：哪些题难标（下一步的统计对象）

- **目标是否单一**：MMMU-Pro 里很多题是整图推理（乐谱、示意图、医学影像，见 T03、T04），可能标不出单一目标框。
- **证据是否分散成多处**：图表的图例加数据点、地图的图例加位置，O3Bench 的解释就是有序的多区域链。
- **不存在类题**：725 题有“不存在”选项，答案为它的题没有目标框。
- **目标是关系而不是物体**：例如 T08 的“上方”、T07 的背景层。

### 9.2 代理信号的候选来源

| 候选 | 思路来源 | 能给出什么 | 已知问题 |
|---|---|---|---|
| A. 教师裁图框作为伪标签 | 蒸馏 / 伪标签 | A-SFT 里 Astra 的裁图框；也可以让教师在 RL 题池上跑一遍 | 那是教师的选择，不是真值；有偏 |
| B. 多个教师的框取共识 | 众包标注聚合 | 几个强模型框的交集或投票 | 调用成本；共识不等于正确 |
| C. OCR 工具定位 | 导师：OCR 也算工具 | OCR 类题：在图里找到选项文字的位置，作为框 | 只覆盖文字类题；选项文字可能不在图上 |
| D. 遮挡 / 反事实敏感度 | 可解释性（occlusion sensitivity） | 遮住某块区域后答案改变 → 这块区域是证据 | 要用模型判断；计算量大；模型本身答错时不可靠 |
| E. 组内成功轨迹的裁图区域 | Hindsight relabeling（HER） | 同一题 8 条 rollout 里，答对那几条的裁图区域作为目标代理 | 答对可能是猜的（T07 DAPO / GiGPO 依据不对但答对了）；全错组没有可用信息 |
| F. 在已标注子集上训练一个“裁图是否命中证据”的打分器，推广到未标注题 | Surrogate model / PRM / value model | 对没有真值的题也能给逼近分 | 要额外训练模型；分布外泛化不确定；本身也可能被 hack |
| G. 开放词表检测器或大 VLM 的 grounding | 用检测模型生成标注 | 用题目里的名词短语去定位 | 名词短语抽取；关系类题不适用 |
| H. 按 BPTT 的思路把多轮轨迹看成展开的循环计算 | BPTT / 截断 BPTT | 把信用沿步骤往回传；HOLD 行类比隐状态；截断窗口类比只往回看 k 步 | 现在整条轨迹共用一个 advantage，transformer 本身就看得到全部历史；能否落地取决于步级 advantage（GiGPO 那一路） |
| I. 用区域邻接图 / 网格的访问记录表示探索过程 | 拓扑、图搜索（DFS / BFS） | 把图像划成网格，记录访问过哪些格子、走的顺序；广度扫（BFS）和深入某个候选（DFS）分开计量 | 网格粒度怎么定；怎么和导师的探索 / 逼近对应。CC 的读法：探索程度偏 BFS，目标逼近偏 DFS，gate 相当于从广度切到深度 |
| J. 记忆的显式化 | 用户 §8.2；协议已有 HOLD | 把已排除的区域、已确认的事实写成 HOLD；协议的 search 提示里本来就有“mention briefly which regions are already excluded”（但 prompt 里没加） | 导师框架要求不为每种现象单独配奖励；HOLD 在 A-SFT 里只有 88 轮 |

## 10. 下一步（由用户定）

1. 按 §9.1 统计 RL 题池里哪些题能标单框、哪些要多框、哪些标不出（先用题型和来源粗分，再抽样看图）。
2. 第一版奖励是否就是“Y + gate(A, E)”；§7 的四个未定项；覆盖项可被整图裁剪刷分的问题，是否一起问导师。
3. §9.2 里哪些值得先做一个小验证。
