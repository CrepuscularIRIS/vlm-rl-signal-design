# L2 事实抽取器 b_if 离线准入（2026-11-05）

代码：`obs/src/l2fact/l2fact.py`（只依赖标准库，可直接拷到训练代码目录供 `l2fact_adv.py` 导入）、`obs/src/l2fact/test_l2fact.py`（反例测试 58/58 通过：`python test_l2fact.py`）。
统计：`admission.py`（`python admission.py`，加 `--with-count` 为含计数版本），输出 `admission_stats{,_with_count}.json`、`states{,_with_count}.jsonl`、`l2fact_index{,_with_count}.json`。
抽检：`audit_b.json`、`audit_lite.json`，由 `make_audit_files.py` 写出；抽样候选在 `audit_round{1..4}_candidates.json`、`audit_lite_candidates.json`。
轨迹：`runs/behavior_20261104/traj_{gate2,pool2}_dapo_{a,b}.jsonl`（12,576 条）。full_response_text 取各轮 `turns[].text` 按顺序直接拼接。有效终答 = 最后一轮 `parsed.kind == "answer"` 且 pred 非空；否则全部置 None。

## 1. 覆盖范围

| 类型 | 状态 | 依据 |
|---|---|---|
| table（数值） | 收入（默认） | 第 5 节终轮抽检 |
| count（每题恰好 1 个 count binding） | 已实现，默认关闭（`build_index(..., include_count=True)` 打开） | 第 5 节终轮 b=0 8/10；第 6 节 5 对中 3 对的差异来自抽取错误 |
| chart | 不收 | 轨迹池内资格题只有 9 个 chart binding，无法抽检 |
| entity、compare、relation、chain | 不收 | 值为自由文本，没有实现可核对的错值判定 |

table 判定规则（详见 `l2fact.py` 文件头）：
- 一句话（剥离 `Step N | WORD`、`[observation_k]`、`<grounding>`、`<answer>` 等元数据）同时提到目标行和目标列，且某个数字直接挂在目标标签上：`标签 + 只含连接词的间隔 + 数字`，或 `数字 + in/for/under/( /= + 标签`；
- 序号类数字（row 39、Year 7、column 3）、年份、选项引用、否定、算式中的操作数、`→` 序列、裸列表、`respectively`、带“need / verify / appears / not bind”等措辞的句子都不算陈述；
- 同一句里多个不同值且没有纠正词 → 归属含混（None）；按出现顺序取最后一次陈述，最后一次之后再出现含混句 → None；
- 值比较：分隔符的任一读法相同 → 1；只在 ×10^k 或四舍五入后相同 → None；否则 0；
- 只有满足以下全部条件时才给 0：行名为“强提及”（完整标签含括号限定词，或复合行名至少 2 段）；数字挂在行上，或与行名在同一个逗号子句；同一句中没有未归属给其他标签的 v*；陈述值与 v* 的百分号一致。不满足时这句只能给 1 或 None。

## 2. 资格题与事实数

| 项 | 数 | 出处 |
|---|---:|---|
| rl4k 中 L0 answerable 且 gt_check=agree 且 L2 bindings 非空 | 1,925 | `admission_stats.json: eligible_L0_L2_all` |
| 其中在 1,572 道轨迹题内 | 1,379 | `eligible_in_traj_pool` |
| 这 1,379 题的 binding（按类型） | table 1,281；entity 520；count 174；relation 63；compare 59；chart 9；chain 2 | `bindings_by_type_eligible_in_pool` |
| 默认索引（table）题数 / 事实数 | 732 题 / 1,089 个（轨迹池内 729 题 / 1,086 个） | `index_questions`、`facts_by_type_index` |
| 含 count 索引 | 886 题；table 1,089 + count 154 | `admission_stats_with_count.json` |

table 1,281 个 binding 中 1,086 个成为事实；其余为非数值（`-`、文本、列表、`999°99′`）、列名为纯数字或通用词（value/amount）、或同一键标了两个不同值。逐项丢弃原因的数量未统计。

轨迹池内每题 table 事实数：1 个 644 题，2 个 20，3 个 15，4 个 13，5 个 6，6 个 9，7 个 1，8 个 5，9 个 3，10 个 8，12 个 4，19 个 1（`facts_per_question_in_pool`）。

## 3. b 的分布（默认 table 版，5,832 条轨迹 = 729 题 × 8；无有效终答 39 条）

| | b=1 | b=0 | None | 合计 |
|---|---:|---:|---:|---:|
| 全部 | 3,066 | 268 | 5,354 | 8,688 |
| 答对 | 2,865 | 43 | 3,391 | 6,299 |
| 答错 | 201 | 225 | 1,963 | 2,389 |

出处 `admission_stats.json: b_dist`。至少有一个非 None 状态的轨迹 2,998 / 5,832。b=0 的 268 个中，陈述值等于 L2 distinguishing_condition 某个负例值的 163 个。

含 count 版本中 count 事实：答对 b=1 222 / b=0 26 / None 157；答错 b=1 5 / b=0 503 / None 319（`admission_stats_with_count.json`）。

## 4. 构造层面统计（n=8，Y=ok，层 = 组内同 Y；P 用 `grpo_sgv4/l2fact_adv.py: pair_potential`）

默认 table 版，729 组：全对 378、全错 23、混合 328。

| | 混合组·答对层 | 混合组·答错层 | 同质组·全对 | 同质组·全错 |
|---|---:|---:|---:|---:|
| 同层轨迹对 | 4,075 | 1,401 | 10,584 | 644 |
| \|C_ij\|>0 | 1,513（.371） | 149（.106） | 4,496（.425） | 102（.158） |
| D_ij≠0 | 33（.0081） | 8（.0057） | 42（.0040） | 0 |

| P_i | 混合组·答对层 | 混合组·答错层 | 混合组合计 | 全部组 |
|---|---:|---:|---:|---:|
| 轨迹数 | 1,694 | 930 | 2,624 | 5,832 |
| P_i≠0 | 48（.028） | 13（.014） | 61（.023） | 105（.018） |
| \|P_i\|（非零）均值 | .344 | .413 | .359 | .323 |
| \|P_i\|（非零）十分位 | .17 .17 .17 .20 .25 .33 .33 .50 .76 | .20 .20 .25 .30 .33 .50 .50 .52 .84 | .17 .17 .19 .24 .25 .33 .50 .50 .73 | .14 .14 .17 .17 .25 .29 .33 .50 .71 |

非零 |P_i| 的取值分布见 `admission_stats.json: P_*.abs_P_hist`。全部组的 105 条里有 44 条在同质组，那里 A^Y=0，A^L2 不变。

含 count 版：混合组 P_i≠0 107 / 3,688（.029），答对层 79 / 2,075，答错层 28 / 1,613；混合组答对层 D_ij≠0 53 / 4,635。

## 5. 人工抽检 b=0 / b=1

方法：对照 L2 标注（binding、values、distinguishing_condition）读轨迹文本，判断标签是否正确；没有看图。逐条记录在 `audit_b.json`。

- 第 1–3 轮（种子 20261105、20261106、20261107）用来改规则，不是独立样本。每轮 table b=0 抽 30 条，判错数：第 1 轮 10、第 2 轮 7、第 3 轮 3（错误类型与对应修改见 `audit_b.json: dev_rounds`，每一类都加了反例测试）。第 1 轮 table b=1 30/30、count b=0 10/10、count b=1 10/10 判对。
- **终轮**（种子 20261108，`audit_round4_candidates.json`，按类型分层：table 各 30、count 各 10）：

| 类型 | 标签 | 抽检 | 判对 | 判错（应为 None） |
|---|---|---:|---:|---:|
| table | b=0 | 30 | 29 | 1：“I need the row … to verify that next EBITDA is 97.456.506” |
| table | b=1 | 30 | 30 | 0 |
| count | b=0 | 10 | 8 | 2：数的是 “filled blocks”，标注条件是全部方块；“= 5 boxes, but this exceeds any multiple-choice option” |
| count | b=1 | 10 | 10 | 0 |

- 终轮之后的修改：HEDGE 加入 need / verify / whether / exceeds / not listed；count 的 `CALC: 2 = 1 + 1` 不再读作 1。在终轮 80 条上重跑，状态改变的只有上表的 table #6 和 count #33（都变为 None）。这次修改之后没有再做独立抽检。
- 终轮 table b=0 中有几条依赖标注才能判断：#16 轨迹明确把 195.00 赋给 Layer Feed Costs（v* 206.88 被说成 Grower Feed Costs 的值）；#25 标注 n2 说明 $7,405 属于改造前 Annual Potential Rent，轨迹把它赋给 Annual Rent；#9 轨迹写 “2026 = Year 7” 并读了那一列。

## 6. 答对层 D_ij≠0 的 15 对

抽样：混合组、两条都答对、D_ij≠0，种子 20261108，抽于含 count 版本（当时池 57 对）。记录在 `audit_b.json: pairs_correct_stratum_D_nonzero`。

| 类型 | 对数 | 差异来自真实读值 / 绑定差异 | 来自抽取或写法 |
|---|---:|---:|---:|
| table | 10 | 10 | 0 |
| count | 5 | 2 | 3（“CALC: 2 = 1 + 1” 读成 1；“37 total branches, but … only 27 labels”；数的是全部悬挂器具而不是木柄器具） |

table 的 10 对：两条轨迹对同一键明确写了不同的值（例如 mme_10650 s6 写 “the 2026 column is Year 8” 后读那一列，s7 读 Year 2 列）。其中 mme_8913 s0 的错绑句带有 “but the column headers are outside this crop”。默认 table 版中这 10 对的 D_ij 仍为 ±1；默认版混合组答对层 D_ij≠0 共 33 对。

对照上次 Φ_bind（`BIND_AUDIT.zh.md`：答对轨迹之间的差异在抽检中全部来自写法能否被识别）：本次 table 的 10 对都是值或绑定的差异。

## 7. 轻量版构造的抽检（主会话追加）

脚本：轻量版回放脚本（本地，未公开）只读；`audit_lite_sample.py` 只 exec 其 `process()` 之前的索引与 receipts 代码，按原脚本公式重算 P_i。重算得到被调整轨迹 816 条（与原脚本输出一致）。被调整轨迹中各 x 的计数：x=0 答对 410 / 答错 364；x=+1 326 / 18；x=−1 20 / 98；⊥ 16 / 22。

抽样：种子 20261105，被调整轨迹中 x=0 答对、答错各 18 条；x=+1、x=−1 各 10 条（答对、答错各 5）。逐条对照 L2 和最后作答回合，记录在 `audit_lite.json`。

x=0 的 36 条：

| 类别 | 合计 | 答对层（18） | 答错层（18） |
|---|---:|---:|---:|
| (a) 确实没有报告该事实 | 7（19%） | 5 | 2 |
| (b) 报了正确读数，规则没识别 | 17（47%） | 12 | 5 |
| (c) 报了错误读数，不在易混淆值集合里 | 9（25%） | 1 | 8 |
| (d) 其他：报了易混淆值，但行或列不在同一句，没匹配上 | 3（8%） | 0 | 3 |

(b) 的 17 条按写法：行或列只在前一句或分号前出现 7 条；`$` 或千分位的表面形式不同（`$33,138,155` 对 `33,138,155`，`10,000` 对 `10000`）3 条；按顺序列出频数、不逐项写键 3 条；列名换了说法（Units、Machinery #）2 条；`≤210` 对 `≤ 210` 1 条；行名写法不同（标注 `P(下方P行)`，轨迹写 `II, P`）1 条。

x=+1：10/10 判对。x=−1：5/10 判对；判错的 5 条：2 条轨迹写的是正确值，但 `$` 形式与 v* 不同、同句列有易混淆值（mme_10355 s6、mme_6801 s0）；1 条最后一句写对了值但该句没有列名（mme_7252 s2）；2 条轨迹给该键的值不在易混淆集合里，−1 来自同一列表中属于其他列的易混淆值（mme_10473 s7、mme_10764 s1）。

## 8. 已知漏判 / 误判类型（table）

- 漏判（给 None）：行或列只在前一句出现；“the value / its value” 指代；逗号后接 “which shows N”（逗号不允许跨越）；只提复合行名的一段且值错误；只提 “2025” 而标注列为 “2025 F”；“respectively” 映射；带 need / verify 等措辞的句子；值经过换算或四舍五入。
- 剩余误判风险（终轮未再观察到，属于规则覆盖不到的情形）：L1 没有标出的同形兄弟标签（例如另一个 “Revenue” 分区）只在不带数字、月份、年份的列名上无法用形状识别；轨迹自称的行与标注行同名但实际是另一张表中的同名行（第 3 轮 % 表一例已用百分号规则拦下，其他单位差异没有规则）。
- 每题事实来自 L2 标注；标注与图像不一致时，b=0 记录的是“轨迹与标注不一致”。抽检没有看图。

## 还没弄清的问题

- 无。
