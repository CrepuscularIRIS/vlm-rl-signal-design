# 剩余题补标注（线索层）与对照实验

2026-11-02。承接 [ANNOTATION_TESTBED](ANNOTATION_TESTBED.md)（部分线索的记录格式与 mme_3905 样例）与 [ROUTE_A_DESIGN_V1](ROUTE_A_DESIGN_V1.md)。目标：**让 4,096 题每一题都有可回像素核对的记录**，并给出用这批记录能做的对照实验。

## 1. 范围：还缺什么

终稿 `out/final/rl4k`（v2）之后仍无框或无核对锚点的题：

| 类 | 题数 | v2 现状 | 缺的东西 |
|---|---:|---|---|
| needs_temporal（驾驶单帧） | 406 | 只有 L0 | 线索框（路牌/标线/灯态/车姿）+ 各自支持什么推测 + 明确不可知项 |
| ambiguous_question | 38 | 只有 L0 | 每种合理解读下的读数与对应选项 |
| needs_external_knowledge（保留的） | 36 | 只有 L0 | 图实际提供了什么、答案还需要什么 |
| evidence_not_found | 31 | 只有 L0 | search_scope + 已检查候选（与 absent 同构） |
| low_resolution / visually_insufficient | 55 | 只有 L0 或无框 | 目标位置框（readable=false）+ 清晰可辨部分与不可辨部分 |
| gt_check=disagree | 152 | 框和读数已有 | 按读数推出的选项字母（仅记录，不改 gt） |

（absent 50 已在 v2 里有 search_scope + checked_candidates，不需补。）

## 2. 线索层记录格式（LC，独立文件，不覆盖 v2）

沿用 testbed 的三层：**可见 → 可推测 → 未解决**。写到 `out/final/rl4k_clues/LC/batch.jsonl`，一题一行：

```json
{"uid":"mme_3905","spec_version":"v2c","class":"needs_temporal",
 "clues":[{"id":"b1","role":"clue","label":"黄底黑色向右指向标","bbox":[0.399,0.453,0.434,0.544],"supports":"前方道路右弯"}],
 "context":"道路边界与车道线远处向右弯，与 b1 印证",
 "inference":"车辆若沿当前车道行驶，前方右转有图像依据",
 "unresolved":["自车此刻实际转向/转角","自车实际速度"],
 "interpretations":null, "search_scope":null, "checked_candidates":null,
 "reading_answer":null, "annotation_confidence":"high","note":""}
```

按类的必填：
- `needs_temporal`：≥1 个 clue；`unresolved` 非空（写明单帧拿不到什么）。
- `ambiguous`：`interpretations` ≥2 条，每条 `{label, reading, answer_letter}`；clues 可选。
- `needs_external_knowledge`：clues = 图里实际可读的内容（可为空）；`unresolved` 写答案还需要的图外条件。
- `evidence_not_found`：`search_scope`（框）+ `checked_candidates`（已查过不像的候选）；clues 可选。
- `low_resolution` / `visually_insufficient`：clues = 清晰可辨的部分；目标框照标、`readable=false` 写进 label。
- `disagree`：`reading_answer` = 按已有 L2 读数推出的选项字母；不改任何 v2 字段。

坐标与 v2 相同（相对原图 [0,1]，3 位小数）。**框只作审计参照，不指定模型必须执行的动作**（testbed 的 annotation_policy）。

## 3. 标注流程（与前两轮同一套）

1. **Astra API 两轮**：草稿（全图+分块）→ 叠框+原分辨率裁图核对。disagree 项不重新看图，直接由 v2 的 L2 绑定推 `reading_answer`（脚本算，不耗模型）。
2. **fable 抽查 20%**（分层：每类至少 5 题）+ 全部低置信度题。
3. 校验：bbox 合法；各类必填齐全；`reading_answer` ∈ A–J。脚本 `annotate_clues.py --validate`。

规模：435 题走模型（406+38+36+31+55 里去掉脚本可算的），~25 分钟（24 并发）；152 题脚本直出。

## 4. 对照实验矩阵（每行回答一个独立问题）

| # | 实验 | 回答什么 | 数据/条件 | 成本 | 判读 |
|---|---|---|---|---|---|
| E1 | hint 推理探针（P1-a，ROUTE_A_DESIGN_V1 §2） | hint 改变的是**动作**还是只改变说法 | dev1399 表格/消歧两类题 × A-SFT、DAPO-32 × none/matched/neutral × 4 采样；用现成 eval_caption | 两张空闲卡 ~半天（需授权） | matched−neutral 的动作差（首裁位置、必要操作完成率）>0 且说法差≈0 才支持 H1 |
| E2 | 线索覆盖离线审计（**不训练**） | 在答案无法核验的驾驶题上，RL 后模型是否更常看该看的线索 | k8 训练 rollouts.jsonl（服务器已有，4 算法 × 驾驶题轨迹）× LC 线索框：算每条轨迹的线索覆盖率 | 只读服务器日志，~1 小时 | GRPO/DAPO/GSPO/GiGPO 覆盖率对比 + 同算法对/错轨迹对比；覆盖率不涨 ⇒ 过程信号在这类题上没有可作用的对象 |
| E3 | gt 噪声上限复算（**不训练**） | 152 题 gt 存疑对已测分数压了多少 | k8 rollouts 的 reward 按 `reading_answer` 重算这 152 题 | 同上，脚本几分钟 | 修正前后每算法差值 = gt 噪声的贡献上限；若 ≈0，disagree 清单只是数据卫生问题 |
| E4 | 过程信号训练臂（后置，需导师同意） | Y + λ·m_q·Φ 能否超 DAPO | §2 的 LC 全量并入 m_q/Φ 定义；按 ROUTE_A_DESIGN_V1 §5.3 矩阵 | 4 卡 × 8–12 h/组 | 主指标：无 hint 条件 E1 口径 + 成本 |
| E5 | 工具增益配对（testbed 已列） | 自由裁图比原图直答多取得什么证据、多大成本 | 同题配对直答 vs 自由裁图（dev1399 子集） | 并入 E1 的第四条件即可 | 按源图族汇总"取得新证据/重复放大/看清即停" |

执行顺序建议：E2/E3 先做（零 GPU、直接产出数字）→ E1（两张空闲卡）→ 视结果定 E4。E5 作为 E1 的加臂。

## 5. 风险与边界（沿用已定原则）

- 线索框**不直接进 reward**：先作审计（E2）；进训练要过 ROUTE_A_DESIGN_V1 §6 的四关，且任务奖励为主（导师 10-31）。
- `reading_answer` 只用于 E3 复算与数据卫生，不修改任何训练 parquet 的 gt。
- LC 记录与 v2 分开存放、分开校验，避免污染已交付的 4,096 题终稿。
