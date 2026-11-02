# 标注 testbed｜部分视觉线索和 hint 值不值得学？

2026-11-02。我们用这套标注检验一个问题：**模型能否利用图中确实存在的线索作出推测，同时指出还有哪些条件无法确定？** 原图直答、利用上下文、补看局部都可能有效。框给研究者一个回到像素核对的地方，哪条观察路线更准确、更省计算，由对照实验回答。

## 路牌能支持哪一句话？

| 层次 | mme_3905 的标注 |
|---|---|
| 像素中可见 | b1 是黄色底板上的黑色右向指向标，原图框为 (638,408)–(695,490)。b2 是路面的 SLOW 字样，框为 (1096,519)–(1220,536)。远处车道线和道路边界也向右弯。坐标以原图左上为原点，右、下边界不含。 |
| 可作的条件推测 | 若自车继续沿这条车道行驶，前方右转有路径和指示牌两项依据。SLOW 说明该处有减速提示。 |
| 仍未解决 | 单帧没有给出本车此刻转角、运动速度或是否遵守减速提示。把路牌和路面都裁出来，仍然无法核实“车速非常快”。 |
| 可对照的模型行为 | 原图上读出右弯并保留速度未知；补读指向标后作同一推测；或者读到路标后断言车速。前两种可能都合理，第三种多出一个无依据的断言。 |

本例作为补充记录保存，原有 L0 和 GT 保留，便于同题复比。读者可凭 UID、源图哈希和框坐标回原数据集复核。[两处框及分项标注](../data/annotation_testbed_20261102/mme_3905.json)

## 哪些图片确实缩过？

在服务器读取 RL4K 训练 parquet 的图像路径，逐一打开文件头。**4,096/4,096** 的路径、宽和高与标注 manifest 一致。对抽查的 23 张核对源文件 SHA-256；其中 15 张由训练行直接指向 MME 数据集文件，另 8 张 WorldBench / MMMU-Pro 提取图与上游 benchmark parquet 的图像字节相同。这能定位到数据集交付版本，拍摄设备或网站是否另存更高清母版仍未知。[4,096 题源尺寸](../data/annotation_testbed_20261102/source_sizes.csv) · [核验记录](../data/annotation_testbed_20261102/audit.json)

又在服务器对这 23 张真实图运行了 A-SFT 评测入口所用的 **CPU 首轮图像预处理**。先由裁图环境呈现整图，再以 589,824 像素为上限处理 PIL 图像，最后读取处理器的视觉网格（宽高分别乘 patch 边长 16）。最后一列是处理器网格覆盖的像素尺寸。这次重放没有生成模型答案，也无法追认某条历史轨迹看清了什么。

| 题目 | 数据集源图 | 环境首图 | 像素上限处理后 | 处理器网格对应尺寸 |
|---|---:|---:|---:|---:|
| mme_3905 · 路牌 | 1600×900 | 1596×896 | 1024×575 | 1024×576 |
| worldbench_1971 · 麦克风状态 | 320×240 | 588×448 | 588×448 | 576×448 |
| test_Accounting_328 · 横向工单表 | 736×145 | 2268×448 | 1728×341 | 1728×352 |
| mme_14135 · 骑手裤腿 | 2040×1356 | 1708×1148 | 936×629 | 928×640 |

路牌题确实被缩小；原图框和模型实际收到的首图不能混作同一像素尺度。会议截图在上游文件中就只有 320×240；放大后有些麦克风记号可见，仍不足以可靠核实全部参会者状态。实验中可以检验局部放大是否帮模型辨认已有细节。[23 题各层尺寸、源图哈希](../data/annotation_testbed_20261102/reviewed_views.csv)

## 537 道止于 L0 的题还能标什么？

| 主原因 | 题数 | 可保留的事实与尚缺的条件 |
|---|---:|---|
| needs_temporal | 406 | 如路牌、道路线形、车辆位置可以核对；实际速度或后续行为需要时间信息。 |
| ambiguous_question | 38 | 可以核对哪些物品受到遮挡；若多个候选符合“未全部可见”，选哪件仍有歧义。 |
| needs_external_knowledge | 36 | 固定梁的支座可见，热应力主要依题面计算。可判断图提供了什么，以及正确答案是否需要它。 |
| evidence_not_found | 31 | 可定位路中央黑车；支撑“因交通信号而加速”的信号状态未找到。 |
| low_resolution | 26 | 可标出清楚的状态图标和无法读出的格子，保留计数的不确定性。 |

29 道 visually_insufficient 已计入末两行。另有 50 道 absent，可为看得清的候选记录排除依据；152 道 gt_check=disagree 中仍可保留正确的视觉读数。全池 3,312 道满足 answerable 且 agree，这只是筛选口径。以上均可作为 testbed 的分层；局部线索的质量、适用面和是否值得进入训练，要用样本与对照决定。本次增补了路牌题，其他例子是可标内容的说明。[23 题逐图核对](../data/annotation_testbed_20261102/spotcheck.md)

## Hint 怎样进入主线实验？

《Thinking with Visual Primitives》§2.4 描述按任务组织冷启动语料：例如粗粒度计数用图像和框条件生成示范并校验，细粒度计数还区分相似候选。我们要测试的迁移是**用行为 hint 帮教师构造 SFT 示范**，让学生在没有提示时也能选择合适的操作。项目的 Hint 讨论稿还提出推理时提示、证据状态审计和后续信用分配。下表把能分别回答的问题拆开。[论文来源与精读](../literature/readings/visual_primitives/SOURCE.md) · [Hint 讨论原稿](Hint.md)

| 待测问题 | 同题对照与判读 |
|---|---|
| **教师得到 hint 后，示范质量是否提高？** | 在训练来源的同题同图上比较无提示、匹配 hint、等长中性提示。保持教师、解码和生成预算一致。盲看三组是否提高有依据的读数、身份绑定与部分推测，记录每种条件的合格示范产出及成本。允许教师直接看原图作答。 |
| **用这批示范做 SFT 后，学生能否无提示迁移？** | 比较匹配 hint 构造的语料与控制语料。采用相同学生起点、训练预算、筛选标准和按源图族隔离的评测集；学生输入统一去掉教师 hint。分开报告语料合格率与无 hint 学生的准确率、证据使用及成本。如果只有教师收益，学生迁移仍未得到支持。 |
| **提示在推理时是否只改变说法？** | 在 dev1399 的已核表格／图表与候选消歧样本上，A-SFT、DAPO step32 各自比较 none / matched / neutral。工单可查“金额对、列错”；骑手可查“字读对、人找错”。相同预算下同时记录最终答案、视图与绑定、裁图次数和 token。P1-a 的改善只回答推理时问题。 |
| **额外裁图能否提供原图没有用上的信息？** | 保留自然产生的零裁直答与自由裁图路径。若要检验工具本身的增益，用同题配对的原图直答与允许自由裁图条件对照。按源图族汇总，分辨取得新证据、重复放大、已经看清后直接停止。自然选择裁图和不裁图的两组难度可能不同，二者准确率之差无法单独解释工具效果。 |

评审只需要分清“观察到什么、由此推测什么、什么仍未知”。哪类题要不要裁、裁哪一个框、是否把部分线索转为 reward，都留给实验。此页交付为设计和 CPU 核验，hint 推理效果、SFT 迁移和训练收益均待测。[已有 P1-a 设计](ROUTE_A_DESIGN_V1.md)

### 两张可复用的 hint 对照卡

无提示条件使用原协议。下列 matched 为 v5.1 协议原文，neutral 为本次拟定的操作无关文字。用参考 tokenizer 对完整题面和首轮 head 编码，相对无提示，两道题的两种附加文字分别同增 62 和 51 tokens；正式比较时再用各目标 checkpoint 核对。

**工单表 · table**

匹配提示

> Hint: Table: find the row by its row header and the column by its column header (the full path for multi-level headers) and keep both with the value; watch units, footnotes and total rows. If a row or column binding is corrected, withdraw the values and comparisons that depended on it.

中性对照

> Hint: This question is part of a collection of items presented in a common format. The collection includes a variety of topics. Each entry appears on its own, with the same general layout as the other entries. This introductory note accompanies the item as part of its presentation. The same convention applies throughout the collection.

**相似骑手 · instance**

匹配提示

> Hint: Find the instance that meets every condition in the question before reading its attribute; if another candidate partly matches, check the condition that separates them; keep that same instance in later views, and when you switch candidates say which earlier attributes no longer apply.

中性对照

> Hint: This question is part of a collection of items presented in a common format. Different entries in the collection may cover different topics. Each item has its own place in the collection. This note accompanies the entry. The same convention applies throughout the collection.
