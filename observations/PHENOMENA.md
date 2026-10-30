# 观察到的现象与题号入口

本页归并先前12题×8模型的逐图记录，只报告病例事实。类别不互斥，不是全库频率标签，不把旧三个point强套为结论。是否“必要”或“有害”不能仅凭答对/答错确定。

| 可复查现象 | 具体记录 | 不能从中推出 |
|---|---|---|
| 对象身份/定位不对应 | [mme_1231](cases/mme_1231.md)：本地多数选目标不存在，GSPO/两外部给运动状态；[vstar_78](cases/vstar_78.md)：右侧人物crop与左侧手表主张 | 算法导致了错误；未找到等于全图不存在 |
| 裁中仍误读或误用 | [test_Music_299](cases/test_Music_299.md)：可见符号数量/形状与描述不同 | 音乐理论GT已独立专家核验 |
| 终答对但引用不支持 | [test_Manage_234](cases/test_Manage_234.md)：RLOO答对，但所引用视图与阈值比较量不符；[mme_14258](cases/mme_14258.md)：DeepEyes答对却crop墙面 | 整条轨迹正确；该crop有因果贡献 |
| 反复观察、没有完成回答 | [worldbench_421](cases/worldbench_421.md)：重复框与无法读号码；外部出现图像预算终止 | 所有重复框均无用，或token是唯一瓶颈 |
| 计算/选项/格式不对应 | [worldbench_362](cases/worldbench_362.md)：decision文本计算2但输出数字6，字母评分器不抽取；DeepEyes正文6却选A/4 | 放宽评分后等于推理被修复 |
| 错误可以跨层共存 | [test_Manage_234](cases/test_Manage_234.md)：轴混用、阈值/计数、截断和终答差异 | 已经发现单一统一原因 |

96条来源、原句、模型/checkpoint、sample、轮次、box、hash和限制在data/audit_findings.jsonl及每个病例页。所有原图保留在非公开原始数据中；网页Pro若未取得原图，只能引用已有视觉审查记录，不能冒称完成新视觉核验。

与既有卡片的对应见 [cards](../cards/README.md)。当前未建立全量“幻觉/有效裁图/必要停止”参考标签，未输出这些现象的总体precision/recall。
