# datasets：题池与标注

- **rl4k**：RL 训练题池 `grpo_caption_v51_4k`，4,096 题，原图互不重复（MME-RealWorld / WorldBench / MMMU-Pro）。每个 RL 方法都在这个池上训练 32 步 × 128 题 × 8 条。
- **E1**：固定评测集，552 题 × 4 次采样 = 2,208 条（MME-RealWorld / V\* / WorldBench / MMMU-Pro）。**与 rl4k 没有交集**，E1 上只有 V\* 有官方框。
- **标注层**（规范见 [design/ANNOTATION_SPEC.md](../../design/ANNOTATION_SPEC.md)）：
  - L0（4,096 题）：可答性、GT 核对、任务标签；
  - L1（3,559 题，10,142 个框）：证据框，原图归一化坐标，带 role；
  - L2（3,255 题）：框中读值、绑定关系、evidence_sets；
  - LC（681 题）：没有常规框的题的线索层。
- 原图、parquet 与标注数据本身不在公开库中。

## annotation_rl4k/：标注流水线

| 文件 | 作用 |
|---|---|
| `annotate_api.py` | L0–L2 两轮 API 标注：先出草稿，再叠框并对照原图裁图核对后写 L2；校验报错回灌，最多 2 次。模型地址和密钥只从环境变量读取（`ANNOT_API_URL`、`ANTHROPIC_AUTH_TOKEN`） |
| `annotate_clues.py` | LC 线索层标注与校验（`--validate`） |
| `validate_annotations.py` | 硬错误 / 警告校验 |
| `build_types.py` | 用协议 `task_hints` 生成单位草案（`hint_keys`） |
| `merge_fixes.py`、`prep_review.py`、`make_review_sheets.py` | 审查合并与审查图 |
| `*_BRIEF.md`、`AGENT_PROMPT.md` | 标注、审查、重标时给模型的说明原文 |
| `build_canonical.py` | relation 方向规范化（computable_from_boxes 的 48 条） |
