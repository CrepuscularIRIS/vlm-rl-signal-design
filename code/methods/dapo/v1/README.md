# DAPO v1

- 代码：没有额外代码；过滤与补采在 `data/easyr1_runtime/ray_trainer.py` L518–556。
- 启动：`$DAPO`：`online_filtering=true filter_key=overall filter_low=0.01 filter_high=0.99 max_try_make_batch=100 clip_ratio_high=0.28`，配 `$U1`（update micro batch 1）。
