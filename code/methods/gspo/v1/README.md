# GSPO v1

- 代码：没有额外代码；序列级 ratio 在 `data/easyr1_runtime/core_algos.py` 的 gspo 分支（L465–480）。
- 启动：`$GSPO`：`loss_type=gspo loss_avg_mode=seq clip_ratio_low=0.0003 clip_ratio_high=0.0004`。K-base GRPO 同 GRPO，起点换成 Qwen3-VL-4B-Instruct。
