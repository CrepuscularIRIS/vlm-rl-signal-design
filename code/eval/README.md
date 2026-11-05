# eval：评测与协议

## e1/：E1 runner 与统一评分器

- `run_eval.sh caption TAG MODEL GPU OUT_DIR` 调用 `eval_v4.py`：552 题 × 4 次采样，temperature 0.7，seed_base 20261017，累计生成上限 8,192，上下文 32,768。
- `summarize.py:extract`（L21）取最后一个 `<answer>字母</answer>`，并带 fallback；截断、错误、无答案都计为错，保留在分母里。
- `eval_base.py`、`eval_minio3.py`、`eval_deepeyes.py`：基座与外部模型的评测入口。
- 本目录的 `caption_contracts.py`、`caption_env.py`、`easyr1_caption.py` 是评测时加载的协议适配版本（对越界 bbox 拒绝执行而不是截断），与 `data/adapter/` 不同；`reward_caption.py`、`reward_direct.py` 与 `data/adapter/` 相同，已去掉。

## protocol/：caption v5.1 协议

- `caption_v51.py`：SYSTEM 原文、`task_hints`（L108，13 类 hint 的生成规则）、`parse_step` / `check_step`（L184 / L190，硬检查）、`soft_flags`、`first_user_text`、`receipt_text`。协议全文见 [protocol/CAPTION_V51.md](../../protocol/CAPTION_V51.md)。
