# RLOO / RLOO + 首动作分叉 v1（v53）

留一基线；奖励 Y·(1−λN)；首动作前缀强制作答 / 裁图各 4 条，用收益差监督首次动作选择（`decision_loss.py`）。训练协议 v5.2i（`caption_v52i.py`）。早先单独发布的 `code/v53/` 是本目录的子集，已合并到这里。

- 本目录文件：`SHA256.txt`, `caption_contracts.py`, `caption_env.py`, `caption_v52i.py`, `decision_loss.py`, `easyr1_caption.py`, `grpo_v53_dl.sh`, `launch_formal.py`, `reward_caption.py`, `signal_check.py`, `test_decision_loss.py`, `test_review_regressions.py`
- 叠加顺序：`data/adapter/` → 本目录。
