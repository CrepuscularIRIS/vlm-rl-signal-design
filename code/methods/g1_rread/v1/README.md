# G1 v1（读值一致性过程奖励）

R = V·(Y + 0.25·m·Φ_read)，Φ_read = 作答文本中字段读数与 L2 值精确相等的比例（`reward_rread.py`、冻结评分器 `rread_scorer.py`；离线准入 `run_admission.py`、`admission_stats.py`）。

- 本目录文件：`admission_stats.py`, `caption_contracts.py`, `easyr1_caption.py`, `grpo_r1.sh`, `reward_rread.py`, `rread_scorer.py`, `run_admission.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_env.py`, `easyr1_patches.py`, `grpo_v52m_c.sh`, `mem_watch.sh`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
