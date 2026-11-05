# DTPO v1（AdaptVision outcome-only 分段 loss）

中间回合为一段、最终回合为一段，各自归一 loss，权重 0.5 / 1，clip [0.8, 1.24]（`dtpo_adapter.py`）；官方实现参照 `dtpo_official.py`（含 `OFFICIAL_LICENSE`）。

- 本目录文件：`OFFICIAL_LICENSE`, `dtpo_adapter.py`, `dtpo_official.py`, `easyr1_caption.py`, `launch.py`, `preflight.py`, `run.sh`, `test_adapter.py`, `test_official.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_contracts.py`, `caption_env.py`, `easyr1_patches.py`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
