# SAPO v1（turn 级 GAE）

n=1；共享 LM head 读 V/Q，turn 级 GAE（γ=λ=0.95），PPO + 0.1·V + 0.1·Q（`sapo_adapter.py`）；官方 `sapo_official.py`。

- 本目录文件：`OFFICIAL_LICENSE`, `easyr1_caption.py`, `launch.py`, `preflight.py`, `run.sh`, `sapo_adapter.py`, `sapo_official.py`, `test_sapo.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_contracts.py`, `caption_env.py`, `easyr1_patches.py`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
