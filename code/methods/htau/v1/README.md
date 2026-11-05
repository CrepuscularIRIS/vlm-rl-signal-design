# Hτ v1（bbox 覆盖过程奖励）

R = V·(Y + 0.25·m·Φ_τ)，Φ_τ = mean_j min(1, ½(maxIoU_j + 并集覆盖_j))，只计目标占裁图面积 ≥ τ=0.005 的裁图（`reward_d2.py`）；DAPO 过滤。导师逼近构造的第一次实现。

- 本目录文件：`caption_contracts.py`, `easyr1_caption.py`, `grpo_d2.sh`, `reward_d2.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_env.py`, `easyr1_patches.py`, `grpo_v52m_c.sh`, `mem_watch.sh`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
