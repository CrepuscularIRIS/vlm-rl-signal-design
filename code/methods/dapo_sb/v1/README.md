# DAPO-SB v1（停止相对 advantage）

每回合之前的停止探针 b_k（`stop_baseline.py`：`build_probe_request` L33、`combine` L62、`run_probes` L79、`install_advantage` L127）；A = (Y − b_k)/σ_q（`stop_relative_advantage.py:build_advantages` L31，Pro 参考实现原样）；rollout 接线 `easyr1_caption.py` L378–414。结果见 [results/sb_20261104](../../../../results/sb_20261104/REPORT.zh.md)。

- 本目录文件：`caption_contracts.py`, `easyr1_caption.py`, `grpo_sb.sh`, `probe_smoke.py`, `reward_sg.py`, `stop_baseline.py`, `stop_relative_advantage.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_env.py`, `easyr1_patches.py`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
