# S 奖励 v1：A / A+B / A+B+C / GRPO+A

A：尺度门控证据获取分 S（`reward_sg.py`：`view_scale` L42、`_h` L64、`_psi` L90、`process_score` L112、自检 `_selfcheck` L126）；索引 `build_sg_index.py`（L1/L2）。B：任务尺度 advantage（`easyr1_caption.py:install_task_scaled_advantage` L105，`CAPTION_ADV_TS=1`）。C：证据续答行（`easyr1_caption.py:encode_prefix` L69；数据构造 `build_cont_pool.py`）。奖励装配 `caption_contracts.py:assistant_reward_batch`（L152）。冒烟与验收 `smoke_cpu.py`、`sg_admission.py`、`step_stats.py`。启动 `grpo_sg.sh`：A+B `CAPTION_ADV_TS=1 CAPTION_SG_LAMBDA=0.1`；GRPO+A `CAPTION_ADV_TS=0`；A+B+C 另加 `TRAIN=…_4k_sgc/train.parquet data.shuffle=false`。

- 本目录文件：`build_cont_pool.py`, `build_sg_index.py`, `caption_contracts.py`, `easyr1_caption.py`, `grpo_sg.sh`, `reward_sg.py`, `sg_admission.py`, `smoke_cpu.py`, `step_stats.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_env.py`, `easyr1_patches.py`, `grpo_v52m_c.sh`, `mem_watch.sh`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
