# GiGPO v1

episode z-score + 1.0 × 锚点组 z-score；锚点 = (uid, 已收视图, 已执行裁图框, 已用裁图数, 步序号)。`gigpo.py`、单测 `test_gigpo.py`、launcher `grpo_gigpo.sh`。

- 本目录文件：`easyr1_caption.py`, `gigpo.py`, `grpo_gigpo.sh`, `test_gigpo.py`
- 与 `data/adapter/` 逐字节相同、已去掉的文件：`caption_contracts.py`, `caption_env.py`, `easyr1_patches.py`, `mem_watch.sh`, `reward_caption.py`, `reward_direct.py`, `update_ledger.py`
- 叠加顺序：`data/adapter/` → 本目录。
