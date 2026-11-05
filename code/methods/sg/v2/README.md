# S 奖励 v2：导师的 psi（逼近 / 探索），2026-11-05

叠加在 `sg/v1` 之上，只改三个文件。`reward_sg.py`：逼近 prox_b = ½(历史裁图最大 IoU + 历史裁图并集覆盖)，探索 = 全部历史裁图并集面积 / 原图面积（不做放大筛选）；`CAPTION_SG_PSI=gate` 时 psi = 0.25 + 0.75·prox（gate 开）否则 0.25·explore，`CAPTION_SG_PSI=max` 时 psi = max(prox, 0.25·explore)；S = clip(psi_t − psi_0, 0, 1)，原图不算裁图；账本记录 prox / gate / explore / psi / state；自检 12/12。`build_sg_index.py`：新增 `--sets l1`（只用 L1 框）。`grpo_sgv2.sh`：代码路径与 `CAPTION_SG_PSI`。审计与离线对比见 [results/diag_d0_20261105/AUDIT_PSI.zh.md](../../../../results/diag_d0_20261105/AUDIT_PSI.zh.md)。

- 本目录文件：`build_sg_index.py`, `grpo_sgv2.sh`, `reward_sg.py`
- 叠加顺序：`data/adapter/` → `methods/sg/v1/` → 本目录。
