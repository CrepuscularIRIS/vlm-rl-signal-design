# 可复算范围

本仓库不需要GPU、权重、SSH或模型API。

```bash
python -m pip install -r requirements.txt
python scripts/verify_bundle.py
python scripts/recompute_pairs.py --out recomputed
```

recompute_pairs读取11个主要公开条件，验证每条件2208条、552题、sample0–3、GT和来源一致，重算题级/来源得失与固定checkpoint下按题bootstrap区间。输出到recomputed，不覆盖冻结results。bootstrap 5000次、随机种子20261029，不代表训练multi-seed，区间不含训练seed不确定性，也未校正多重比较。

原始训练ledger/所有图像不在本库，所以仅凭此库不能重放训练梯度、独立判定裁图内容或验证全部GT。training_signals.json是先前CPU汇总结果，不能从E1反推。图片图表是实际数据绘制；验证曲线仅0/25/32三个记录点。

GitHub MCP读取可用Markdown病例与CSV，不必运行脚本。public仓库可通过GitHub和raw内容读取，但本次未代用户验证其具体网页Pro会话的MCP账号权限。

新增训练聚合可重画：`python scripts/plot_algorithm_curves.py --out recomputed/curves`。输入为公开192行逐步聚合和18个验证点；可核对合计并重画，不代表独立复核私有原始训练日志。旧冻结分数与轨迹不改。

## DTPO 与全条件总览

从仓库根执行 `python results/dtpo_e1_20261103/plot.py`。九组训练曲线、全条件取舍和 DTPO 逐步诊断均从同目录 CSV 重画。私有逐轨评分的输入指纹与公开聚合表分列；基线原始验证器的 25 条件、96 条审查计数只管其原有范围。[结果与全部图表](results/dtpo_e1_20261103/RESULTS.zh.md) · [流程与发布验收](protocol/EXPERIMENT_LIFECYCLE.zh.md)
