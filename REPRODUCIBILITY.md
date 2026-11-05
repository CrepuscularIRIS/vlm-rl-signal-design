# 可复算范围

## 11-05 max2新增范围

[最新报告](results/sgv2_max2_20261105/REPORT.zh.md)。`python results/sgv2_max2_20261105/verify_delivery.py` 复核六条件各2208条匿名正确性、552题各4次、总体/分源四格、按题10000次bootstrap和文件SHA；该复用验证器保留strat256行专检，并对全部family行检查键唯一与数值有限。max2 355行及dev 0/25/32点另经私有原始日志回归测试（1 passed）；不将其写成公开原始日志复评分。`python results/sgv2_max2_20261105/plot_results.py --data results/sgv2_max2_20261105` 重画训练与E1全部图（需现有numpy/matplotlib）。

公开包不含原始响应、题面、原图或权重。训练成本口径为32个step计时之和，包含计时内验证/保存，不是完整run墙钟；缺原训练退出码及逐rank更新事件。文件指纹见同目录`FILES.json`，私有输入指纹和重放边界见`PROVENANCE.json`。

## 11-05 strat新增范围

`python results/sgv3_20261105/verify_delivery.py` 核对四条件各2208条匿名正确性、552题各4次、总体/分源四格及按题bootstrap（10,000次，seed20261105）、256行strat训练指标与文件SHA。`python -m pytest results/sgv3_20261105/test_compare.py -q` 检查四格函数；`python results/sgv3_20261105/plot_delivery.py` 重画全部family及E1成本图。

[报告](results/sgv3_20261105/REPORT.zh.md) · [指纹](results/sgv3_20261105/PROVENANCE.json)。这不替代私有原始响应重评分、训练梯度或逐图重放。下方旧25条件验证器只覆盖原冻结范围。

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
