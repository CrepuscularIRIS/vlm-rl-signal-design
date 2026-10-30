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
