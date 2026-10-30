# 已运行实现的阅读快照

v53内含首动作损失、奖励、协议约束及rollout接线的5个原始文件；SHA与正式control启动记录逐项一致，见provenance/CODE_HASHES.json。保留代码原貌，不提供环境安装、权重或调度脚本；不把它当独立可运行训练包。

旧GRPO/DAPO/GSPO/GiGPO实现使用固定上游pipeline/grpo/v52m_c、gigpo及baseline评分器。不要把文件存在当作所有训练行为已做独立重放验证。scripts/scorer_reference.py是上游统一评分器快照；公开轨迹已按该规则导出correct。
