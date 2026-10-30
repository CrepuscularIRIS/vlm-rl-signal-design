# 当前已运行协议

## 数据与评分

E1固定552题，每题4次采样，每条件2208条。来源分母：MME-RealWorld 339题/1356条，MMMU-Pro39题/156条，V*112题/448条，WorldBench62题/248条。样本身份/GT/来源逐键对齐。

统一评分器抽取末轮`<answer>字母</answer>`，再按既定fallback提取；截断、错误、无可解析答案计错，全部保留分母。**不是按每轮协议标记直接清零E1**。`correct_train_rule`是另一种记录口径，不能与E1混用。数字6即使对应某个选项内容，也不会被既定字母提取器自动等价转换。

评分实现固定上游 [summarize.py](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/2c5155d7e629e6eadb043bb8c751035f9132fca9/pipeline/eval/baselines/summarize.py)。本库 `scripts/scorer_reference.py` 为同文件快照。公开轨迹已含correct，不需要重新调用模型。

E1与开发验证/部分训练来源存在交叉；不能称独立盲测。原始图像族全量隔离未在本次新建立，不从qid不同宣称无泄漏。

## 本地六组

底座Qwen3-VL-4B-Instruct，原A-SFT为2epochs。主要RL题池4096题，128题/step×8主rollout，32step，actor minibatch32题、ppo_epochs1、seed42、lr1e-6、max_grad_norm1、无KL。验证仅0/25/32，不能补画不存在的密集点。

| 条件 | 实际训练设置要点 |
|---|---|
| GRPO | caption v5.1；严格协议门后0/1结果奖励；组内标准化 |
| DAPO | 同结果奖励；dynamic filtering .01/.99、clip .2/.28；无新增超长惩罚；实际是项目变体，不等同论文所有组件 |
| GSPO | seq模式，clip .0003/.0004；不是每个裁图独立贡献标签 |
| GiGPO | episode advantage + 1.0×step advantage；步骤分组为项目环境状态移植 |
| RLOO control | 训练v5.2i；R=Y(1-.01N)；RLOO；clip .2/.3；beta0，无额外分叉 |
| RLOO decision | 与control同初始化/设置；beta1/256；两支各K4；abs(D)加权BCE，H0=32 |

v53有1步warmup，首步lr0不证明学习提升。末步32作为这次固定同预算E1检查点；不是按E1选最好checkpoint。旧6k×16/47步、追加40步在历史条件目录单列，不能混入32步算法表。

E1本地六组都使用既定caption v5.1；图像像素上限589824，最多8次裁图。temperature .7，top_p1，seed_base20261017，累计生成8192，上下文32768。训练v5.2i与测试v5.1差异必须保留。

## 成本定义

- 首裁：每条是否至少执行一次crop；后续裁图=max(N−1,0)。
- 处理token：各轮完整prompt（含重复历史和图像）+生成token之和；不是唯一文本长度，也不等于纯视觉token。
- 工具请求文本不等于实际执行；用n_crops/boxes和runner分支判定。
- 训练主rollout、额外fork、验证与存档分别列；嵌套timing不重复相加。

## 可比性

- GRPO/DAPO/GSPO/GiGPO共享主要池与起点，但采样/更新/clip等机制不同，不能把得分差归结为单个advantage项。
- RLOO相对旧GRPO同时变了奖励和训练协议，不能当“仅去std”的纯消融；control与decision才是当前配对。
- 原模型无工具/原生zoom→SFT caption也不是单因素SFT效果。
- 外部7B与本地4B底座/协议/像素预算不同；见 [外部协议](EXTERNAL_PROTOCOLS.md)。
