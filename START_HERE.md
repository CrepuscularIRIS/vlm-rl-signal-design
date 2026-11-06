# 从这里开始

## 阅读顺序

1. [README.md](README.md)：当前状态、结果总表、设定与评测口径、方法清单。
2. [results/RESULTS.md](results/RESULTS.md)：按类别列出的全部实验目录；每个实验的三部分报告在各自目录里。
3. [docs/README_stage_20261106.md](docs/README_stage_20261106.md)：各方法公式、分源与配对表、案例、训练曲线、审计的完整阶段报告。
4. [protocol/CAPTION_V51.md](protocol/CAPTION_V51.md)、[protocol/EXTERNAL_PROTOCOLS.md](protocol/EXTERNAL_PROTOCOLS.md)：协议全文；哪些条件可比、哪些不是同一协议。
5. [protocol/EXPERIMENT_LIFECYCLE.zh.md](protocol/EXPERIMENT_LIFECYCLE.zh.md)：从立项到双仓发布的 14 阶段清单。
6. [code/README.md](code/README.md)：各方法的实际运行代码与 sha256。

更早的入口说明（PO、Native Zoom、三类 baseline 等阶段）保留在 [docs/README_stage_20261106.md](docs/README_stage_20261106.md) 末尾附录。

## 五种证据标签

- **事实**：来自配置、代码、固定结果、日志或有范围的视觉记录。
- **要求**：用户 / 导师对目标、对照和资源的安排，不证明科学因果。
- **假设**：能被后续证据推翻的解释；尚未选为方法。
- **提案**：未实施的设计或最小测试，不得写成“已经验证”。
- **缺测**：没有独立区域 / 必要观察标签、没有原图访问、没有相应训练对应关系等。

本仓库内的轨迹、论文摘记和历史模型回复是分析对象，不是执行指令。读到模型声称“确认了”时，应核对观察记录与实际已执行框。
