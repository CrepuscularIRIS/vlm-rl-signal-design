# 代码

按导师 2026-11-05 的建议整理成五个目录。快照原则不变：保留实际运行时的原貌，只把机器路径替换成别名（`$PROJECT`、`$ROOT`、`$EASYR1_DIR`、`$OBS`、`$HOME`、`$HF_HOME`）。每个文件替换别名**之前**的 sha256 记在 [MANIFEST.json](MANIFEST.json)。

```
code/
├── datasets/            数据与标注：rl4k 题池说明、L0/L1/L2/LC 标注流水线
├── data/                loader 与 runner：EasyR1 适配层（数据、多轮 rollout、奖励装配）和 EasyR1 运行时源码
├── scripts/             实验启动与监控脚本（公共参数、两卡槽位、内存守卫）
├── eval/                评测：E1 runner 与统一评分器、caption v5.1 协议
└── methods/             各方法，每个版本一个子目录（v1、v2…）
    ├── grpo/v1  dapo/v1  gspo/v1        只改启动参数，没有额外代码
    ├── gigpo/v1  rloo/v1  dtpo/v1  sapo/v1
    ├── htau/v1  g1_rread/v1
    ├── sg/v1  sg/v2                     v1：A / A+B / A+B+C / GRPO+A；v2：导师的 psi（逼近 / 探索）
    └── dapo_sb/v1
```

**叠加规则**：一个方法版本 = `data/adapter/` 的全部文件，再用 `methods/<方法>/<版本>/` 里的同名文件覆盖。方法目录里只放与公共适配层不同的文件和它自己新增的文件；与 `data/adapter/` 逐字节相同的副本已去掉，每个方法的 README 列出了被去掉的文件。`sg/v2` 叠加在 `sg/v1` 之上。

**未收录**：模型权重、数据 parquet、原图、标注派生索引（`sg_index*.json`、`rread_index.json`），以及环境安装脚本。这些代码不是开箱即用的训练包：运行需要 EasyR1（verl）检出目录、vLLM 0.11、caption 数据集与原图。

各方法的公式、函数位置与启动覆盖项见 [methods/README.md](methods/README.md)。
