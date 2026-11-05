# SG 双臂冻结记录（Design_New 算法，2026-11-04）

用户指示（2026-11-04）：按 `papers/Design_New.md`（Pro）的算法做两条训练：**A+B** 与 **A+B+C**，按规范自动推进，训练后做 audit。

## 唯一变量与对照

| 臂 | tag | 卡 | 奖励 | advantage | 数据 |
|---|---|---|---|---|---|
| A+B | `sg_ab` | 0,1 | R = V·(Y + λ·m·S) | task-scaled（§4B） | 4k train.parquet，shuffle（seed 42 RandomSampler，与 k8 GRPO 同序） |
| A+B+C | `sg_abc` | 2,3 | 同上；续答行 R = V·Y | 同上 | `grpo_caption_v51_4k_sgc`：同一顺序预排，384 行替换为续答行，shuffle=false |
| 对照 | k8 GRPO | — | V·Y | 标准 GRPO | 4k，E1 .6390（复用） |

- A+B 对 k8 GRPO：差异 = 奖励中的 S 项 + advantage 归一方式。
- A+B+C 对 A+B：差异 = 384 行续答行（前 8 步每步 32 行，第 9–16 步每步 16 行，之后 0 行）。续答行所用的题目取自题序尾部，因此每道题在每条臂中只出现一次。

其余与 k8 GRPO 逐项相同：A-SFT 起点（`outputs/sft_v22_turnwise_merged`）、n=8、128 题/步、mini 32、32 步、lr 1e-6 const、无 KL、clip .2/.3、无过滤、seed 42、val 4k val_nosft（训练前 + step 25）、save_freq 5、保留 2。

## A：尺度门控获取分 S（`obs/src/grpo_sg/reward_sg.py`）

- 定义见文件头；v0（原图）计入，裁图尺度 s(v) 按 `caption_env.present` 与 `max_pixels=589824` 计算。
- 常数：λ=0.1，α=0.25，门阈值 U=0.5，覆盖 0.9。尺度函数 h 为**未校准代理**：文字（行高）6→14 px 线性上升，物体（短边）16→48 px；context 节点（membership_context / main_figure）只看覆盖。
- 资格 m=1：D2 reason=enabled、L1 unit 不是 absent/region_global、L1 置信度不为 low、至少一组 evidence_set 成员齐全且没有 readable=false。共 3,014/4,096 题。
- 自检：`python reward_sg.py` 共 11 条断言，全部通过（本地与服务器均跑过）。

## B：task-scaled advantage（`easyr1_caption.install_task_scaled_advantage`，CAPTION_ADV_TS=1）

A_i = ((y_i − ȳ) + (s'_i − s̄')) / max(std(y), 1/√G)，其中 y = 1[R ≥ 1]，s' = R − y（λ·S ≤ 0.1 < 1，因此可以精确分解）。λ=0.1 < 1/(G−1)。
合成批测试：混合组符号不翻转；全错组的值等于 λ(s−s̄)/σ_min（最大 0.1768）；全对且 S 相同的组 advantage 为 0。

## C：证据续答行（`build_cont_pool.py`）

- 视图：取第一组 evidence_set 的并集框（每边外扩 10%）；若任一节点 a_b<0.5，则改为每个节点各一个视图（最多 3 个）。注入一步合成的 locate 回合（prompt token，不进 loss），模型从第 2 步接着答。
- CONT_MANIFEST：384 行（单视图 380，二视图 1，三视图 3）；从题序尾部跳过 158 题（无资格 134，构造不出视图 24）。
- 冒烟：前缀两次编码一致、`_fresh_env` 保留注入视图、actor 重渲染 grid 一致（3/3）；3 张注入视图人工看过，证据可读。

## 离线准入（`sg_admission.py`，A-SFT 真实轨迹 256 题 × 8，rl4k 池）

| 指标 | 值 |
|---|---|
| 资格题 / 轨迹 | 253/256；2,024 |
| S 均值；S>0 比例 | 0.123；24.5% |
| 门在 v0 已开 / 终态开 | 69.8% / 80.7% |
| 无任务信号组中 S 有组内差异 | 46/114 = 40.4%（准入线 10%） |
| 混合组 P(S_对 > S_错) + 平局/2 | 0.589 |
| 攻击：追加整图裁剪 | ΔS 均值 +0.0000 |
| 攻击：8 格平铺 | ΔS 均值 +0.164，40.5% 的轨迹 ΔS>0 |

预登记监控项：平铺类多裁图会提高 S（设计中不加裁图惩罚），训练中逐步记录裁图数。

## 版本

code_sha16 d9e6396720e75a1b（grpo_sg/*.py）；sg_index 1830cfd0；reward_sg 022b900a；easyr1_caption 81284bdf；caption_contracts 2c039c51；sgc train.parquet 4c81b867；4k train.parquet 26964faf。

## 磁盘与内存（启动前）

- 服务器上 4 个 ckpt 目录（d2_ht5、dtpo_f1102a2、po_spr1031a1、po_car1031a1）的 HF 权重已备份到本地 本地备份盘。rsync dry-run 差异为 0 后删除，可用空间 215G → 557G。
- cgroup 上限 440GiB；启动前 used 26G。

## 启动与首步验收（2026-11-04 02:12 CST 启动）

| 臂 | m=1 占比 | S 均值 | S>0 | 门控开（终态 / v0） | acc | 裁图/条 | 有效终答 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sg_ab step1 | 74.8% | 0.106 | 18.9% | 86.2% / 76.0% | .572 | 0.71 | .992 |
| sg_abc step1 | 58.3% | 0.109 | 19.8% | 83.8% / 73.4% | .598 | 0.84 | .994 |
| 离线准入（A-SFT） | — | 0.123 | 24.5% | 80.7% / 69.8% | — | — | — |

- sg_abc 的续答行：256 条，S 全部为 0，acc .668，裁图 1.38/条（**含注入的 1 张**），有效终答 .992。
- advantage 补丁：两臂日志中 `adv_estimator grpo -> task-scaled` 均出现。ledger 中 abc call1：组构成为全对 48 / 全错 13 / 混合 67，adv 非零 token 占 0.737（标准 GRPO 下退化组为 0，约 52%）。
- 判定：奖励路径实值与离线准入同量级，进入正式训练。

## 训练结束与事故记录

- 两臂均 32/32 步、32 个更新事件、ckpt 32 完整，日志 Traceback 0。sg_abc 约 11:50 CST 结束，sg_ab 约 11:57 CST 结束。
- 守卫误报：sg_abc 正常退出后被守卫判成 DEAD（守卫先检查进程存活、后检查 ckpt 32，顺序写反了），不是真实故障。
- **sg_abc 第 31–32 步的题目构成偏差（构造副作用）**：续答题从题序尾部选，被跳过的 158 道不合格题留在尾部，挤进了最后两步。m=0 的题数：第 31 步 49/128，第 32 步 109/128（不可答 57、全图 24、GT 分歧 14 等）。同一批题上 k8 GRPO 的正确率为 .566 / .318，sg_abc 为 .560 / .311。因此这两步 sg_abc 的原题集合与 sg_ab 不同，A+B+C 对 A+B 的唯一变量在第 31–32 步不成立。E1 评测用的是 step 32 终点权重，这两步的更新已经包含在内。
- 曲线：`data/sg_curves.{png,pdf}`、长表 `data/sg_curves_long.csv`、重画脚本 `data/plot_sg_curves.py`。日志 reward 与账本重算 reward 逐步核对，32/32 步 × 2 臂差值均 ≤0.01。
- 附带发现：G1 v2 主图（`rread_adm_20261103/data/curves/plot_g1_v2.py`）的熵面板对 `actor/entropy_loss` 取了负号，而该字段本身就是平均 token 熵（为正值）。图的形状不变，只是符号反了，待修正。

## 单 A（sg_a），2026-11-04 14:31 CST 启动，GPU 0,1

- 依据：用户 11-04 指示（GRPO 补一个单 A；DAPO 主线另做）；Design_New 第二轮 §一、§六。
- 与 A+B 的唯一差别：`CAPTION_ADV_TS=0`，即标准 GRPO 归一 A = (R − mean)/(std(R) + 1e-6)，R = V·(Y + 0.1·m·S)。S、λ、索引、题序、其余超参与 sg_ab 完全相同。
- 代码：服务器目录 code_sha 显示为 802fd1d4，与 d9e63967 不同，原因是 10:25 上传的统计脚本 `step_stats.py` 也被计入；去掉它后拼接哈希为 d9e63967。逐文件：reward_sg 022b900a / easyr1_caption 81284bdf / caption_contracts 2c039c51 / sg_index 1830cfd0 / grpo_sg.sh 2fe21730，与 sg_ab 一致。
- evidence_sets fallback：m=1 的 3,014 题中 0 题退回使用全部 L1 框。
- 首步验收：第 1 步权重和采样种子与 sg_ab 相同，rollout 账本应与 sg_ab 第 1 步逐行一致（只有 advantage 不同）。
- sg_a 首步验收（14:5x CST）：第 1 步 1,024 行 rollout 与 sg_ab 第 1 步在 qid / terminal / crops / correct / gen_tokens / s / m / boxes 上逐行一致；日志中 task-scaled 补丁安装行数为 0。
