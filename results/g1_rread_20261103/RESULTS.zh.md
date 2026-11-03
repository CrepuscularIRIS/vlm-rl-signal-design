# G1 (R-read) 训练报告（三部分格式）

2026-11-03 | run: `outputs/d2_g1_rread`（attempt 3）| GPU 2,3 | 32 步全程 5h51m

## 一、设定

| 项 | 值 |
|---|---|
| SFT 起点 | A-SFT (`sft_v22_turnwise_merged`, E1 .5670)，**本轮无 SFT 改动** |
| RL | 无过滤 GRPO（adv_estimator=grpo，保留全错/全对组），clip .2/.3，lr 1e-6 const，无 KL，seed 42 |
| 奖励 | **R = V·(Y + 0.25·m·Φ_read)**（冻结 `design/R_READ_FREEZE`） |
| Φ_read | commit 文本（最后 `</grounding>` 后）读数 vs L2 读取索引（2,699 题 m=1）规范化精确匹配；缺失/枚举/否定=0；每字段一次 |
| 池/步数 | 4k 池 × 8 rollouts，128 题/步 × 32 步，mini 32 |
| 唯一变量 vs G0 (k8 GRPO) | 奖励加 0.25·m·Φ_read 项；其余逐项相同 |
| 事故 | attempt1 phi 日志半改 NameError；attempt2 `text` 循环泄漏（步 1–2 奖励污染作废）；attempt3 干净全程。step1 验收 phi=0.5579 对齐离线探针 0.5518 |
| 中途 | 无暂停（bsftc 侧减载时 G1 独跑） |

## 二、定性 + 定量

### Case: mme_14131（门牌号读取，字段真值 3840）

**A-SFT（训练前）**，4 步后 commit：
> The plaque … visible digits read **3, 4, 7, 0**, matching option C … answer is 3470.
> `<answer>C</answer>` → **错**；该题 Φ 均值 0.25

**G1-32（训练后）**，2 步后 commit：
> The crop of the house doorway shows the number plate clearly reading **"3840"** above the right door panel …
> `<answer>B</answer>` → **对**；该题 Φ 均值 0.88

### 定量

| 指标 | G1 (R-read) | G0 (k8 GRPO) | Δ |
|---|---:|---:|---|
| **E1（552×4）** | **.6395** (1412/2208) | .6390 | +0.05pp（净 +1 题） |
| 逐题配对 | 仅G1对 256 | 仅G0对 255 | 对称，共同对 1156 |
| 裁图/条 | 2.308 | 1.905 | +21% |
| E1 分源 | mme .6645 / vstar .8527 / wb .3266 / mmmu .3077 | 同量级 | — |

**同题同设置读取探针**（256 题 × 8，与准入探针完全同题同参数；题属训练池）：

| | A-SFT | G1-32 |
|---|---:|---:|
| Φ_read 均值 | .5518 | **.6911**（配对 Δ+0.139，750↑/140↓/1158 持平） |
| 写出≥1 正确读数 | 81.1% | 88.1% |
| 同池终答正确率 | ~.79 | .7983（持平） |

**判读（按冻结协议预登记行）**：训练读取分↑ + 训练池同题读取↑ + 端到端 E1 不变 → 落在「读取改善未转化」行。注意该探针题 G1 训练过且含选项，**去选项内容变化测试未跑**，不能排除训练池内的表达适配；留作下一步判别。

## 三、训练 log 曲线

图：`curves/g1_curves.png`（四联：train accuracy / mean reward / policy entropy / Φ_read+crops）
- reward 均值 0.666 → ~0.75（含 Φ 加成；G0 同期 ~0.60）
- Φ_read（m=1）step1-8 均值 ~0.55 → step25-32 ~0.65（明确上升）
- entropy（−entropy_loss）与 G0 同降趋势，无异常塌缩
- ledger：step8 组构成 all_correct 74 / all_wrong 10 / mixed 44，adv_nonzero_tok **0.821**（纯 Y 下 59% 组本会零优势）
