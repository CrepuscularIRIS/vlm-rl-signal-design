# 重审：11-03 以来 oracle 探针 / V* 分解 / G1 audit 链（2026-11-04）

起因：用户要求亲自判断模型裁图“是裁对了读不出，还是裁错了”，并全面重审标注后的一系列判断。

## 1. Probe A 计分错误（DAPO 与 G1 两次均受影响）

- 代码：`obs/notes/oracle_probe_20261103/run_probe.py:77`、`run_probe_v2.py:56`
  `lm = re.search(r'\b([A-J])\b', text)`；计分 `probe_answer == gt`。
- Probe A 的输入只有裁图 + “Transcribe what you see”，**不给题面和选项**（`build_probe_g1.py:51-53`）。模型输出的是自由描述，正则从中取第一个独立大写字母当“选项”。
- 复核命令（输出见下）：
  ```
  python3 -c "import json,re;rows=[json.loads(l) for l in open('<probe_a_results.jsonl>')];..."
  ```
  | 运行 | n | 计为命中 | 命中处上下文 |
  |---|---:|---:|---|
  | DAPO（`vlm-rl-signal-design/results/oracle_probes_20261103/probe_a_results.jsonl`） | 180 | 15 | 14 条为冠词 “A …”（“A green street sign”“A man in…”），1 条为句中 “E” |
  | G1（`obs/notes/oracle_g1_20261103/probe_a_results.jsonl`） | 180 | 14 | 抽查 5 条均为冠词 “A …” |
  - 抽取结果分布：DAPO `None` 94 / `A` 63；G1 `None` 98 / `A` 56。
- 事实：8.3% 与 7.8% 不是读取准确率，是正则命中冠词 “A” 且 gt 恰为 A 的比例。

## 2. Probe B 测到的是什么

- 输入：题面 + 模型自己错答轨迹 `<think>` 内的文字（含其结论）。
- G1：probe 答案 == 模型原错答 167/180，== gt 5/180；180 条笔记中 63 条显式写了选项字母。
- 事实：Probe B 的 1.7% / 2.8% 主要反映“笔记与原答案一致”，不是独立的推理测试。

## 3. 24 条错答裁图逐张人工核对（G1 Probe 样本，`random.Random(7)`，每源 8）

材料：`obs/notes/oracle_g1_20261103/audit_crops_*.png`（= Probe A 送入的“末裁图 +25% 边距”），`audit_sample.jsonl`（题面、gt、pred、commit）。

| 类 | 判据 | 条目 | 数 |
|---|---|---|---:|
| 末裁图不含答题所需内容 | 我看该图也答不了 | 00 鸭子广告牌（末裁为左下角路面，8 次裁图）、05 轮胎、10 花数（只含右半）、11 左墙肖像（末裁为右侧单幅）、13 曲线峰值（无时间轴）、14 座位排数、20 漫画主旨（只含两格说明） | 7 |
| 内容清楚、模型读对，空间指代/顺序/绑定错 | commit 中读值正确 | 07 读出 “PROTOTYPE 003” 与 “LAND ROVER 70TH EDITION” 两块牌，把左侧手动挡上方的牌当成“lever 下方”；02 “从下往上第三行”裁到上部；01 “左侧穿黑裤行人”指代 | 3 |
| 内容清楚、模型读对，选项匹配错 | commit 读值 = 图中值，答案字母错 | 03 读出 $1,790,441，选 B(1750441)；选项 D 为 1790442（与图差 1）；06 读出 “樓會慶”，写“matches option B”，B 为 “樓慶會”、D 为 “樓會慶” | 2 |
| 内容清楚、模型读错 | 我能读出、模型读出不同值 | 16 电容 “¼F” 读成 4F；17 阴影图形顶点读成 (0,0)-(5,5)（实为 (1,0)(3,5)(5,0) 减 (1,0)(3,2)(5,0)，面积 6） | 2 |
| 内容清楚、读对，计算/验证错 | | 12 27° 行读对，判为错误（数值验算 cos27°≈0.891 与表达式一致） | 1 |
| 依赖领域知识/估计 | 裁图内容可见，答案需外部知识或主观判断 | 04 驾驶决策、08 鹿描述、09 窗户方向（题意不清）、15 乌贼长度估计、18/22/23 病理切片、19/21 白蜡树病因 | 9 |

- 末裁图 ≠ 证据裁图：多裁轨迹（如 00 共 8 次）的末裁不一定是看证据的那一张；Probe A 只用末裁。
- 数据：03 的正确选项值（1790442）与图中值（1,790,441）不一致。

## 4. V* “看到但用错”复算（`results/rereview_20261104/scripts/vstar_recheck.py`）

```
python3 obs/notes/rereview_20261104/vstar_recheck.py obs/runs/baselines_v52_20261024/full/traj_k8_grpo.jsonl obs/runs/rread_adm_20261103/data/e1_g1_rread/traj_g1_rread.jsonl
```
| 条件 | 错答 | 命中（单裁 ≥90% 覆盖每个 GT 框） | 命中裁图 ≥25% 全图面积 |
|---|---:|---:|---:|
| k8 GRPO | 70 | 38 (54.3%) | 4 |
| G1 | 66 | 40 (60.6%) | 6 |

数值复现。命中定义不含物体在裁图中的尺度；按“GT 框长边 / 命中裁图长边”：

| 条件 | 组 | 命中 n | 中位数 | <5% 占比 |
|---|---|---:|---:|---:|
| k8 GRPO | 答对 | 338 | 0.193 | 9% |
| k8 GRPO | 答错 | 38 | 0.066 | 37% |
| G1 | 答对 | 349 | 0.167 | 15% |
| G1 | 答错 | 40 | 0.038 | 60% |

G1 命中答错 8 条渲染（`vstar_hit_*.png`，红框 = GT，`vstar_hit_render.py ... 8 7`）逐张看：
- 物体极小（8×12 px 汽水罐在 562×375 裁图；26×27 轮椅在 1238×450 裁图）：00、03
- 物体清楚，属性绑到邻近物体：06 扫帚（黑刷头灰杆，答 yellow——旁边有黄色管件）
- 物体清楚，commit 写了 “green striped” 却答 white：07 围巾
- 两物体均可见，左右/上下关系判反：01、02、04、05

## 5. 受影响主张的状态

| 主张 | 出处 | 状态 |
|---|---|---|
| DAPO 错答 = 搜索 46% + 读取 43% + 推理 8% + 绑定 3% | `vlm-rl-signal-design/results/oracle_probes_20261103/RESULTS.zh.md`、`design/PERCEPTION_GAP_BRIEF.zh.md`、README:39 | 读取 43% 依据为 Probe A/B（§1–2），无有效测量 |
| 4B 给了单图也读不对（8.3%） | 同上 | 计分错误，无效 |
| G1 Φ 上升 = 表达适配/回填实锤；感知与推理两层未移动 | `results/oracle_g1_20261103/RESULTS.zh.md`、`results/RESULTS.md:5` | 依据为 Probe A，无效；Φ .5518→.6911 与 E1 .6395 持平这两个测量本身不受影响；回填与否未测 |
| 错答中 54%（G1 60.6%）“看到但用错” | 同上、skill `evidence-gap-diagnosis` | 数值复现；“看到”=覆盖，不控制尺度（§4） |
| RL 把搜索失败从 82% 压到 46% | 同上 | 同一覆盖定义，尺度未控制；未复算 |
| R-bind 可行性 11.5% / 19.1%；风险 3 引 “Probe A 8%” | `design/R_BIND_DIRECTION.zh.md` | 同一覆盖定义；风险 3 的引用无效 |
| V* 看到但用错中 71% 方向反转 | oracle_probes RESULTS | 未复算 |
| G1 R-read 的立项依据（“读取层 43%”） | `PERCEPTION_GAP_BRIEF` → `R_READ_FREEZE` | 依据无效；G1 训练与评测数据本身有效 |

## 6. 未复核（本轮没做）

- L1/L2/LC 标注质量本身（框位置、转录值）未逐图抽查。
- Hτ / E 系列的结论未重审。
- 修正后的 oracle-view（L1 GT 框裁图 + 题面选项 + 按选项计分）未跑。

## 产物

`vstar_recheck.py`、`vstar_hit_render.py`、`vstar_hit_*.png`、`vstar_hit_meta.json`、`vstar_uid_q.json`（服务器 `vstar_holdout.parquet` 提取）、`vstar_holdout.parquet`。
