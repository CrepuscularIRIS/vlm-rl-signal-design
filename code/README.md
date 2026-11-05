# 代码：每个方法实际运行的实现快照

这里收录 README 中 12 种 RL 训练、E1 评测和 rl4k 标注实际使用的源码，按方法分目录。

- **快照原则**：保留原貌，只把机器路径替换成别名（`$PROJECT`、`$ROOT`、`$EASYR1_DIR`、`$OBS`、`$HOME`、`$HF_HOME`；本地服务 URL 写成 `http://localhost:PORT`）。
- **哈希**：每个文件在替换别名**之前**的 sha256 记在 [MANIFEST.json](MANIFEST.json)。其中奖励文件的哈希已和训练记录逐一核对：Hτ `reward_d2.py` dc8fbbb4；G1 `reward_rread.py` 22b28fc3、`caption_contracts.py` 39157c10；A+B `reward_sg.py` 022b900a、`easyr1_caption.py` 81284bdf、`caption_contracts.py` 2c039c51。
- **未收录**：模型权重、数据 parquet、原图、标注派生索引（`sg_index.json`、`rread_index.json`），以及环境安装脚本。
- 这些代码不是开箱即用的训练包：运行需要 EasyR1（verl）检出目录、vLLM 0.11、caption 数据集与原图。

## 公共框架：EasyR1 适配层

所有方法都通过 `easyr1_caption.py:install()` 把 caption 多轮协议接进 EasyR1。接入点有三处：

| 接入点 | 函数 | 作用 |
|---|---|---|
| 数据 | `caption_getitem` | 用与 rollout 完全相同的原图呈现规则构造首轮 prompt |
| rollout | `generate_sequences_multiturn` | 多轮循环：每轮 vLLM 生成 → `proto.parse_step` / `check_step` 硬检查 → `CropEnv.step` 裁图 → 回执和新图追加到 prompt；最多 8 次裁图 + 1 次作答 |
| 奖励 | `caption_contracts.assistant_reward_batch` | 计算 V·Y（以及各方法的过程项），奖励放在最后一个 assistant token |

其他公共文件：
- `caption_env.py`：裁图与呈现尺寸规则（56 px 最小裁图、短边 448、放大上限 4×、面积上限 1400²、对齐 28）。
- `reward_caption.py`：Y 的规则（恰好一个 `<answer>X</answer>` 且 X 等于 gt）。
- `update_ledger.py`：每步记录组构成（全对 / 全错 / 混合）和非零 advantage 占比。
- `easyr1_patches.py`：GSPO 在 seq 模式下按序列数给 micro-batch 加权；空 chunk 过滤。
- 训练时每条轨迹写一行 rollout 账本（`CAPTION_ROLLOUT_LOG`）。

## 各方法目录

| 方法 | 目录 | 核心公式所在 | 启动方式（相对 GRPO 的覆盖项） |
|---|---|---|---|
| GRPO | [common_grpo_v52m_c](common_grpo_v52m_c/) | advantage：[easyr1_runtime/core_algos.py](easyr1_runtime/core_algos.py) `compute_grpo_outcome_advantage`（L176–216）；clip loss 在同文件 `compute_policy_loss`（L410） | `slot2.sh`：`run k8_grpo grpo_v52m_c.sh $SFT 2,3 $U2`；公共参数见 `common.sh` |
| DAPO | 同上 | 过滤：[easyr1_runtime/ray_trainer.py](easyr1_runtime/ray_trainer.py) L518–556（组均值 ∈ (filter_low, filter_high) 才保留，不够就补采） | `$DAPO`：`online_filtering=true filter_key=overall filter_low=0.01 filter_high=0.99 max_try_make_batch=100 clip_ratio_high=0.28`，配 `$U1` |
| GSPO | 同上 | 序列级 ratio：`core_algos.py` 中 `loss_type=gspo` 分支（`compute_policy_loss` L410 起，gspo 分支 L465–480：`s_i = mean_t(logπ−logπ_old)`）；`average_loss` L383；seq 平均 | `$GSPO`：`loss_type=gspo loss_avg_mode=seq clip_ratio_low=0.0003 clip_ratio_high=0.0004` |
| K-base GRPO | 同上 | 同 GRPO | 起点换成 Qwen3-VL-4B-Instruct |
| GiGPO | [gigpo](gigpo/) | `gigpo.py:step_advantages`（L23）：锚点 = (uid, 已收视图, 已执行裁图框, 已用裁图数, 步序号)，组内对终奖励做 z-score；`install`（L47）把 A_step×w 加到 GRPO 的 A_episode 上；单测 `test_gigpo.py` | `CAPTION_GIGPO=1`，`CAPTION_GIGPO_STEP_W` 默认 1.0，launcher `grpo_gigpo.sh` |
| RLOO / RLOO + 首动作分叉 | [rloo_v53](rloo_v53/) | RLOO：`core_algos.py` L268 起；奖励 Y·(1−λN)：`reward_caption.py`；分叉监督：`decision_loss.py`（`select_slot` L69 选主 rollout，`find_boundary` L75 定位首动作前缀，`decision_gap` L53 = mean R_answer − mean R_crop，`pair_loss` L57 = \|D\|·BCE，`install_actor_patch` L155 把辅助损失加到 actor）；训练协议 v5.2i：`caption_v52i.py` | `launch_formal.py`：`LAMBDA=0.01`，`BETA=0`（control）/ `0.00390625`（decision），`K=4`；launcher `grpo_v53_dl.sh` |
| DTPO | [dtpo](dtpo/) | `dtpo_adapter.py`：`turn_mask` / `partitions`（L16–21）把 assistant token 分成中间段 I 和最终段 F；`update_policy`（L62）计算 L = 0.5·mean_I ℓ + mean_F ℓ；`token_loss`（L26）的 clip 为 [0.8, 1.24]。官方实现参照 `dtpo_official.py`（含 `OFFICIAL_LICENSE`） | `run.sh` / `launch.py` |
| SAPO | [sapo](sapo/) | `sapo_adapter.py`：`add_boundaries`（L19）标出每轮动作边界；`capture_forward`（L49）用共享 LM head 读 V/Q（FIM token 151659/151661）；`compute_advantage`（L104）做 turn 级 GAE（γ=λ=0.95）；`update_policy`（L132）= PPO + 0.1·V loss + 0.1·Q loss。官方 `sapo_official.py:compute_arq_gae_targets`（L8） | `run.sh`：`rollout.n=1` |
| Hτ（bbox 覆盖过程奖励） | [htau_box_coverage](htau_box_coverage/) | `reward_d2.py:process_score`（L64）：只计“框占裁图面积 ≥ τ=0.005”的裁图，Φ = mean_j min(1, 0.5·(maxIoU_j + 并集覆盖_j))；`caption_contracts.py:assistant_reward_batch`（L147）组装 V·(Y+0.25·m·Φ) | `grpo_d2.sh` + DAPO 过滤（`filter_key=accuracy`），clip high 0.28 |
| G1（读值一致性过程奖励） | [g1_reading_consistency](g1_reading_consistency/) | `reward_rread.py`：`commit_text`（L68）取最后一次 `</grounding>` 之后的文本；`match_num` / `match_txt`（L38 / L52）做规范化精确匹配，排除枚举和否定；`process_score`（L74）= 命中字段比例。冻结评分器 `rread_scorer.py`（成对反例自检 `_selfcheck`，L124）；离线准入 `run_admission.py`、`admission_stats.py` | `grpo_r1.sh` |
| A+B / A+B+C（尺度门控证据奖励 + 任务尺度 advantage + 证据续答行） | [sg_scale_gated](sg_scale_gated/) | 见下一节 | `grpo_sg.sh`；A+B：`CAPTION_ADV_TS=1 CAPTION_SG_LAMBDA=0.1`；A+B+C 另加 `TRAIN=…_4k_sgc/train.parquet data.shuffle=false` |
| GRPO+A（单 A：只加 S 奖励） | [sg_scale_gated](sg_scale_gated/) | 同 A+B 的 S（`reward_sg.py`），advantage 用标准 GRPO | `grpo_sg.sh`：`CAPTION_ADV_TS=0 CAPTION_SG_LAMBDA=0.1` |
| DAPO-SB（停止相对 advantage） | [dapo_sb](dapo_sb/) | 探针：`stop_baseline.py:build_probe_request`（L33；以 `<answer` 结尾读 `>X`）、`combine`（L62；P(X)=P(`>X`)+P(`>`)·P(X\|`<answer>`)）、`run_probes`（L79；每个不同状态 2 次请求，max_tokens 1、temperature 0、top-20 logprob）；advantage：`stop_relative_advantage.py:build_advantages`（L31；Pro 参考实现原样，δ=b[k+1]−b[k]，终止项 Y−b[-1]，λ 递推，除以组内 Y 的样本标准差）、`broadcast_to_tokens`（L96）；`stop_baseline.py:install_advantage`（L127）在 DAPO 的 compute_advantage 之后替换 advantage；rollout 接线 `easyr1_caption.py` L378–414 | `grpo_sb.sh` + `$DAPO` 覆盖项：`CAPTION_SB=1 CAPTION_SB_LAMBDA=1 CAPTION_SG_LAMBDA=0 CAPTION_ADV_TS=0` |

## A+B / A+B+C 的实现细节（[sg_scale_gated](sg_scale_gated/)）

| 部件 | 位置 | 说明 |
|---|---|---|
| A：S 的计算 | `reward_sg.py` | `view_scale`（L42）：原图像素 → 最终模型输入像素的倍率（依次走 `present()` 和 `max_pixels`）；`_h`（L64）：未校准的线性尺度代理；`_psi`（L90）：逐框计算 z_b，在 evidence_sets 上取 Φ，按门控分段得到 Ψ；`process_score`（L112）：S = Ψ_T − Ψ_0；`_selfcheck`（L126）：11 条反例断言（不裁图、整图裁剪、只裁到部分证据、补齐证据组合、一次大裁 vs 多次小裁、远离目标的放大探索、原图已能看清的题等） |
| A：资格与证据组合 | `build_sg_index.py` | 由 L1/L2 和资格索引生成每题的节点（bbox / kind / 行数 / 是否关键框）、`sets`、`m` |
| A：奖励装配 | `caption_contracts.py:assistant_reward_batch`（L152） | R = V·(Y + λ·m·S)；续答行（`caption_cont_views > 0`）只给 V·Y；rollout 账本额外写入 s、gate、phi、explore、执行的框 |
| B：advantage | `easyr1_caption.py:install_task_scaled_advantage`（L105） | 把 R 精确拆成 y = 1[R ≥ 1] 和 s′ = R − y（λS ≤ 0.1 < 1，所以可以精确拆分）；A = ((y−ȳ)+(s′−s̄′)) / max(std(y), 1/√G)；替换 EasyR1 的 `ADV_ESTIMATOR_MAP["grpo"]` |
| C：续答前缀 | `easyr1_caption.py:encode_prefix`（L69） | 对续答行注入合成的第 1 步 locate 回合和对应回执图，经过与真实裁图相同的 `parse_step` / `check_step` / `CropEnv.step`；数据侧（`caption_getitem`）和 rollout 侧共用这一个函数，保证前缀逐 token 一致；之后模型的步号从 n_pre+1 开始校验 |
| C：数据构造 | `build_cont_pool.py` | 题序 = `torch.randperm(4096, seed 42)`，与 EasyR1 RandomSampler 相同；续答题从题序尾部挑，GT 视图 = 第一组 evidence_set 并集框外扩 10%，否则改为每个框一个视图（最多 3 个）；SCHEDULE = 第 1–8 步每步 32 行，第 9–16 步每步 16 行 |
| 冒烟与验收 | `smoke_cpu.py`、`sg_admission.py`、`step_stats.py` | 前缀两次编码一致、actor 重渲染的 grid 一致、advantage 合成批断言；离线准入（S 方差、混合组判别、整图 / 平铺攻击）；逐步统计 |

## 评测与评分（[eval_e1](eval_e1/)）

- `run_eval.sh caption TAG MODEL GPU OUT_DIR` 调用 `eval_v4.py`：552 题 × 4 次采样，temperature 0.7，seed_base 20261017，累计生成上限 8,192，上下文 32,768。
- `summarize.py:extract`（L21）取最后一个 `<answer>字母</answer>`，并带 fallback；截断、错误、无答案都计为错，保留在分母里。
- `grpo_v51fix_codex/` 是评测时加载的协议适配目录，对越界 bbox 拒绝执行而不是截断。

## 协议（[protocol](protocol/)）

- `caption_v51.py`：SYSTEM 原文、`task_hints`（L108，13 类 hint 的生成规则）、`parse_step` / `check_step`（L184 / L190，硬检查）、`soft_flags`、`first_user_text`、`receipt_text`。

## 标注（[annotation_rl4k](annotation_rl4k/)）

| 文件 | 作用 |
|---|---|
| `annotate_api.py` | L0–L2 两轮 API 标注：先出草稿，再叠框并对照原图裁图核对后写 L2；校验报错回灌，最多 2 次。模型地址和密钥只从环境变量读取（`ANNOT_API_URL`、`ANTHROPIC_AUTH_TOKEN`） |
| `annotate_clues.py` | LC 线索层标注与校验（`--validate`） |
| `validate_annotations.py` | 硬错误 / 警告校验（规范见 [design/ANNOTATION_SPEC.md](../design/ANNOTATION_SPEC.md)） |
| `build_types.py` | 用协议 `task_hints` 生成单位草案（`hint_keys`） |
| `merge_fixes.py`、`prep_review.py`、`make_review_sheets.py` | 审查合并与审查图 |
| `*_BRIEF.md`、`AGENT_PROMPT.md` | 标注、审查、重标时给模型的说明原文 |
| `build_canonical.py` | relation 方向规范化（computable_from_boxes 的 48 条） |

## 旧目录

[v53/](v53/) 是早先发布的 v53 阅读快照，内容已包含在 [rloo_v53/](rloo_v53/) 中，保留它是为了不改动 `provenance/CODE_HASHES.json` 的引用。
