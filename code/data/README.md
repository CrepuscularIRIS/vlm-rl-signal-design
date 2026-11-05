# data：loader 与 runner

## adapter/：EasyR1 适配层（所有方法共用）

所有方法都通过 `easyr1_caption.py:install()` 把 caption 多轮协议接进 EasyR1。接入点有三处：

| 接入点 | 函数 | 作用 |
|---|---|---|
| 数据 loader | `caption_getitem` | 用与 rollout 完全相同的原图呈现规则构造首轮 prompt |
| rollout runner | `generate_sequences_multiturn` | 多轮循环：每轮 vLLM 生成 → `proto.parse_step` / `check_step` 硬检查 → `CropEnv.step` 裁图 → 回执和新图追加到 prompt；最多 8 次裁图 + 1 次作答 |
| 奖励装配 | `caption_contracts.assistant_reward_batch` | 计算 V·Y（以及各方法的过程项），奖励放在最后一个 assistant token |

其他文件：
- `caption_env.py`：裁图与呈现尺寸规则（56 px 最小裁图、短边 448、放大上限 4×、面积上限 1400²、对齐 28）。
- `reward_caption.py`：Y 的规则（恰好一个 `<answer>X</answer>` 且 X 等于 gt）；`reward_direct.py`：直答对照用。
- `update_ledger.py`：每步记录组构成（全对 / 全错 / 混合）和非零 advantage 占比。
- `easyr1_patches.py`：GSPO 在 seq 模式下按序列数给 micro-batch 加权；空 chunk 过滤。
- 训练时每条轨迹写一行 rollout 账本（`CAPTION_ROLLOUT_LOG`）。

## easyr1_runtime/：训练时使用的 EasyR1 源码（Apache-2.0）

| 文件 | 内容 |
|---|---|
| `core_algos.py` | `compute_grpo_outcome_advantage`（L176–216）、RLOO（L268 起）、`average_loss`（L383）、`compute_policy_loss`（L410 起；GSPO 分支 L465–480） |
| `ray_trainer.py` | DAPO 过滤与补采（L518–556：组均值 ∈ (filter_low, filter_high) 才保留，不够就补采） |
| `config.py` | 配置项定义 |
