# DAPO-SB 训练冻结记录（2026-11-04）

用户指示：用 GPU 2,3 训练 Advantage.md（Pro，11-04 改写版）中的 DAPO-SB。

## 唯一变量（对 k8 DAPO）
- advantage：$A_{i,t}=(Y_i-b_{i,k(t)})/(\sigma_q+\epsilon)$，λ=1；回执、图像 token 为 0。替换掉 DAPO 的组均值基线；$\sigma_q$ 仍为组内 Y 的样本标准差。advantage 核心直接使用 Pro 的参考实现 `stop_relative_advantage.py`（sha16 a4f3374f，自检通过）。
- 探针 $b_{i,k}$：在第 k 个 assistant 回合之前，用 rollout 引擎当前权重（θ_old）计算，输入为 caption v5.1 SYSTEM + 题面与选项 + 该回合前实际呈现过的图像（原图 + 前 k 张裁图），不含模型自己的文本。读出方式：
  - 请求 1 以 `<answer` 结尾，读 `>A`…`>J`（训练文本中 `<answer>D` 被切成 `'<' 'answer' '>D'`）和单独的 `>`；
  - 请求 2 以 `<answer>` 结尾，读 `A`…`J`；
  - 合并：$P(X)=P(\texttt{>X})+P(\texttt{>})\cdot P(X\mid\texttt{<answer>})$；top-20 原始 logprob，一次前向，不采样。
- 其余与 k8 DAPO 相同：A-SFT 起点、4k 池、seed 42、n=8、128 题/步、mini 32、32 步、lr 1e-6、无 KL；online_filtering=true，filter_key=overall（λ_SG=0 时 overall = V·Y），0.01 / 0.99，max_try 100；clip .2/.28；token 平均；update micro 1、experience micro 4。
- 记录事项：k8 DAPO 运行时设置了 CAPTION_MM_CACHE_GB=16 / PREFETCH=16 / WINDOW=128；本次为控制内存（与 sg_a 并行）设 CACHE=0、PREFETCH=16、WINDOW=128。这只影响速度和内存，不改变计算。

## 启动前检查
- CPU：token 区间、回执掩码、按回合广播、gt 解析；参考实现自检通过。
- tokenizer：`>A`…`>J`、`A`…`J` 都是单个 token；`<answer` 前缀的切分与训练文本一致。
- 探针冒烟（A-SFT，40 道训练题，只看原图）：
  - 答对选项的平均概率 .371，取最大概率选项的正确率 .525，字母概率总量均值 .778（最低 .293）；
  - 缺失的概率质量流向单独 `>` 之后的 `UNC`（UNCLEAR）或数字，不是漏读的字母；
  - 取最大概率时的选项分布：E 10、B 9、C 7、A 6、D 6、F 1、H 1。
- 代码：服务器 `myproject/grpo_sb`（由 grpo_sg 复制；新增 stop_baseline.py，并改动 easyr1_caption.py / caption_contracts.py 的探针与账本接线）。

## 首步观测（18:56 CST，`outputs/sb_dapo_sb/`）
- 启动：GPU 2,3；同机 sg_a 在 GPU 0,1 运行；cgroup 内存 232G。
- 第一批（128 个 mixed 组 × 8 = 1024 条轨迹，DAPO 补采到第 7 次奖励调用）：
  - `[sb] call=1 b0=0.257 acc=0.482 gap=+0.225 mass=0.730 |A| sb=1.003 dapo=0.809`；
  - turns_per_traj 1.770；mean_raw_adv +0.183；raw_adv>0 占比 .462。
  - updates.jsonl：all_correct 0 / all_wrong 0 / mixed 128；adv_nonzero_token_share 1.0。
- rollout 账本（1792 行，覆盖全部 7 次奖励调用）：每行都有 b；b 的长度与裁图数+1 一一对应，(长度, 裁图数) 分布为 (1,0) 1116、(2,1) 391、(3,2) 171、(4,3) 62、(5,4) 17、(8,7) 11。
  - mean b0 .332，mean letter mass(turn0) .761；
  - 答对轨迹 b0 均值 .436，答错 .195。
- 每次奖励调用的探针请求数：120–274（两卡，每卡 128 条轨迹）。
- Runner Step 1 完成（19:10 前后，单步约 27 分钟；与 sg_a 并行）。与 k8 DAPO 的第 1 步对照（同一起点、seed、题池；第 1 步 lr 都是 0.0，处于预热）：

  | 指标 | DAPO-SB | k8 DAPO |
  |---|---:|---:|
  | critic/score/mean | .482 | .482 |
  | actor/entropy_loss | .448 | .448 |
  | actor/grad_norm（裁剪前） | 11.059 | 8.225 |
  | actor/pg_loss | −.454 | .003 |
  | critic/advantages/mean | +.113 | −.007 |

  advantages 的范围为 [−2.719, 2.823]，pg_clipfrac 为 0，ppo_kl 为 0。
- `[sb] call=2 b0=0.257 acc=0.498 gap=+0.241 mass=0.711 |A| sb=0.986 dapo=0.788`。
- 守护：每 10 分钟记一行到服务器 `/tmp/sb_guard.log`；遇到 Traceback/OOM、内存 ≥405G、step32 actor 存盘或进程退出时结束。

## 终点验收（2026-11-05 08:08 CST）
- 32/32：tracker last_global_step 32；step32 rank0/1 的 model / optim / extra_state 文件非空；updates.jsonl 32 行，sb_log.jsonl 32 行；日志 0 Traceback/OOM；进程已退出。
- dev val（4k val_nosft）：训练前 .533 → 第 25 步 .590 → 第 32 步 .6066。sg_a 的终点 val 也是 .6066，但第 25 步不同（.583），记作计数相同。
- model_merger 合并完成（huggingface/ 9.1G）。E1 在 GPU0 上运行，tag sb_dapo_sb，输出 → evidence/e1_sb_dapo_sb_20261105/。
- 训练耗时：Runner 进度条记录 32 步约 13.3 小时（与 sg_a 并行 12.2 小时）。
