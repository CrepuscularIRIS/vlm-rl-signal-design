# methods：各方法与版本

叠加规则：一个方法版本 = `../data/adapter/` 全部文件 + 本版本目录里的同名覆盖与新增文件（`sg/v2` 叠加在 `sg/v1` 之上）。所有方法共用 `../scripts/` 的公共参数，E1 用 `../eval/e1/` 评测。

| 方法 | 版本目录 | 改动落在哪一层 | 核心公式所在 | 启动方式（相对 GRPO 的覆盖项） | E1 |
|---|---|---|---|---|---:|
| GRPO | [grpo/v1](grpo/v1/) | — | `data/easyr1_runtime/core_algos.py` `compute_grpo_outcome_advantage`（L176–216） | `scripts/slot2.sh`：`run k8_grpo grpo_v52m_c.sh $SFT 2,3 $U2` | .6390 |
| DAPO | [dapo/v1](dapo/v1/) | 采样 / 组过滤 | `data/easyr1_runtime/ray_trainer.py` L518–556 | `$DAPO` + `$U1` | .6635 |
| GSPO | [gspo/v1](gspo/v1/) | 序列级 ratio | `core_algos.py` gspo 分支 L465–480 | `$GSPO` | .6395 |
| GiGPO | [gigpo/v1](gigpo/v1/) | 信用（锚点 step advantage） | `gigpo.py:step_advantages`（L23）、`install`（L47） | `CAPTION_GIGPO=1`，`grpo_gigpo.sh` | .6209 |
| RLOO / + 首动作分叉 | [rloo/v1](rloo/v1/) | 组基线；首动作辅助损失 | `decision_loss.py`（`decision_gap` L53、`pair_loss` L57、`install_actor_patch` L155）；奖励 Y·(1−λN) 在 `reward_caption.py` | `launch_formal.py`：`LAMBDA=0.01`、`BETA=0 / 0.00390625`、`K=4` | .6318 / .5919 |
| DTPO | [dtpo/v1](dtpo/v1/) | 信用（分段 loss 分母） | `dtpo_adapter.py`：`partitions`（L16–21）、`update_policy`（L62）、`token_loss`（L26） | `run.sh` / `launch.py` | .6553 |
| SAPO | [sapo/v1](sapo/v1/) | 信用（turn 级 GAE） | `sapo_adapter.py`：`capture_forward`（L49）、`compute_advantage`（L104）、`update_policy`（L132） | `run.sh`：`rollout.n=1` | .5770 |
| Hτ | [htau/v1](htau/v1/) | reward（bbox 覆盖过程奖励，导师逼近构造的第一次实现） | `reward_d2.py:process_score`（L64）；`caption_contracts.py:assistant_reward_batch`（L147） | `grpo_d2.sh` + DAPO 过滤（`filter_key=accuracy`） | .6508 |
| G1 | [g1_rread/v1](g1_rread/v1/) | reward（读值一致性） | `reward_rread.py:process_score`（L74）；`rread_scorer.py` | `grpo_r1.sh` | .6395 |
| A / A+B / A+B+C | [sg/v1](sg/v1/) | reward（尺度门控 S）；B：advantage；C：数据 | 见 [sg/v1/README.md](sg/v1/README.md) | `grpo_sg.sh` | GRPO+A .6436 / A+B .6472 / A+B+C .6395 |
| 导师 psi（逼近 / 探索） | [sg/v2](sg/v2/) | reward | `reward_sg.py:_psi`、`process_score` | `grpo_sgv2.sh`（`CAPTION_SG_PSI=max`） | 训练中 |
| DAPO-SB | [dapo_sb/v1](dapo_sb/v1/) | 组内基线 / 信用（停止探针） | `stop_baseline.py`、`stop_relative_advantage.py:build_advantages`（L31） | `grpo_sb.sh` + `$DAPO`，`CAPTION_SB=1` | .6413 |
