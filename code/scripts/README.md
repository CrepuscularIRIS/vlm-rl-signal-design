# scripts：实验启动与监控

| 文件 | 作用 |
|---|---|
| `grpo_v52m_c.sh` | 公共 launcher：k8 GRPO 配方（4k 池、n=8、128 题/步、actor mini-batch 32、lr 1e-6、max_grad_norm 1.0、无 KL、clip .2/.3） |
| `common.sh` | 公共参数与覆盖项变量：`$DAPO`（online_filtering、filter_key=overall、0.01/0.99、max_try 100、clip high 0.28）、`$GSPO`、`$U1`/`$U2`（update micro batch） |
| `slot1.sh`、`slot2.sh` | 两个两卡槽位的排队启动（GRPO / DAPO / GSPO / GiGPO / K-base） |
| `eval_k8.sh` | k8 组训练结束后的合并与 E1 评测 |
| `mem_watch.sh` | 共享内存守卫 |

方法专用的 launcher 放在各自的 `methods/<方法>/<版本>/` 里。
