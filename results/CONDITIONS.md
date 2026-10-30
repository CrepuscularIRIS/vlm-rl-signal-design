# 历史条件目录

各条件都为552题×4；此目录是快照，不代表所有条件同预算/同协议。上游说明为原始定义。

| 文件/条件 | 正确/2208 | E1 |
|---|---:|---:|
| [B0-ans.jsonl.gz](../data/rollouts/B0-ans.jsonl.gz) | 904 | 0.4094 |
| [B0.jsonl.gz](../data/rollouts/B0.jsonl.gz) | 1152 | 0.5217 |
| [B1.jsonl.gz](../data/rollouts/B1.jsonl.gz) | 1300 | 0.5888 |
| [B2.jsonl.gz](../data/rollouts/B2.jsonl.gz) | 15 | 0.0068 |
| [B3-ans.jsonl.gz](../data/rollouts/B3-ans.jsonl.gz) | 853 | 0.3863 |
| [B3-cot.jsonl.gz](../data/rollouts/B3-cot.jsonl.gz) | 1188 | 0.5380 |
| [B3-zoom.jsonl.gz](../data/rollouts/B3-zoom.jsonl.gz) | 710 | 0.3216 |
| [B3.jsonl.gz](../data/rollouts/B3.jsonl.gz) | 1252 | 0.5670 |
| [B4.jsonl.gz](../data/rollouts/B4.jsonl.gz) | 1458 | 0.6603 |
| [B5.jsonl.gz](../data/rollouts/B5.jsonl.gz) | 17 | 0.0077 |
| [B6.jsonl.gz](../data/rollouts/B6.jsonl.gz) | 1316 | 0.5960 |
| [H1.jsonl.gz](../data/rollouts/H1.jsonl.gz) | 997 | 0.4515 |
| [H2-6k.jsonl.gz](../data/rollouts/H2-6k.jsonl.gz) | 1113 | 0.5041 |
| [H2.jsonl.gz](../data/rollouts/H2.jsonl.gz) | 1058 | 0.4792 |
| [K-DAPO-40.jsonl.gz](../data/rollouts/K-DAPO-40.jsonl.gz) | 1438 | 0.6513 |
| [K-DAPO.jsonl.gz](../data/rollouts/K-DAPO.jsonl.gz) | 1465 | 0.6635 |
| [K-GRPO.jsonl.gz](../data/rollouts/K-GRPO.jsonl.gz) | 1411 | 0.6390 |
| [K-GSPO-40.jsonl.gz](../data/rollouts/K-GSPO-40.jsonl.gz) | 1442 | 0.6531 |
| [K-GSPO.jsonl.gz](../data/rollouts/K-GSPO.jsonl.gz) | 1412 | 0.6395 |
| [K-GiGPO.jsonl.gz](../data/rollouts/K-GiGPO.jsonl.gz) | 1371 | 0.6209 |
| [K-base.jsonl.gz](../data/rollouts/K-base.jsonl.gz) | 12 | 0.0054 |
| [V53-control-32.jsonl.gz](../data/rollouts/V53-control-32.jsonl.gz) | 1395 | 0.6318 |
| [V53-decision-32.jsonl.gz](../data/rollouts/V53-decision-32.jsonl.gz) | 1307 | 0.5919 |
| [X1.jsonl.gz](../data/rollouts/X1.jsonl.gz) | 1412 | 0.6395 |
| [X2.jsonl.gz](../data/rollouts/X2.jsonl.gz) | 1234 | 0.5589 |

主要标签：B0=原模型CoT，B0-ans=原模型direct，B1=原生zoom，B2=caption zero-shot；B3=SFT caption，B3-cot/ans/zoom为各对应接口；K-*为4k×8/32步（*-40除外）；V53两组32步；X1=Mini-o3，X2=DeepEyes；B4/B5/B6等是旧预算历史条件。

精确定义见[固定上游README](https://github.com/CrepuscularIRIS/vlm-caption-trajectories/blob/2c5155d7e629e6eadb043bb8c751035f9132fca9/trajectories/e1_rollouts/README.md)。不得把H2等旧条件当成最新同预算baseline。
