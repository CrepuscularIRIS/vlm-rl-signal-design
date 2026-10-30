# 外部协议：已有结果与尚未完成的工作

Mini-o3-7B=.6395、DeepEyes-7B=.5589为**原生工具接口＋E1采样/预算适配**。这不是完整官方预算复现，不是外部互换协议结果，也没有新增本地六方法运行外部协议的结果。

| 项目 | 本地六方法 | Mini-o3 | DeepEyes |
|---|---|---|---|
| 底座 | Qwen3-VL-4B系列 | Qwen2.5-VL-7B系列 | Qwen2.5-VL-7B系列 |
| 工具 | caption v5.1 | 原生grounding相对坐标、支持已有observation | 原生tool_call、原图绝对坐标 |
| 最大像素 | 589824 | 2000000 | 12845056 |
| 工具/图像预算 | 8裁图 | 12轮/12图 | 11轮/12图 |
| caption WORD/action标记 | 适用 | N/A | N/A |

已有E1共同使用每题4条、temperature .7、top_p1、累计生成8192、context32768；同任务同评分器不等于同协议。

已核对官方固定代码的区别：
- Mini-o3提示常量逐字一致；README示例Avg@32也支持greedy Avg@1，不能把Avg@4称官方论文复现。
- DeepEyes官方示例temperature0、每轮10240；已有E1为.7及整条累计8192。
- 同轮tool+answer：官方先进入工具分支，本地先answer终止；无tool无answer、length分支也不同。
- DeepEyes官方没有设置`</tool_call>`停止词。不能为改善结果自行添加后仍称官方协议。

DeepEyes旧E1有232条截断/overlong；216条截断末轮含tool_call字符串，916条至少一轮tool与answer共现。这些是字段/词面计数，不是证明哪一差异导致掉分。没有“修复后”分数。

固定来源及SHA在 [sources](evidence/sources.json)，提示核对在 [prompt checks](evidence/prompt_checks.json)，词面计数在 [edge counts](evidence/edge_counts.json)。未来如需官方原生复现或协议交叉，必须独立标记运行和预算；当前没有派发这项工作。
