# Qwen3-VL 在 V*、HR-Bench、MME-RealWorld 上的评测设置核实

核实日期：2026-11-06。范围：公开论文原文、官方 GitHub、官方 Hugging Face model card/config。仅资料核实，未运行评测、训练或 GPU 程序。本文所有列入的数值和设置均通过 WebFetch 读取对应原文；WebSearch 仅用于定位。GitHub/HF 的 main 是核实时页面，不保证与论文实验提交版本相同。

## 1. 读表口径与可直接引用的数值范围

- 分数均为 accuracy %，V* 和 HR-Bench 取 overall，而非 Attr/Spatial 或 FSP/FCP 单项。
- **base** 在本文指未经该论文方法训练的 Qwen checkpoint，不指预训练 Base checkpoint；多数是 Instruct。**base + tool 与无工具直接作答分开列**。
- `—` 表示该项未在已读原文中报告；`未核实` 表示所需信息未从所读来源取得，不能补成默认设置。论文未写某值，不代表运行时不存在该值。
- MME-RealWorld、MME-RealWorld-Lite、RealWorldQA、MME-P、MMStar 不互换。MME-RW 未标 Lite 的行仍保留论文自己的名称，不据此断言其具体数据版本/样本数。
- 已读第三方原文中，**Qwen3-VL-4B 无工具 base**：V* **78.5–80.10**；HR-4K **78.25–79.6**；HR-8K **72.88–74.6**。这是不同论文已报告值的范围，不是本地复现实验或置信区间；BVS 未明确 checkpoint 后缀，其他来源按下表区分。
- ZwZ 对官方工具流程的复现另报 **Qwen3-VL-4B+tool：88.0 / 81.3 / 74.4**（V* / HR-4K / HR-8K），不能合并进无工具 base 范围。
- **Qwen 官方 4B/2B/8B 的这三个 benchmark 的精确数值本次未核实**：官方 README/HF 图表没有可抽取文本；技术报告 §5.6 的文字只明确给出旗舰带工具分数，不能移植到 4B。

## 2. 汇总表

来源编号对应第 6 节，可点击回原文。设置栏只记录已读取的信息；训练时采样参数不写成评测参数。

| 论文 / 资料 | 模型与条件 | V* | HR-4K | HR-8K | MME-RW | 解码 | 分辨率设置 | 坐标格式 | 来源链接与核实状态 |
|---|---|---:|---:|---:|---|---|---|---|---|
| Qwen3-VL 官方 | 2B/4B/8B Instruct/Thinking | 未核实 | 未核实 | 未核实 | 未核实 | 官方推荐为采样，见 §3 | 4B processor config：65,536–16,777,216 pixels；不等于这几项 benchmark 的设置 | 原生 grounding XYXY，[0,1000] | [S1–S6](https://arxiv.org/html/2511.21631)；已核实推荐/config，未核实小模型目标分数 |
| Qwen3-VL 技术报告 | 235B-A22B，带工具；仅作官方文字出处 | 93.7 | 85.3 | 82.3 | — | 目标三项的专门解码未核实 | 未核实 | [0,1000] | [S1](https://arxiv.org/html/2511.21631) §5.6；已核实，非 2B/4B/8B |
| OPO | Qwen3-VL-2B，作者复现 base **+Zoom** | 75.9 | 66.0 | 60.6 | — | greedy | 仅写按 Qwen-VL visual grid resize，无数值 cap | XYXY，[0,1000] | [S9](https://arxiv.org/html/2611.04943) Table 1/2、App. B/C/E；已核实 |
| OPO | Qwen3-VL-8B，作者复现 base **+Zoom** | 88.0 | 81.4 | 76.5 | — | greedy | 同上，max_pixels 未核实 | 同上 | [S9](https://arxiv.org/html/2611.04943)；已核实 |
| OPO | OPO-2B，方法训练后 | 85.9 | 78.4 | 76.4 | — | greedy | 未核实数值 cap | XYXY，[0,1000] | [S9](https://arxiv.org/html/2611.04943) Table 1；已核实 |
| OPO | OPO-8B，方法训练后 | 92.7 | 85.5 | 80.8 | — | greedy | 未核实数值 cap | XYXY，[0,1000] | [S9](https://arxiv.org/html/2611.04943) Table 1；已核实 |
| Zooming without Zooming / ZwZ | Qwen3-VL-2B-Instruct，无工具 base | 72.77 | 71.75 | 70.12 | en 59.52；cn 60.77 | 2B model card 未写；仓库公共评测代码为采样，见 §4 | 2B 对应实验具体配置未核实；公共评测脚本见 §4 | 单次直接作答，无工具框 | [S12](https://huggingface.co/inclusionAI/ZwZ-2B)；已核实 model card，非原始论文 Table 2 行 |
| Zooming without Zooming / ZwZ | Qwen3-VL-4B-Instruct，无工具 base | 80.10 | 78.25 | 72.88 | en 63.47；cn 63.63 | 公共代码 T=.7、p=.8、k=20 | 公共代码 65,536–16,777,216 pixels | 单次直接作答，无工具框 | [S11](https://arxiv.org/html/2602.11858) Table 2；[S14](https://raw.githubusercontent.com/inclusionAI/Zooming-without-Zooming/main/mm-eval/infer_without_tool.py)；数值/当前代码均已核实，论文提交版本对应未核实 |
| Zooming without Zooming / ZwZ | Qwen3-VL-8B-Instruct，无工具 base | 86.39 | 78.88 | 74.63 | en 65.96；cn 66.67 | 同上 | 同上 | 无工具框 | [S11](https://arxiv.org/html/2602.11858)、[S14](https://raw.githubusercontent.com/inclusionAI/Zooming-without-Zooming/main/mm-eval/infer_without_tool.py)；已核实，同上版本边界 |
| Zooming without Zooming / ZwZ | Qwen3-VL-4B+tool，官方 cookbook 流程复现 | 88.0 | 81.3 | 74.4 | en 65.4 | 论文未列该行具体解码值 | 未核实；不把无工具脚本 cap 填给工具行 | cookbook 原生 XYXY，[0,1000]；是否有覆盖未核实 | [S11](https://arxiv.org/html/2602.11858) Table 4；已核实分数 |
| Zooming without Zooming / ZwZ | Qwen3-VL-8B+tool，官方 cookbook 流程复现 | 90.1 | 82.3 | 78.0 | en 66.0 | 同上 | 同上 | 同上 | [S11](https://arxiv.org/html/2602.11858) Table 4；已核实分数 |
| Zooming without Zooming / ZwZ | ZwZ-4B，DAPO 训练后，无工具 | 92.67 | 81.75 | 79.50 | en 68.52；cn 68.09 | 公共无工具脚本如上 | 公共无工具脚本如上 | 测试时不提供框 | [S11](https://arxiv.org/html/2602.11858)、[S12a](https://huggingface.co/inclusionAI/ZwZ-4B)；已核实 |
| GazeVLM | Qwen3-VL-4B-Instruct base | — | 79.5 | 73.6 | — | 主表模型解码未核实；不能把 judge greedy 当模型 greedy | 未报告数值 cap | 方法自身 LOOK 用 XYXY、[0,1]；base 无 LOOK | [S17](https://arxiv.org/html/2605.07817) Table 1、App. A.3/A.4；已核实 |
| GazeVLM | GazeVLM-4B，GRPO 后 | — | 83.4 | 78.0 | — | 主表未核实；Table 4 消融明确 greedy | 图像只编码一次；数值 cap 未报告 | `<LOOK ... bbox=[x1,y1,x2,y2]>`，示例为 [0,1] | [S17](https://arxiv.org/html/2605.07817)；已核实；HR-8K 是 78.0，不是二手摘要的 78.3 |
| BVS | Qwen3-VL-4B，w/o reasoning；未写 Instruct 后缀 | 78.5 | 79.6 | 74.6 | — | 未报告 | 未报告数值 cap | 未报告数值坐标规范 | [S18](https://arxiv.org/html/2607.03184) Table 1；已核实，training-free |
| BVS | Qwen3-VL-8B，w/o reasoning；未写 Instruct 后缀 | 82.7 | 77.9 | 72.8 | — | 未报告 | 未报告数值 cap | 同上 | [S18](https://arxiv.org/html/2607.03184) Table 1；已核实 |
| BVS | Qwen3-VL-4B+BVS，training-free search | 91.1 | 82.4 | 79.5 | — | 未报告 | 未报告数值 cap | 未报告数值坐标规范 | [S18](https://arxiv.org/html/2607.03184) Table 1；已核实 |
| Act Wisely / Metis | Qwen3-VL-8B-Instruct base | 86.4 | 78.9 | 74.6 | 61.9，论文列 MME-RW | 评测未报告 | 评测未报告数值 cap | 方法工具为相对 [0,1]，非原生 [0,1000] | [S19](https://arxiv.org/html/2604.08545v1) Table 1、App. A；已核实 |
| TACO | Qwen3-VL-8B-Instruct base | 86.4 | 78.9 | 74.6 | 60.3，论文列 MME-RealWorld | 仅写 VLMEvalKit default protocol；未给参数 | 未报告数值 cap | 未报告 | [S22](https://arxiv.org/html/2606.30251v1)；已核实；不与 Metis 的 61.9 或 ZwZ 的 65.96 合并 |
| TACO | TACO-Qwen3-VL-8B，训练后 | 92.9 | 85.3 | 84.1 | — | 同上 | 同上 | 未报告 | [S22](https://arxiv.org/html/2606.30251v1)；已核实；文中 66.2 属于 Qwen2.5-7B 方法行，不填此格 |
| Thinking Without Images / Imagine-OPD | Qwen3-VL-4B base | 80.1 | 78.3 | 72.9 | **Lite 50.1** | 评测未报告 | 未报告数值 cap | base 无工具；self-bbox 消融用原图整数像素 XYXY | [S23](https://arxiv.org/html/2606.08719)；已核实，蒸馏而非工具 RL |
| Thinking Without Images / Imagine-OPD | Qwen3-VL-8B base | 86.4 | 78.9 | 74.6 | **Lite 50.3** | 同上 | 同上 | 同上 | [S23](https://arxiv.org/html/2606.08719)；已核实 |
| Thinking Without Images / Imagine-OPD | Imagine-OPD-4B，训练后 | 88.0 | 83.3 | 78.6 | **Lite 56.9** | 评测未报告 | 未报告数值 cap | 测试时无外部 crop 工具 | [S23](https://arxiv.org/html/2606.08719)；已核实 |
| InSight-doc | Qwen3-VL-8B，无 zoom；r=.25 / .5 | — | — | — | **Lite 41.0 / 49.2** | 未报告 | 初始图像边长比例 r=.25 / .5；无数值 max_pixels | base 无 zoom | [S24](https://arxiv.org/html/2608.10628)；已核实，非 full MME-RW |
| InSight-doc | Qwen3-VL-8B+zoom；r=.25 / .5 | — | — | — | **Lite 42.5 / 50.6** | 未报告 | 同上，工具从高分辨率源图裁剪 | box 规范未报告 | [S24](https://arxiv.org/html/2608.10628)；已核实 |

### 一个没有强行换算 overall 的 Thinking 来源

Reliable Thinking with Images（RTWI）[S25] 是推理时多轨迹聚合，不是 RL。Table 1 的 `Qwen3-VL Thinking / Base` 行：V* Attr **78.3**、Spatial **73.7**；HR4K FSP **83.0**、FCP **57.3**；HR8K FSP **78.3**、FCP **56.0**。该行自身未打印模型规模，附录把 Qwen3-VL-8B-Thinking 列为 default。**本表不把它无条件改写成 8B overall，也不根据四舍五入的子分数构造 overall。** 2B/4B/8B/32B scale 图的具体数值本次未核实。

## 3. Qwen 官方推荐、实际 checkpoint 默认与示例不是同一件事

### 3.1 生成参数

| 来源 | 对象 | greedy / do_sample | temperature | top_p | top_k | 其他 |
|---|---|---|---:|---:|---:|---|
| 官方 GitHub README [S2] | Instruct，官方复现推荐 | greedy=false | .7 | .8 | 20 | seed=3407；repetition_penalty=1；presence_penalty=1.5；out_seq_length=32768 |
| 官方 GitHub README [S2] | Thinking，官方复现推荐 | greedy=false | .6 | .95 | 20 | seed=1234；repetition_penalty=1；presence_penalty=0；out_seq_length=40960 |
| 4B-Instruct HF card [S3] | VL 推荐 | greedy=false | .7 | .8 | 20 | repetition_penalty=1；presence_penalty=1.5；out_seq_length=16384 |
| 4B-Instruct HF card [S3] | Text 推荐 | greedy=false | 1 | 1 | 40 | repetition_penalty=1；presence_penalty=2；out_seq_length=32768 |
| 4B-Instruct generation_config.json [S4] | checkpoint 配置 | do_sample=true | .7 | .8 | 20 | 不据此补出该文件未列的 presence penalty 或序列长度 |

README 与 4B card 的输出长度不同，以上保留原值；**不将它们拼成一个所谓统一默认**。技术报告 §5.11 给出的 text-centric eval 设置也不冒充 V*/HR-Bench 设置。

### 3.2 图像像素 / visual token

- **4B-Instruct checkpoint 文件 [S5]**：`size.shortest_edge=65536`、`size.longest_edge=16777216`、`patch_size=16`、`merge_size=2`。这里 `size` 表示 H×W 的范围，不是把最长边设置成 16,777,216 像素。按 32×32 pixels / merged visual token 的换算，对应 **64–16,384 个 visual tokens**；实际 resize 会做网格对齐。
- **README 的 processor 配置示例 [S2]**：`shortest_edge=256*32*32`、`longest_edge=1280*32*32`，即 **262,144–1,310,720 pixels / 256–1,280 visual tokens**。它是手动设置示例，**不是 4B checkpoint 默认，更不是已核实的 benchmark 分辨率**。
- **qwen-vl-utils 当前源代码 [S6]**：`IMAGE_MIN_TOKEN_NUM=4`、`IMAGE_MAX_TOKEN_NUM=16384`、`SPATIAL_MERGE_SIZE=2`；`fetch_image` 和 `process_vision_info` 的 `image_patch_size` 参数默认仍为 **14**。单独采用 utility 的像素缺省值与采用 checkpoint processor 不同。以 Qwen3 显式 patch=16 代入，utility 的 min/max 是 **4,096–16,777,216 pixels**，不能把 utility 最小值 4 tokens 当成 4B processor 的 64 tokens。
- README 明确要求 Qwen3-VL 调用 `process_vision_info(..., image_patch_size=16, ...)`；utility 已 resize 后，processor 用 `do_resize=False`，避免再次 resize [S2]。
- 官方 2D grounding notebook 的 API helper 又给出 **min_pixels=64*32*32**、**max_pixels=9800*32*32**，即 **65,536 / 10,035,200** [S8]。仍是该 helper 的默认，不是上述 benchmark 的通用值。
- HR-Bench 的 4K/8K 是数据集名称；如果没有 preprocessing 参数，**不能从名称推断模型实际吃进 4K/8K 全图**。

### 3.3 Grounding / crop 坐标

技术报告 §3.2.4 和官方两个 notebook [S1,S7,S8] 明确：原生 box/point grounding 归一化至 **[0,1000]**。box 为 `[x1,y1,x2,y2]`，分别左上、右下；工具参数名 `bbox_2d`，还包括 `img_idx` 和 `label`。反算公式为 `x_pixel=x/1000*width`，`y_pixel=y/1000*height`。

官方 think_with_images notebook 使用 `image_zoom_in_tool`，示例 `bbox_2d=[574,301,660,468]`、`img_idx=0`，并列出 T=.7、p=.8、k=20。示例 active model 是 `qwen3-vl-235b-a22b-instruct`，不是 4B。所读内容未给数值 tool-turn cap 或图像 cap。**Metis/GazeVLM 的 [0,1] 和 Imagine-OPD self-bbox 的像素坐标是自定义接口，不据此修改 Qwen 原生坐标结论。**

## 4. 论文执行设置逐项核实，含 RL max_pixels

| 工作 | 评测解码 / 输出预算 | 评测分辨率 | 工具 / 交互上限 | RL 训练 max_pixels 与其他已核实设置 |
|---|---|---|---|---|
| OPO [S9,S10] | 明确 greedy；每次生成一个 assistant turn | HR-Bench 按 Qwen-VL visual grid resize；无数值 max_pixels | 评测最多 **10 tool-use turns/sample**；训练最多 **5 assistant turns/sample**。两种计数术语保留原文 | **未报告数值 max_pixels**；仓库 README 仍写 Code coming soon，不能从尚未公开的 config 补值 |
| ZwZ [S11–S16] | 公共 `infer_without_tool.py`：T=.7、p=.8、k=20、presence=1.5、repetition=1、seed=42、`max_tokens=8192` | 每张 prompt image `min_pixels=65536`、`max_pixels=16777216`，processor 也传 max_pixels=16777216 | 主方法/无工具 base 单次直接作答；工具行采用官方流程，但论文只说遵守各官方 max steps，无数值 | `EasyR1/examples/perception_config.yaml`：**min_pixels=262144、max_pixels=16777216**；T=1、p=1、n=8；response=2048。`single_node.sh` 确实引用该 YAML，同时把 max_prompt_length 从 YAML 的 32768 覆盖成 **25000**。这些是核实时公开代码，不保证论文实验 commit 相同 |
| GazeVLM [S17] | 主表未明确模型 greedy；明确 greedy 的位置是 judge 与 Table 4 LOOK-removal 消融 | 未报告数值 cap；图像编码一次 | **未报告硬上限**。数据 curation 对 >5 looks 加惩罚；GRPO 对 N_L≥11 加 excess penalty，**都不是推理 hard cap**；两项 HRBench 实测均值 1.19 LOOK/trace | **未报告数值 max_pixels** |
| BVS [S18] | 未报告 T/p 或 greedy | 未报告数值 cap | 默认最多 **5 search steps**，阈值 τ=.2；原文称 search steps，不直接改称 5 个工具调用 | training-free，**无 BVS RL max_pixels** |
| Metis [S19–S21] | benchmark 解码未报告；16384 是训练 response token cap | 论文和所读 train script 未列 max_pixels；script 指定 patch_size=16 | 论文未给数值上限；公开训练 script `actor_rollout_ref.agent.max_turns=10`，**不是已核实评测上限** | **所读训练 script 未显式指定 min/max_pixels**，不能据此断言底层 config 没有限制；script rollout T=1、p=1、k=-1、n=16；max_prompt=24576、response=16384 |
| TACO [S22] | 仅写 VLMEvalKit default protocol；不据此填 greedy | 未报告数值 cap | 未报告 | **未报告数值 max_pixels**；RL rollout G=8、T=1，不是评测解码 |
| Imagine-OPD [S23] | 评测参数未报告；训练 rollout T=1、一次 rollout/prompt、response=256 | 未报告训练/评测数值 max_pixels | 测试为无工具蒸馏模型 | 不是 RL；训练 max_pixels **未报告**；self-bbox 消融为原图整数像素 XYXY |
| InSight-doc [S24] | 未报告评测 T/p | 初始边长比例 r=.25/.5；MME-RW-Lite 不涉及 PDF rasterize | 只说回答或到 tool limit 停止，未给数值 | **未报告数值 max_pixels**；训练有 24k 估计总 token cap；超 cap 的 r=.5 行改成 r=.35，仍超则丢弃 |
| RTWI [S25,S26] | Thinking T=.6/p=.95/k=20；Instruct T=1/p=1/k=0；论文列 max seq len=51200 | 未报告数值 pixel cap | 多轨迹预算 32 **不是**工具轮数；未报告数值 hard turn cap | inference-time 方法，无 RL。当前 `config.py` 将 Thinking ≥32B 的 T 改为1，小模型仍 .6 |

### ZwZ 代码的可追踪边界

`run_baseline.sh` 调用 `infer_without_tool.py`，默认 path 字符串为 `Qwen3/Qwen3-VL-8B-Instruct`，没有另外传 T/p/pixels；实际 inference 文件用 vLLM `LLM`、`SamplingParams` 和 `AutoProcessor`。因此可以核实**当前公共无工具评测脚本的采样与像素值**，但不能仅凭 README 证明每张论文表、每个模型都由同一 commit/命令产生。未读取到官方工具 baseline 对应的独立评测命令，故未把上述 cap 和解码值直接填给 `+tool` 行。

## 5. 用户点名工作：原版是否真的使用 Qwen3-VL

以下均读过论文原文；“未找到 Qwen3 版”只描述本次检索，不宣称不存在任何后续工作。

| 点名工作 | 已读原版 | 本次能核实的模型身份 | Qwen3-VL 2B/4B/8B 结论 |
|---|---|---|---|
| DeepEyesV2 | [S27](https://arxiv.org/html/2511.05271v1) | **Qwen2.5-VL-7B** | 已读原版没有对应 Qwen3 小模型行；其 81.8/77.9/73.8/64.9 不属于 Qwen3 |
| Thyme | [S28](https://arxiv.org/html/2508.11630v1) | **Qwen2.5-VL-7B** | 原版没有对应 Qwen3 行；Qwen3 版未核实 |
| Chain-of-Focus | [S29](https://arxiv.org/html/2505.15436) | **Qwen2.5-VL-7B** | 原版没有对应 Qwen3 行；Qwen3 版未核实 |
| Mini-o3 | [S30](https://arxiv.org/html/2509.07969) | **Qwen2.5-VL-7B-Instruct** | 原版没有对应 Qwen3 行；名为 Mini-o3 的 Qwen3 后续本次未核实；不与 OpenAI o3-mini 混淆 |
| Pixel Reasoner | [S31](https://arxiv.org/html/2505.15966v3)、[S32](https://github.com/TIGER-AI-Lab/Pixel-Reasoner) | **Qwen2.5-VL-7B**；官方 README 的发布模型为 v1 | 已读 paper/README 没有对应 Qwen3 行；“Pixel Reasoner 的 Qwen3 版”未核实 |

## 6. 一手来源与具体定位

### Qwen 官方

- **S1** [Qwen3-VL Technical Report，arXiv HTML](https://arxiv.org/html/2511.21631)；另读 [v2](https://arxiv.org/html/2511.21631v2)。§3.2.4 grounding，[0,1000]；§5.6 Fine-grained Perception；§5.11 text-centric eval；App. B.9 工具 schema。全文分两次 WebFetch 阅读。没有从图像曲线猜测 4B 数值。
- **S2** [官方 GitHub README 原文](https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md)，亦读 [仓库页面](https://github.com/QwenLM/Qwen3-VL)。generation 推荐表、processor 示例、patch_size=16 与 do_resize=False。
- **S3** [Qwen/Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。VL/Text 推荐参数；性能表是图片，本次未核实图片内目标分数。
- **S4** [4B-Instruct generation_config.json](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/generation_config.json)。do_sample=true、T=.7/p=.8/k=20。
- **S5** [4B-Instruct preprocessor_config.json](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/preprocessor_config.json)。size、patch_size、merge_size。
- **S6** [官方 qwen-vl-utils vision_process.py](https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)。utility 的 token 常数、patch 默认与像素公式。
- **S7** [think_with_images.ipynb](https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/cookbooks/think_with_images.ipynb)。WebFetch 的已读首段含工具规范和生成配置；后续大段 embedded image 未通读，不从未读部分断言不存在设置。
- **S8** [2d_grounding.ipynb](https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/cookbooks/2d_grounding.ipynb)。WebFetch 已读首段含反归一化函数、bbox JSON 和 API helper pixel 默认；embedded image 后文未通读。

### 论文与其公开实现

- **S9** [Learning Where to Look: Observation Policy Optimization for Thinking with Images，arXiv:2611.04943](https://arxiv.org/html/2611.04943)。Table 1/2；App. B 明确复现 base 也启用 zoom；App. C greedy / 10 turns；App. E schema；后半段明确 [0,1000]。分两次读取全文。
- **S10** [OPO 官方 GitHub](https://github.com/Yann0122/Observation-Policy-Optimization)。核实时 README 为 Code coming soon。
- **S11** [Zooming without Zooming，arXiv:2602.11858](https://arxiv.org/html/2602.11858)。Table 2 base、Table 4 +tool；App. 9.1–9.3。分两次读取。
- **S12** [ZwZ-2B 官方 model card](https://huggingface.co/inclusionAI/ZwZ-2B)。补充 2B base 数值，明确来源为 card，不假称原始 Table 2 包含 2B。
- **S12a** [ZwZ-4B 官方 model card](https://huggingface.co/inclusionAI/ZwZ-4B)。base 与训练后分数。
- **S13** [ZwZ 官方 GitHub](https://github.com/inclusionAI/Zooming-without-Zooming)；[run_baseline.sh](https://raw.githubusercontent.com/inclusionAI/Zooming-without-Zooming/main/mm-eval/run_baseline.sh)。
- **S14** [ZwZ infer_without_tool.py](https://raw.githubusercontent.com/inclusionAI/Zooming-without-Zooming/main/mm-eval/infer_without_tool.py)。评测采样、image cap、max_tokens。
- **S15** [ZwZ EasyR1/examples/perception_config.yaml](https://raw.githubusercontent.com/inclusionAI/Zooming-without-Zooming/main/EasyR1/examples/perception_config.yaml)。训练 min/max_pixels 与 rollout。
- **S16** [ZwZ EasyR1/single_node.sh](https://raw.githubusercontent.com/inclusionAI/Zooming-without-Zooming/main/EasyR1/single_node.sh)。YAML 路径及 prompt length 覆盖。
- **S17** [GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning，arXiv:2605.07817](https://arxiv.org/html/2605.07817)。Table 1，App. A.3/A.4，Table 4；明确 HR8K=78.0、主表 judge greedy 与模型 greedy 的区分。
- **S18** [BVS: Bayesian Visual Search with Multimodal Large Language Model for Fine-grained Perception，arXiv:2607.03184](https://arxiv.org/html/2607.03184)。Table 1 base/BVS；§4.1 search steps=5。
- **S19** [Act Wisely: Cultivating Meta-Cognitive Tool Use in Agentic Multimodal Models，arXiv:2604.08545v1](https://arxiv.org/html/2604.08545v1)。Table 1/3，App. A Figure 6 坐标。
- **S20** [Metis 官方 GitHub](https://github.com/Accio-Lab/Metis)。模型身份、训练入口。
- **S21** [Metis train_metis.sh](https://raw.githubusercontent.com/Accio-Lab/Metis/main/examples/train/train_metis.sh)。rollout 采样、agent turns 与 token caps。
- **S22** [TACO: Tool-Augmented Credit Optimization for Agentic Tool Use，arXiv:2606.30251v1](https://arxiv.org/html/2606.30251v1)。Qwen3-8B baseline/方法行、VLMEvalKit protocol 与训练 rollout。
- **S23** [Thinking Without Images: Internalizing Visual Manipulation with On-Policy Self-Distillation，arXiv:2606.08719](https://arxiv.org/html/2606.08719)。主表与后半段 self-bbox prompt，明确 MME-RW-Lite。一次请求 socket hang up，后续 WebFetch 成功；分两段读取。
- **S24** [InSight-doc: Agentic Visual Perception for Long-Document Understanding，arXiv:2608.10628](https://arxiv.org/html/2608.10628)。MME-RW-Lite、r=.25/.5、训练 token cap。
- **S25** [Reliable Thinking with Images，arXiv:2602.12916](https://arxiv.org/html/2602.12916)。Table 1 子项、Appendix C 解码；[官方仓库](https://github.com/XLearning-SCU/Reliable_TWI)。
- **S26** [RTWI/config.py](https://raw.githubusercontent.com/XLearning-SCU/Reliable_TWI/main/RTWI/config.py)。Thinking/Instruct 采样参数与 ≥32B 特例。
- **S27** [DeepEyesV2，arXiv:2511.05271v1](https://arxiv.org/html/2511.05271v1)。原版 Qwen2.5 backbone 与 Table 2。
- **S28** [Thyme，arXiv:2508.11630v1](https://arxiv.org/html/2508.11630v1)。原版 backbone。
- **S29** [Chain-of-Focus，arXiv:2505.15436](https://arxiv.org/html/2505.15436)。原版 backbone。
- **S30** [Mini-o3，arXiv:2509.07969](https://arxiv.org/html/2509.07969)。原版 backbone。
- **S31** [Pixel Reasoner，arXiv:2505.15966v3](https://arxiv.org/html/2505.15966v3)。原版 backbone。
- **S32** [Pixel-Reasoner 官方 GitHub](https://github.com/TIGER-AI-Lab/Pixel-Reasoner)。发布 checkpoint 与原版模型身份。

## 7. 本次未核实项

1. Qwen 官方 2B/4B/8B 在 V*/HR4K/HR8K/MME-RW 的精确官方图表分数；不能拿 235B 数值或第三方复现冒充。
2. OPO 的数值 eval / RL max_pixels；GazeVLM 的数值 pixel cap 与主表模型解码；Metis 的评测分辨率/解码；TACO 的数值 pixel cap/具体 decoder。
3. 所有引用当前 GitHub main 的配置，与论文实验 commit 的逐一对应关系。
4. Thyme、Chain-of-Focus、Pixel Reasoner、Mini-o3 的专门 Qwen3-VL 2B/4B/8B 后续版本。本次已读原版不能提供这类分数。

**可引用的设置总结仅限以上证据：官方 Instruct 推荐采样 T=.7/p=.8/k=20；OPO 明确 greedy；ZwZ 当前公开无工具评测用采样和最高 16,777,216 pixels。现有已读材料不支持“大家一律 greedy”或“统一 max_pixels”的表述。原生 Qwen3 grounding 为 XYXY、[0,1000]，自定义工具接口需逐文区分。**
