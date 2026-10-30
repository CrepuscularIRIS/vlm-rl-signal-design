> 阅读状态：SOURCE记录首轮10项核查及修正；本次仅打包，没有重新逐条复核。 本文件是研究笔记，不是作者原文或复现结果；涉及“我们的设置”的旧描述以本仓库 protocol 为准。机器路径已替换。

# AdaptVision (arXiv 2512.03794v2) — facts

Authors (PDF title block p.1): Zichuan Lin*, Yicheng Liu*, Yang Yang, Lvfang Tao, Deheng Ye; Tencent Hunyuan. CVPR 2026 per task message; the PDF shows "arXiv:2512.03794v2 [cs.CV] 28 Feb 2026" and does not print a venue (p.1). Code/models: github.com/adaptvision/adaptvision (p.1).
Page numbers = PDF pages (15 pages). "‡" = read off a figure / computed by me.

## 1. Setting (p.1, §4.1 p.3)
- Backbone Qwen2.5-VL-7B-Instruct (§5.1 p.6, A.2 p.11). Pure RL via veRL; no SFT / cold start is mentioned anywhere in text or appendix (searched).
- Input x = {x_sys, I_low, q}; I_low = 1/4-resolution image = 25% of the visual tokens (§4.1 p.3; Table 1 caption p.6).
- Per question the model outputs either a direct answer, or `<tool_call>[x1,y1,x2,y2]</tool_call>` and then, after receiving the high-res crop I_crop (cut from the original high-resolution image, p.2), reasons further and answers (§4.1; Eq.5 p.3). Tool = `request_local_region`, bbox in "absolute pixel coordinates of the current image" (Table 3 p.13).
- Turns: "reasoning process is constrained to a maximum of two turns" and "relies on a single visual tool and a fixed initial compression ratio (1/4 resolution)" (Conclusion p.8). So at most one crop per question. (Prompt in Table 3 says "You may call ONE function tool per step".)
- Policy factorisation (Eq.5 p.3): direct: pi(o_{1:N}|x); tool call: pi(o_{1:T}|x) * pi(o_{T+1:N}|x,o_{1:T},I_crop). o_{1:T} = "tool tokens" (first turn), o_{T+1:N} = "answer tokens" (second turn).
- Cost definition (§4.1 p.3): n_img = n_low + 1_tool * n_crop (visual tokens). Table 1 "#Token" = visual-token consumption ratio relative to the vanilla model across all benchmarks (how it is averaged: not stated).
- Case-study counts (Fig.7 p.8; Fig.9 p.11): vanilla 1768 / down-sample 442 / AdaptVision "442 + 4" and "442 + 6" visual tokens; second case 1658 / 414 / "414 + 66".

## 2. Method (§4.2 p.4, §4.3 p.5)
Reward R = R_oc + R_tool (Eq.6).
- R_oc = R_acc + R_form + R_bal.
  - R_acc: LLM judge (GPT-4o, Table 4 p.14) binary 1/0 on final answer (§4.2 p.4; A.1 p.11).
  - R_form: 0.5 for full format compliance (<think>, <answer>, <tool_call> with valid JSON), else 0.
  - R_bal (Eq.7): direct answer: -0.1 * I(r < theta) * I(R_acc = 1); tool call: -0.1 * I(R_acc = 1). r = C_direct / (C_direct + C_tool) (Eq.8) = counts of correct direct / tool-call answers within the group; theta = 0.2. Purpose stated: prevent over-reliance on tools; discourage "lucky guesses" on direct answers when correct-from-low-res probability is low.
- R_tool = R_crop - alpha * R_area (Eq.9), alpha = 2.
  - R_crop: GPT-4o judges whether the cropped region contains the primary subject / information to answer the question; 1/0 (Table 5 p.15: "check for presence, not completeness").
  - R_area (Eq.10): I(R_acc=1) * I(R_crop=1) * clip(r_a / mu_a - 1, 0, 1); r_a = (x2-x1)(y2-y1)/(H_low*W_low); mu_a = mean r_a over group G(a) of responses with correct answer AND correct crop.
  - What R_tool is for rollouts that answer directly: not stated (searched §4.2, §4.3, appendix).
- Reward of the GRPO-trained model: §4.3 p.5 says "Based on our reward design, we initially employ GRPO"; p.7 calls the w/o-DTPO row "the GRPO-trained model". So GRPO used the same reward design (R = R_oc + R_tool).
- GRPO baseline (Eq.1-3 p.3): one sequence-level advantage A = (R - mean)/std over G rollouts, loss normalised 1/G * 1/N_i per sequence. Authors' two stated problems (p.5): (i) "Ambiguous credit assignment" — one reward for tool-decision tokens and answer tokens; example given: bbox correct but answer wrong still gives a positive reward to answer tokens; (ii) "Imbalanced optimization" — tool tokens suppressed by 1/N_i and 1/G normalisation (Eq.11).
- DTPO (Decoupled Turn Policy Optimization), two changes:
  1. Loss split by turn (Eq.12 p.5): J = E[ 1/(sum_i T_i) * sum_i sum_{t<=T_i} L_{i,t}  +  1/(sum_i (N_i - T_i)) * sum_i sum_{t>T_i} L_{i,t} ]. Tool tokens and answer tokens each normalised by their own total token count over the group. Direct answers have T_i = 0, so all their tokens fall in the answer term.
  2. Advantage split (Eq.13-14): A_tool^(i) = (R_tool^(i) - mean)/std over the G rollouts; A_oc^(i) = (R_oc^(i) - mean)/std over the G rollouts. Token advantage: direct answer: A_{i,t} = A_oc + lambda * A_tool (as printed, no indicator); tool call: A_{i,t} = A_oc + lambda * A_tool * I(1 <= t <= T_i). lambda = 0.3.
  - Position: credit assignment (advantage) + loss normalisation; reward terms shaped as above. No auxiliary loss is described. G = 16 (§5.1); no extra rollouts are mentioned (my reading); GPT-4o calls for R_acc and R_crop are implied by the reward design (my reading).
- Reading (mine): 'at most one crop' is read from Eq.5 (single I_crop) and the Conclusion's "maximum of two turns"; 'no extra rollouts' is read from G = 16 being the only sampling quantity given; GPT-4o overhead is inferred from the reward design.
- Tool-token advantage for a tool-call rollout = A_oc + 0.3*A_tool; answer-token advantage = A_oc only. (reading of Eq.13)

## 3. Training data / hyper-parameters (§5.1 p.6; A.2 p.11)
- Data: training data from Yang et al. [37] (VisionThink-Smart-Train, HF link p.6 fn.1): "VQA samples that can be answered directly using low-resolution images, as well as samples that require high-resolution images". Size/filtering/masks: not stated.
- batch 512, mini-batch 32, KL dropped, lr 1e-6, G = 16 samples/prompt, T = 1.0, clip ratios 0.24 / 0.20, max prompt & response length 8192, FP16, AdamW (0.9, 0.999), wd 0.01, 4 nodes x 8 H20 GPUs, trained 80 steps (A.2). Inference: vLLM, temperature 0.
- Figure x-axes: Fig.5a runs to ~48 steps‡, Fig.5b / Fig.6a to 70 steps‡ (differs from "80 steps"; not explained).

## 4. Results as reported
Table 1 (p.6) / Table 2 (p.12): Qwen2.5-VL-7B; ChartQA test, OCRBench test, DocVQA val, MME test, MMVet test, RealWorldQA test, POPE test, MathVista testmini, MathVerse testmini. Each cell: raw score then % of vanilla. Single row per method; no seeds, no error bars; checkpoint selection not stated.
| Method | ChartQA | OCRB | DocVQA | MME | MMVet | RWQA | POPE | MathVista | MathVerse | #Token | Avg |
| Vanilla | 79.8 | 81.5 | 95.1 | 2316 | 61.6 | 68.6 | 86.7 | 68.2 | 46.3 | 100% | 100% |
| Down-Sample (25%) | 62.9 | 68.8 | 94.3 | 2270 | 54.5 | 68.8 | 82.8 | 62.2 | 43.1 | 25% | 92.1% |
| SparseVLM 50% | 73.2 | 75.6 | 66.8 | 2282 | 51.5 | 68.4 | 85.5 | 66.6 | 45.1 | 50% | 92.2% |
| FastV 50% | 72.6 | 75.8 | 93.6 | 2308 | 52.8 | 68.8 | 84.7 | 63.7 | 45.0 | 50% | 95.8% |
| VisionZip 50% | 71.5 | 70.5 | 93.8 | 2209 | 57.0 | 68.6 | 86.3 | 64.1 | 45.1 | 50% | 94.8% |
| VisionThink (their repro) | 73.6 | 76.8 | 92.9 | 2320 | 61.7 | 65.6 | 86.3 | 62.2 | 42.5 | 52% | 95.8% |
| VisionThink† (released) | 73.88 | 80.8 | 93.7 | 2392 | 60.18 | 68.37 | 86.69 | 65.7 | 45.68 | 99% | 98.4% |
| AdaptVision w/o DTPO (= GRPO-trained, text p.7) | 73.74 | 75.9 | 93.1 | 2354 | 61.28 | 65.7 | 86.8 | 64.4 | 44.2 | 57% | 96.7% |
| AdaptVision | 75.92 | 76.9 | 92.6 | 2379 | 64.8 | 67.32 | 86.8 | 65.9 | 42.3 | 33% | 97.9% |
- Table 2 (p.12) adds 70% rows and omits the "w/o DTPO" row: SparseVLM 70% avg 93.6%, FastV 70% avg 98.4%, VisionZip 70% avg 99.1% (VisionZip 70% higher Avg than AdaptVision's 97.9% at 70% tokens vs 33%).
- Per-column: ChartQA 75.92 is best among compressed/dynamic methods (VisionThink† 73.88; SparseVLM 70% 75.8) but below Vanilla 79.8; MMVet 64.8 is the highest in the table (Vanilla 61.6, VisionThink 61.7); behind VisionThink† on OCRBench (76.9 vs 80.8), MME (2379 vs 2392), MathVerse (42.3 vs 45.68; also below Down-Sample 43.1, SparseVLM/VisionZip 45.1); behind Down-Sample/FastV on RealWorldQA (67.32 vs 68.8); behind SparseVLM 50% on MathVista (65.9 vs 66.6); DocVQA 92.6 vs Down-Sample 94.3 and FastV 93.6; POPE 86.8 vs vanilla 86.7. Static baselines (FastV/VisionZip/SparseVLM) run at 50% retention, "for fair comparison" (p.6).
- Text claim (p.7): vs the down-sample model, accuracy "improves by 5.8% (92.1% -> 97.9%) with only 7% (25% -> 33%) more visual tokens". Text says 7%; the table values 25% -> 33% differ by 8 points‡.
- VisionThink: authors say the released model "consumed substantially more visual tokens than our method, making the comparison unfair", so they report both a reproduction (52%) and the released model (99%) (p.6-7).
- Latency (Fig.4 p.6): "1.67x overall speedup" vs vanilla and VisionThink†; vs down-sample model AdaptVision is slower ("additional generated tokens for reasoning and tool calls ... acceptable"); measured with per-dataset bars, no numbers in text. The 1.67x label is on the 'Overall' bars; on DocVQA_val the AdaptVision bar is above the VisionThink† bar‡ (~0.46 vs ~0.41 s, read off Fig.4); vanilla there ~0.75 s‡.
- Tool-call proportion by benchmark (Fig.6b p.7, printed labels): MathVerse 57.0%, MathVista 50.3%, ChartQA 43.5%, MMVet 38.5%, OCRBench 28.2%, MME 22.4%, RealWorldQA 14.9%, DocVQA 4.5%, POPE 3.8%.

## 5. Ablations / controls
- Fig.5a (p.7; tool-call ratio only, ~48 steps‡): full AdaptVision rises to ~0.7‡ with a dip near step 15‡; w/o balance reward -> ratio goes to ~1.0‡ ("collapse to excessive tool use"); w/o tool reward -> ratio falls to ~0‡ ("fails to invoke the tool after just 10 training steps"). No accuracy or token numbers for these two runs.
- Fig.5b (p.7; to 70 steps‡) GRPO vs DTPO: GRPO tool-call ratio drops near 0 early, rises after ~20-30 steps‡, reaches 1.0 at ~50 steps‡; DTPO rises steadily to ~0.75‡. Outcome reward at the end: GRPO ~0.87-0.92‡, DTPO ~0.77-0.82‡ (GRPO higher on the training outcome reward). Tool reward at the end: GRPO ~0.7-0.77‡, DTPO ~0.62-0.7‡.
- Fig.6a (p.7) tool-call ratio on "hard" vs "simple" training samples: GRPO -> 1.0 for both; DTPO ~0.75‡ hard, ~0.17‡ simple. How "hard"/"simple" is defined: not stated.
- Text (p.7): "the GRPO-trained model not only performs worse than DTPO, but it also uses far more visual tokens" (Table 1: 96.7% / 57% vs 97.9% / 33%). Text says GRPO "calls tools on all samples, resulting in a 100% tool call ratio" (Fig.6a, training-set curve); Table 1 reports 57% visual tokens for the w/o-DTPO model (test benchmarks). Different data (train vs test); not directly comparable.
- Not ablated (searched main text + appendix): each of the two DTPO changes alone (loss split vs advantage split); lambda, alpha, theta; R_form; effect of R_crop / R_area on final accuracy/tokens; training/test seeds or variance; matched-compute controls; the same reward given to the VisionThink baseline; GPT-4o judge reliability.

## 6. Limits stated by authors (Conclusion p.8)
- Single visual tool and fixed 1/4 initial resolution ("expanding the toolset and enabling dynamic resolution selection could further enhance adaptability").
- At most two turns, "may limit deep visual reasoning for complex tasks".
