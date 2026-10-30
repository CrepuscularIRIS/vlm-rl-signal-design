> 阅读状态：SOURCE记录首轮10项核查及修正；本次仅打包，没有重新逐条复核。 本文件是研究笔记，不是作者原文或复现结果；涉及“我们的设置”的旧描述以本仓库 protocol 为准。机器路径已替换。

# SOURCE — AdaptVision

Read from: arXiv 2512.03794v2 (https://arxiv.org/pdf/2512.03794v2), downloaded to [local-source] sha256 9abd3f7986659b70f026c84cedac3104a1016680c08f6572efb31908b311da4a (15 pages, PDF dated 28 Feb 2026 on the arXiv stamp).
Paper figures shown: none
Style: palette forest · layout sidebar

Venue: per the task message (our records) CVPR 2026; the PDF itself prints no venue line, so the poster shows only "arXiv v2 · 28 Feb 2026".

## Provenance of numbers
- Table 1 (p.6) and Table 2 (p.12): all benchmark scores, #Token, Avg (read visually from page images; text extraction scrambles column order).
- Eq.5 p.3; Eq.6-10 p.4; Eq.11-14 and Fig.3 p.5; hyper-parameters §5.1 p.6 and A.2 p.11; prompts Tables 3-5 pp.13-15.
- Fig.6b labels (57.0 ... 3.8%) are printed on the figure, p.7.
- Fig.4 "1.67x" is in the figure and §5.2 text (p.6-7).

## ‡ numbers (not stated as numbers in the paper)
- "8 points" (33% - 25%, vs the text's "7%"): computed from Table 1 #Token column.
- Fig.5a (p.7), read off the curves: full model tool-call ratio ~0.7 at the end; run length ~48 steps; w/o balance reward -> ~1.0; w/o tool reward -> ~0 (text says "after just 10 training steps").
- Fig.5b (p.7), read off the curves: x-axis to 70 steps; GRPO tool-call ratio reaches 1.0 at ~50 steps; DTPO ~0.75 at the end; outcome reward at the end GRPO ~0.87-0.92 vs DTPO ~0.77-0.82 (poster shows "约 0.9 对 0.8"); tool reward at the end GRPO ~0.7-0.77, DTPO ~0.62-0.7.
- Fig.6a (p.7), read off: DTPO ~0.75 (hard) and ~0.17 (simple); GRPO 1.0 (text states 100% for GRPO).
All ‡ values are approximate eyeball reads of line plots.

## Reading notes
- "AdaptVision w/o DTPO" is described in the text (p.7) as "the GRPO-trained model". §4.3 p.5 "Based on our reward design, we initially employ GRPO" and p.7 "the GRPO-trained model": same reward design, single sequence-level advantage (Eq.1-3 + Eq.6).
- Eq.13 is quoted as printed: the direct-answer case has no indicator, so A_tool applies to all tokens; the tool-call case has I(1 <= t <= T_i). R_tool for direct-answer rollouts is not defined in §4.2 / §4.3 / appendix (searched).
- "Our setting" comparison lines use only the facts in [local-source] the first-turn input resolution of our setup is not given there and is posed as a question.
- Searched for SFT / cold-start / seeds / variance / checkpoint selection / ablations of lambda, alpha, theta: none found in text or appendix.
- Text vs table: text says "only 7% (25% -> 33%) more visual tokens" (p.7); table values differ by 8 points. Training length: A.2 says 80 steps; Fig.5 curves end near 48 and 70 steps.
- GRPO "100% tool call ratio" (Fig.6a, training-set curve) and Table 1's 57% visual tokens (test benchmarks) come from different data; not directly comparable (not a demonstrated contradiction).

Files: facts.md (page-level facts), 精读.zh.md, poster.png/.pdf, make_poster.py.

## Fact-check round 1 (checker 1; each item re-verified in the PDF)
1. ChartQA "最高" wrong — APPLIED. Table 1 p.6: Vanilla 79.8 > 75.92. Now: best among compressed/dynamic methods (also > SparseVLM 70% 75.8, Table 2 p.12), below Vanilla; MMVet 64.8 highest in table (Vanilla 61.6). Edited 精读, facts.md.
2. Q1 on A_tool normalisation — APPLIED. Eq.13-14 p.5 normalise over {R_tool^(i)}, i=1..G. Q1 now asks only what R_tool is for a direct answer (undefined in §4.2/§4.3/appendix).
3. DocVQA_val latency exception — APPLIED. Fig.4 p.6 (looked at 300 dpi crop): on DocVQA_val the orange AdaptVision bar is above the green VisionThink† bar (‡ ~0.46 vs ~0.41 s, eyeball); 1.67x is the Overall bar. Added to 精读, facts.md, poster panel 5.
4. "7%" vs 8 points — no change needed (already reported neutrally with table value; confirmed Table 1/2 print 25% and 33%).
5. GRPO 100% vs 57% — APPLIED. Fig.6a/5b are training curves, Table 1 is test benchmarks; reworded to "different data, not directly comparable" in SOURCE, facts.md, 精读; poster panel 4 now labels "训练曲线" vs "表 1（测试集）".
6. 80 steps vs Fig.5 curves — no change (already flagged in facts.md/SOURCE; poster "70 步" matches the figure; paper does not explain).
7. CVPR badge — APPLIED. Badge replaced with "arXiv v2 · 28 Feb 2026" (p.1 stamp); venue disclosure kept in this file.
8. Readings marked — APPLIED. Poster: "纯 RL" -> "veRL 强化学习"; "至多一次裁图" marked as my reading (Eq.5 single I_crop; p.8 "maximum of two turns"); "无额外 rollout" -> "G = 16（未提额外 rollout，我的读法）"; 精读: GPT-4o overhead and "纯 RL" marked as my reading.
9. Translated quote marks — APPLIED. 精读 p.8 limitation now the English original; other quoted translations (信用含糊, 优化失衡, 检查存在而非完整, 猜对, 低分辨率即可答) replaced by the verbatim English phrase or plain description.
10. Small mismatches — APPLIED. 精读 "not done" list now includes "给 VisionThink 同样奖励" and the two DTPO changes ablated separately; SOURCE note on w/o-DTPO reward corrected (§4.3 p.5); poster panel 6 now says "工具调用时答案 token 只得 A_oc"; 精读 now also lists FastV 70% (98.4%) alongside VisionZip 70% (99.1%).
Checker's "Verified clean" items were not re-opened beyond spot checks of Table 1 and Eq.13-14.
