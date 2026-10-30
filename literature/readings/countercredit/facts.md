> 阅读状态：已有详细阅读稿；原核查中断，不能标成全文独立复核通过。 本文件是研究笔记，不是作者原文或复现结果；涉及“我们的设置”的旧描述以本仓库 protocol 为准。机器路径已替换。

# facts.md — CounterCredit (arXiv 2609.22910v1), "When Should a VLM Look? Paying Only for Visual Calls That Were Needed and Used"
Authors (title block p.1): Kunyu Peng, Junming Liu, Ruiqi He, Qingzhuo Wang, Jianzhong Qi, Xianhui Liu (Tongji Univ.; Nankai Univ.; Univ. of Melbourne). Preprint, 19 Sep 2026, cs.AI. Page numbers = PDF/printed pages (30 pp).

## 1. Setting
- Agent: Qwen2.5-VL-7B-Instruct, cold-start SFT then GRPO (§4.1 p.6). One tool: crop of the image (§3.1 p.3). Per turn: <think> reasoning, then either <grounding>{"bbox_2d":..., "source": original_image|observation_i}</grounding> (crop, returned as image turn) or <answer>..</answer> (App. E p.21-22). Mini-o3 chat format.
- Outcome r_out(τ) ∈ {0,1}: exact/option-text match; free-form fails judged by GPT-4o (99.4% agreement on 500 hand-labelled, 497/500; App. D p.20).
- Max 6 tool interactions per rollout (n_max = 6; App. C p.19).
- Problem claim (Abstract p.1): on the cold-start checkpoint only 10% to 12% of visual calls were both needed and used (Table 7 base rates "every call paid": both = 11.8 V*, 9.8 HR-4K, 10.4 HR-8K, p.23); outcome reward pays calls that are neither needed nor used for about two thirds of its payments (Fig.1b).

## 2. Method (§3, pp.3-6; App. A, C)
Probe at each executed image-returning call i, from its realized pre-call prefix h_i (prompt + earlier turns + reasoning; h_i ends with the <think> block before call i; c_i = the <grounding> block; o_i = returned crop; App. E p.21).
Three continuations of h_i (Fig.2a p.4, §3.1 p.3):
 1. answer now: (h_i)
 2. real observation: (h_i, c_i, o_i)
 3. random observation: K=3 (h_i, c_i, õ_ik): same c_i text, returned pixels replaced by an unscreened same-size patch sampled uniformly at random from the original image, same format.
Score (Eq.1, p.3): U_θ(p) = π_θ(y | p,e) / Σ_{a∈O} π_θ(a | p,e); y = gold option; e = fixed answer elicitation appended as user turn ("Based on everything so far, answer the original question now. Reply with the option letter only."), assistant turn prefilled with "<answer> " (App. E p.22). Same e/prefill for all three continuations. Free-form: mean per-token log-prob of reference answer under teacher forcing (nats/token), separate deadzone ε_free = 0.44 (App. C p.19, App. D p.21). Probe scores: current policy, gradients stopped (p.4). Each probe is a forward pass without generation (App. C p.20: "one forward pass without generation per evaluation"). The probes do NOT sample continuations to the end of the episode.
Eq.2: D_i = U_real − U_now (decision value).  Eq.3: E_i = U_real − mean_k U_rand,k (evidence value).  Eq.4: q_i = min{E_i, D_i}. Prop.1 (p.5): q_i = inf_λ [U_real − (λU_now + (1−λ)Ū_rand)].
Eq.5: v_i = r_out(τ)·a_i·1[q_i > ε]; a_i = eligibility (valid single image return, distinct patch support, finite scores). ε = 0.05 (MC), from null calibration (leave-one-out random-patch null, 500 train-pool questions; false-verification 2.39% V*, 2.51% HR-4K; App. C p.19, Fig.6).
Eq.6 (p.5): P(τ) = min{ C, γ Σ_{i∈V(τ)} (q_i − ε) } − ρ ( n(τ) − |V(τ)| ); R(τ) = r_out(τ) + P(τ).
 Constants γ=0.5, ε=0.05, ρ=0.05, C=0.7, K=3, n_max=6, σ_min=0.15, G=8 (Table 5 p.17; App. C p.18). Every unverified executed call pays rent ρ, including calls on wrong/unanswered trajectories, parse failures. Wrong/unanswered trajectories run no probes (App. C p.19; Alg.1 line 4).
Prop.2 (p.5): ρ n_max<1 ⇒ correct trajectory R ≥ 1−ρ n_max = 0.7; wrong/unanswered R = −ρ n(τ) ≤ 0. Prop.3 (p.6): chance verification has negative expected price if false-verification rate α < ρ/(ρ+γ(1−ε)) ≈ 9.5% (premise measured, not enforced; A.8 p.17-18).
Eq.7 (p.6) dual-channel advantage: Â_j = (O_j − Ō_g(j)) / max{std(O_g(j)), σ_min} + (P_j − P̄_g(j)); outcome channel group-normalized, price channel centered only; if no rollout in the group is correct, price advantage set to 0 (p.6). Â_j is indexed by rollout j only; fed into the clipped GRPO objective (p.6). The paper does not state a per-token or per-call split of Â_j (searched §3.5, App. A.6, App. C, Alg.1). Per-call credits are summed into P_j (Eq.6, Alg.1 lines 9-15).
Cost per correct trajectory with m eligible calls: m(K+2) score evaluations (p.6). Step time 2,972 s (CounterCredit) vs 2,881 s (outcome-only GRPO, which also runs probes for logging only) on 8×H100; ratio 1.03. Probes-disabled run 2,320 s per step: probing adds 24%, CounterCredit = 1.28× a standard outcome-only pipeline. ~2,200 eligible calls probed per step (App. C pp.19-20). Probes run only during training; inference has no probe cost.
Position in our taxonomy: task reward (Eq.6 adds P(τ) to r_out) AND advantage normalization (Eq.7 dual channel); no auxiliary loss; no SFT/data change from the probes. Requires gold answer y at training time.

## 3. Data (§4.1 p.6; App. B p.18)
- Cold start: 6,960 of 7,267 released Mini-o3 trajectories after filtering; 3 epochs, lr 1e-5, global batch 32, all params except vision encoder & projector, 2M max input pixels.
- RL pool: 16,000 prompts from VisualProbe train (4,000) + 24,000 sampled DeepEyes prompts (16,800 visual search, 3,600 ArxivQA, 3,600 ThinkLite-VL) = 28,000 → dedup 26,600 → decontaminate (pHash Hamming ≤5 vs all eval images) 26,200 → format filter 25,700 → difficulty filter (8 samples from SFT ckpt; drop 0/8 correct, keep 1-7, keep random half of 8/8) = 16,000. Same list for every RL arm.
- RL: GRPO 3 epochs over pool, 188 steps, 256 prompts/step, 8 samples/prompt, mini-batch 32, lr 1e-6 constant, no KL/entropy, T=1.0, ≤6 tool interactions, 8×H100 80GB (App. C p.19).
- Baselines: 7 released tool agents re-evaluated (DeepEyes, Pixel-Reasoner, Mini-o3, Thyme, CodeV-RL, DeepEyesV2, PyVision-RL); single runs, spread not measured (Table 1 caption p.7).

## 4. Results as reported
Table 1 (p.7), accuracy %, Qwen2.5-VL-7B base; columns V*, HR-4K, HR-8K, MME-RW, MMStar, ChartQA, BLINK, Avg (7):
- Qwen2.5-VL-7B (no tool): 77.0 / 69.1 / 65.6 / 58.0 / 64.4 / 83.9 / 56.2 / 67.7
- Mini-o3: 87.4 / 78.0 / 73.4 / 65.2 / 64.1 / 85.3 / 55.6 / 72.7
- PyVision-RL: 86.9 / 77.3 / 73.8 / 62.4 / 65.7 / 86.2 / 57.9 / 72.9
- CounterCredit (seed 0; ±SD over 3 seeds): 89.5±0.81 / 80.2±0.75 / 76.4±0.90 / 67.7±0.50 / 68.9±0.35 / 87.1±0.30 / 62.8±0.65 / 76.1±0.40
- Not tool-using rows that are higher than CounterCredit in some columns: Gemini-2.5-Pro HR-4K 84.2, HR-8K 81.1, MMStar 73.4, BLINK 73.6 (Avg 76.3); Qwen3-VL-8B ChartQA 88.4, MMStar 70.5, BLINK 68.9. GPT-5: lower than CounterCredit on V*/HR-4K/HR-8K (72.3/75.6/73.8; text p.7). Text says "leads every tool agent on all seven" (p.7; the agents "differ in tools, prompts, and interaction formats").
Table 2 (p.8): calls/question V*, HR-4K, HR-8K and spurious-call rate (min{E,D} ≤ ε, each agent under its own gold-answer score): CounterCredit 1.78/1.76/1.87 calls; 31.4/35.1/36.3 %. Mini-o3 2.50/2.46/2.60; 40.7/50.5/51.0. DeepEyes 0.91/0.98/0.96; 87.4/83.6/82.1. Thyme 0.14/0.19/0.22 calls; spurious 36.1 (V*), 77.2, 75.3. Latency (s, H20, bs 1) CounterCredit 5.4/8.8/10.0; Mini-o3 7.3/12.1/13.4; Pixel-Reasoner 2.0/2.0/3.0; Thyme 2.3/2.6/2.9. Spurious-rate metric is built from the same two contrasts the reward trains on (§5 p.10, authors' limit).
Table 3 (p.8; all RL rows from SFT ckpt, same pool & budget, differ only in reward; single seed shared with seed 0), columns V*, HR-4K, HR-8K, MME-RW, MMStar, ChartQA, BLINK, Avg, V* calls, V* spurious %:
- SFT cold start: 71.7/70.4/61.6/55.6/60.3/80.8/54.5; 65.0; 2.68; 88.2
- Outcome-only GRPO: 80.1/73.9/69.4/56.8/61.7/86.4/54.3; 68.9; 1.84; 82.1
- Branch reward (TACO-style): 85.1/76.2/72.0/60.7/62.8/86.9/57.4; 71.6; 2.45; 62.9
- Verify on E only: 82.2/74.6/68.3/59.0/62.3/86.4/53.1; 69.4; 1.46; 71.3
- Verify on D only: 84.3/76.5/72.5/60.5/63.5/86.5/55.5; 71.3; 1.65; 65.0
- γ=0 (no cashback): 81.2/74.6/70.2/59.4/62.9/86.3/55.4; 70.0; 1.35; 62.2
- ρ=0 (no rent): 85.9/77.9/74.9/63.0/65.3/86.9/58.0; 73.1; 2.70; 48.7
- Single-channel GRPO: 62.8/68.8/63.9/57.1/60.0/83.7/55.3; 64.5; 0.78; 62.5
- CounterCredit: 89.5/80.2/76.4/67.7/68.9/87.1/62.8; 76.1; 1.78; 31.4
 Text (p.9): "Verifying on E or on D alone recovers 0.5 and 2.4 of the 7.2-point gain"; TACO-style recovers 2.7; removing cashback recovers 1.1; removing rent recovers 4.2. Single-channel arm "also lacks the floor σ_min, so more than the normalization differs between the two" (p.9).
 Abstract: "6.3 to 9.4 points above outcome-only GRPO at 1.78 against 1.84 calls per question".
Table 9 (App. F.2 p.24): accuracy when trajectory is stopped at first tool call and answer elicited from cached prefix (no call / random / real / full rollout): V*: SFT 75.4/73.8/75.4/71.7; Outcome-only 77.5/74.7/76.4/80.1; CounterCredit 78.5/74.3/85.3/89.5. HR-4K: SFT 71.6/68.9/71.9/70.4; Outcome-only 73.2/69.9/72.9/73.9; CounterCredit 73.5/70.5/77.8/80.2. Text p.9 (Fig.5a): real vs no call: 0.0 (SFT), −1.1 (outcome-only), 6.8 (CC); real vs random: 1.6, 1.7, 11.0.
Table 10 (p.24) V* by target-size tercile, acc/calls: SFT 65.1/3.33, 70.3/2.84, 79.7/1.88; Outcome-only 66.7/2.38, 89.1/1.77, 84.4/1.39; CC 82.5/2.19, 92.2/1.70, 93.8/1.45.
Table 4/11 (pp.9, 25): Qwen3-VL-8B base 85.9/79.2/74.3/60.7/70.5/88.4/68.9 Avg 75.4 → +CounterCredit 91.6/85.4/84.0/70.1/72.2/90.0/72.3 Avg 80.8. "no reward ablation was run on Qwen3-VL-8B" (p.9).
Table 12 (p.25) three seeds Avg/calls/spur%: seed0 76.1/1.78/31.4; seed1 76.4/1.84/33.3; seed2 75.6/1.72/29.5; mean 76.0; SD 0.40.
Table 14 (p.25) V*/HR-4K acc, calls, spur%: reference 89.5/80.2, 1.78/1.76, 31.4/35.1; rent only on correct 87.4/78.5, 2.27/2.23, 44.1/47.2; γ=0.25 87.4/78.6, 1.57/1.55, 36.2/39.7; γ=1.0 88.0/79.1, 2.05/2.02, 38.4/41.8; ρ=0.02 86.4/76.2, 2.20/2.17, 42.3/45.4; ρ=0.1 86.9/78.4, 1.50/1.48, 34.7/38.1. "every departure costs 1.5 to 3.1 points on V*" (p.24).
Table 7 (p.23) audit of existing rewards on SFT trajectories (2,594 scored calls): outcome reward pays 40.2/39.2/33.7 per 100 calls, of paid calls passing both 12.8/12.4/14.6 % (V*/HR-4K/HR-8K); TACO branch probe pays 5.3/5.6/6.7 per 100, both 41.7/41.7/49.7 %; external judge both 21.0/19.3/22.4; self-judge 14.7/20.9/22.6. Only TACO's branch reward was retrained (Table 3); CodeV/FaithEyes judges are audited only, not retrained (§4.1 p.7, Fig.1b).
Table 8 (p.23) answer-level counterfactual V*, baselines pooled, decoded answer changed under random patch: Real 33.3%, Performative 2.2%, Unnecessary 17.1%, Waste 2.0%; authors: "corroboration rather than an independent test".
Table 13 (p.26) sensitivity of "used" label: K=1 agreement 94.7/94.8 (V*/HR-4K), K=2 97.5/96.7, K=5 97.8/97.1, gray patch 90.0/90.0, blurred real crop 88.5/89.5, shifted patch 97.5/95.9.
Cost metrics reported: crops (calls) per question; latency (s/question); extra training cost (1.28× vs standard pipeline; 1.03 vs probes-logged outcome-only). Not reported: separate first-vs-later crop counts, image/visual tokens, generated tokens (searched; "token" appears only in log-prob context).

## 5. Where gains are absent / smaller
- Table 9 first-call stop: outcome-only and SFT arms: real crop −1.1 / 0.0 points vs answering at once (V*) (p.9, Table 9).
- Fig.5b/Table 10: gain concentrated on smallest-target tercile (+15.8 over outcome-only) vs +3.1 and +9.4 on middle and largest (p.9).
- ChartQA (Table 1): CounterCredit 87.1 vs base 83.9; DeepEyesV2 86.3, PyVision-RL 86.2; Table 3 arms 86.3-86.9 (outcome-only 86.4).
- Table 1: Qwen3-VL-8B (no tool) higher on ChartQA 88.4, MMStar 70.5, BLINK 68.9; Gemini-2.5-Pro higher on HR-4K, HR-8K, MMStar, BLINK.
- §4.1: baselines re-evaluated "in the same pipeline" (p.6); baselines are single runs; CounterCredit headline = seed 0 (Table 12 seed 1 has Avg 76.4; seed 2 75.6).
- Controls not found in the paper (searched App. A-G): ablation arms with multiple seeds (Table 3/14 are single-seed, shared seed 0); reward ablation on Qwen3-VL-8B ("no reward ablation was run", p.9); retraining CodeV/FaithEyes/AdaTooler/AXPO-style rewards; a random-crop-only or no-gold-score version of the probe; a probe variant using sampled continuations (rollouts) instead of immediate elicitation; matched-compute comparison with a mechanism-given baseline other than the probes-for-logging outcome-only arm.

## 6. Limits the authors state
- §5 p.10: "The decision value does not separate the information in the crop from the effect of eliciting an answer at a prefix where the policy had decided to crop." Evidence value defined only for crops (matched null return); text/structured tools need their own null. "The efficiency metric is built from the two contrasts the reward is trained on, so an evaluation defined apart from the reward would settle the question."
- App. G p.30: "Per-call values do not sum to the value of the trajectory." "Verification weakens when the gold-answer score saturates" (also inflates spurious rate of an agent scored by its own saturated distribution). Rent on unnecessary (confirmatory) calls "penalizes confirmatory looks; whether such looks aid robustness under distribution shift was not tested." "Random patches are unscreened, so a patch that contains the target raises the reference mean and lowers E_i; the reference errs toward under-verification." MC uses renormalized probability, free-form log-probability, separately calibrated deadzone. Open: tools returning text, evidence accumulated across several calls, further base families, "longer training in which the policy could adapt to the probe."
- App. A.8 p.17-18: all scores are the policy's own gold-answer probabilities under a fixed elicitation; "We make no claim that the counterfactual values identify a causal effect of the pixels on the answer, and no convergence claim for the optimizer"; Prop.3 premise is a proxy (null calibration), taken on the cold start, not controlled during training.
- App. A.5 p.16: Hoeffding bound with K=3 is vacuous at t=ε; deadzone calibrated on null draws instead.
- Table 6 note: "gaming of the scoring policy itself" is outside the table.

## 7. Text vs table notes
- Abstract/§1 say "10% to 12%"; §5 says "9.8% to 11.8%"; Table 7 base rates both: 11.8, 9.8, 10.4.
- Table 4 caption says Avg is over all seven benchmarks but only four columns are shown; Avg values equal Table 11 / Table 1 values.
