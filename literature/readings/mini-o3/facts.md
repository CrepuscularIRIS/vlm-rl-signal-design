> 阅读状态：已有详细阅读稿；独立核查未完成；arXiv v1与会议稿的差异保留。 本文件是研究笔记，不是作者原文或复现结果；涉及“我们的设置”的旧描述以本仓库 protocol 为准。机器路径已替换。

# facts.md — Mini-o3 (arXiv 2509.07969v1)
Read from: arXiv v1 PDF (`paper.pdf`, sha256 d6bc21d118a7772fb735643bb72fb41d0182c9fb51d7beec8ec35bf537dcd036), 18 pages. Only v1 exists (abs page lists [v1] only; v2/v3 pdf URLs return 404 on 2026-10-28).
Page numbers = printed page numbers of the arXiv PDF. Items marked **[ICLR]** come only from the local ICLR 2026 proceedings text (`litscan_70410.txt`, sha256 aff5e56e2a617da9778528daadb3e60e19ae45f911b65291386c37612f70193e); they are NOT in arXiv v1.
Authors (title block p1): Xin Lai*†, Junyi Li*, Wei Li, Tao Liu, Tianjian Li, Hengshuang Zhao† (ByteDance; The University of Hong Kong). * equal contribution, † corresponding.

## 1. Setting
- Task: visual search on high-resolution images; question + image in, final answer out (§3.1, p5; Fig 3 p4).
- Per turn the policy emits thought T_i and action A_i; the environment returns observation O_i; loop ends on final answer or when context-length / turn limits are reached (§3.1 p5).
- Action space: two options, (1) grounding, (2) final answer. Grounding params: `bbox_2d` normalized in [0,1]^2; `source` in {original_image, observation_i}, i.e. the crop can be taken from any prior observation (§3.1 p5). Observation = image patch cropped from the original or a historical observation.
- Base model Qwen2.5-VL-7B-Instruct (§3.2 p6, §4.1 p9). Two phases: cold-start SFT, then RLVR with GRPO (§3.1 p5). Reward from an external LLM judge because many answers need semantic matching (§3.1 p5); judge model name not given in v1. **[ICLR]** App. E gives the judge prompt (text only, outputs "Score: 0/1"), reward = parsed 0/1.
- Training caps: 6 interaction turns, 32K context (§3.1 p5, §4.1 p9). Max pixels per image lowered 12M -> 2M (or lower) so more turns fit in 32K (§3.3 p7).
- Inference: temperature 1.0 because greedy decoding repeats (§3.4 p8).

## 2. Data
- VisualProbe: 4,000 QA pairs train + 500 test, three levels easy/medium/hard; small targets, many distractors, high-res images (§3.2 p6, Fig 4 p6, example image 7952x5304).
- RL training data: VisualProbe train + 8,000 samples randomly drawn from DeepEyes-Datasets-47k "to preserve performance on simpler visual search cases" (§4.1 p9).
- Test suites: VisualProbe-test, V* Bench, HR-Bench, MME-Realworld (§4.1 p9).
- Paper says RL-only (no SFT) "tended to produce concise responses and trajectories with few turns" (§3.2 p6).

## 3. Cold-start (SFT) trajectory generation — close-read
- Pipeline (Fig 5 p6; §3.2 p6-7; Intro p2): hand-craft a small set of exemplars (input image + question + per-turn observation/thought/action). Intro p2: exemplars "cover varied reasoning strategies, including depth-first search, self-reflection, and goal maintenance". An existing VLM with in-context ability is prompted few-shot to "mimic" and produce thought + action at each turn on new queries; the action is executed in the environment to produce the next observation (Fig 5: Multi-turn Examples + Observation O_i -> VLM -> Thought T_i/Action A_i -> Execute Action -> O_i). Loop "terminates upon emitting a final answer or reaching a pre-defined turn limit" (§3.2 p7).
- Filter, verbatim (§3.2 p7): "We retain only trajectories whose final answers are correct." Intro p2: "Only trajectories that culminate in a correct answer are retained."
- Size: "approximately 6,000 cold-start trajectories from 6 exemplars" (§3.2 p7).
- "the base VLM used for data synthesis need not possess native thinking-with-image ability; in-context mimicking suffices" (Intro p2-4).
- Intro calls the pipeline "iteratively synthesize"; the abstract says "iterative data collection pipeline". The v1 text describes the per-turn loop; no multi-round regenerate/refine stage is described.
- SFT hyper-params (§4.1 p9): ~6,000 samples, 3 epochs, lr 1e-5, global batch 32, max pixels 2M unless otherwise specified.
- NOT stated in v1 (searched main text + appendix; appendix A holds only three trajectory illustrations, Fig 9-11 p16-18): identity of the generator VLM; the prompt text; the exemplar contents; the turn limit used in generation; whether the generator sees the ground-truth answer; how "correct" is judged for filtering (exact match vs LLM judge); the source/number of questions the 6,000 are drawn from (VisualProbe train has 4,000); per-pattern or per-turn-count statistics of the 6,000; any dedup/length/validity filter; which tokens carry the SFT loss.
- **[ICLR] App. F (text pp.17 ff.)**: prompt template (Table 6): agent has two actions `grounding([x1,y1,x2,y2], source)` and `final(answer)`; format "Thought: ... Action: ...". Instruction for thought generation (Table 8), verbatim: "You should mimic human to think carefully and comprehensively. Output the thought as detailed as possible, along with some useful description of the observation, suspective conjectures, and underlying world knowledge that might be useful for problem solving." "If the target object is present in observation i, please analyze whether final answer is ready or further grounding is needed. Otherwise, you can go back to either original image or any of the history observations to find another region. In this case, clarify this thinking process, and analyze how the old bounding box deviates from the correct one, and how you obtain the coordinates of the new bounding box in Thought i." Output as JSON {"thought","action"}. Figures 9-12 show 4 in-context examples (the text still says 6 exemplars).
- **[ICLR] App. D**: general-VQA data (filtered from LLaVA-OneVision and Cambrian-1) added to both cold-start and RL stages; "during the cold-start phase, each VQA sample is paired with a reasoning trajectory generated by Gemini". Not in arXiv v1.
- **[ICLR]** The generator for the main (VisualProbe) cold-start trajectories is still "an existing VLM" (no name).

## 4. RL and over-turn handling — close-read
- Optimizer: GRPO (Eq 1, 2 p7) with DAPO tricks: clip-higher, dynamic sampling, token-level policy loss; group size G=16; clip 0.30/0.20; global batch 256, mini-batch 32; lr 1e-6 constant; no KL, no entropy reg.; async rollouts (§4.1 p9).
- Eq 2: A_i = (r_i - mean({r})) / std({r}).
- Vanilla problem (§3.3 p7): when a response hits max turns or exceeds context, "the reward is set to 0, as no valid answer can be produced"; after normalisation this gives negative advantages, i.e. such responses "are penalized and discouraged". Two stated drawbacks: (1) correctness of over-turn responses is unknown -> label noise in the return; (2) with a modest training turn cap (<10), over-turn responses are frequent ("exceeding 20% at the beginning of training") and "naive penalization biases the model to answer prematurely, substantially suppressing the number of interaction turns (see Fig 6)".
- Over-turn masking (Eq 3, 4; Fig 7 p8): completion mask M_i; A'_i = M_i * A_i. Objective: J = E[ (1 / sum_i M_i) * sum_{i=1..G} min( ratio_i * A_i * M_i , clip(ratio_i, 1-eps, 1+eps) * A_i * M_i ) ]. M_i = 1{|o_i| <= C_context} * 1{turn(o_i) <= C_turn}. "we normalize the objective by the number of completed generations, sum_i M_i, rather than by the total number of generations G."
- Position: advantage/loss mask on over-turn trajectories (zero gradient contribution); reward definition, rollouts count and compute unchanged.
- Not stated in v1: whether mean/std in Eq 2 are computed over all G (including the r=0 over-turn samples) or only completed ones; how Eq 3's 1/sum M interacts with the DAPO token-level loss; Eq 4 uses "<=" so a response that is cut off at exactly C_turn is not distinguished from one that answers at turn C_turn in the equation as written (text says responses that "hit" the max are masked).
- No turn/crop cost term in the reward: r depends on answer correctness (§3.3 p7: "Rewards r are then computed based on the correctness of the responses"). No turn penalty is described.
- Training turn budget: Intro p4: "reducing the training budget from 16 to 6 turns shortens a 10-day training run to about 3 days, with negligible impact on test accuracy". §4.3 p10 / Fig 8 p11 compares 6 vs 12 turns: "A lower budget leads to faster initial convergence, but the performance plateaus after approximately 150 steps. In contrast, a higher turn budget attains a superior performance ceiling, albeit with slower convergence." (text says 16 in intro, 12 in §4.3; "negligible" vs "superior ceiling"; both reported as is.)

## 5. Results (as reported)
Table 1 (p9), all 7B. VisualProbe and V*: Avg@32; HR-Bench: Avg@8; MME-Realworld: Avg@1. Columns: VP-hard / VP-medium / VP-easy / V* / HR-4K / HR-8K / MME-RW.
- GPT-4o 11.2 / 15.4 / 47.5 / 65.2 / 62.0 / 58.3 / 45.2
- LLaVA-OneVision 13.4 / 12.5 / 36.2 / 70.9 / 61.2 / 54.0 / 57.4
- Qwen2.5-VL-Instruct 23.9 / 26.0 / 39.1 / 75.5 / 68.2 / 62.7 / 57.3
- SEAL† V* 75.4; DyFo† V* 81.2; Chain-of-Focus† V* 88.0 († = only report Avg@1, weights unavailable)
- Pixel Reasoner‡ 28.8 / 29.6 / 58.4 / 86.3 / 74.0 / 66.9 / 64.4
- DeepEyes‡ 35.1 / 29.8 / 60.1 / 83.3 / 73.2 / 69.5 / 64.0 (‡ = re-evaluated with official model/code, Avg@32)
- Mini-o3 48.0 / 50.4 / 67.0 / 88.2 / 77.5 / 73.3 / 65.5
- Mini-o3 is the highest in every column of the table; V* margin over Chain-of-Focus (Avg@1, different metric) is 0.2 (88.2 vs 88.0); MME-RW margin over Pixel Reasoner 1.1‡ (65.5 vs 64.4).
- Table 1's Mini-o3 row equals the Table 3 row "2M" (48.0 / 50.4 / 67.0): Table 3 is evaluated on the VisualProbe test set, the same set as Table 1's VisualProbe columns. The paper does not say the max-pixel value was chosen on a separate split.
- Headline row = one trained model; number of training seeds not stated; Avg@K averages K samples at temperature 1.0 (§4.1 p9), not training seeds.

Table 2 (p10): max pixels 1M, training cap 6 turns, VisualProbe test. Columns: hard RL data / cold-start / over-turn -> Hard / Medium / Easy / Avg turns (correct):
- ID1: no hard RL data, cold-start yes, over-turn yes: 35.8 / 46.4 / 66.7 / 4.8
- ID2: hard RL data yes, cold-start NO, over-turn yes: 25.4 / 18.7 / 57.3 / 1.0
- ID3: hard RL yes, cold-start yes, over-turn NO: 32.2 / 45.7 / 61.1 / 3.0
- ID4: all yes: 44.4 / 47.9 / 67.4 / 5.5
Text: removing hard RL data -8.6 on Hard ("approximately 8.6 points", 44.4-35.8 = 8.6); cold-start: "performance collapses without it"; over-turn: comparison ID3 vs ID4. No SFT-only (pre-RL) row, no RL-without-masking-but-penalising-differently row, no row varying SFT data size / filtering / exemplars / generator.

Table 3 (p10): max pixels | Hard | Medium | Easy | Avg turns (all) | Avg turns (correct): 0.5M 36.4/44.8/64.8/8.0/6.7; 1M 44.4/47.9/67.4/6.3/5.5; 2M 48.0/50.4/67.0/6.5/5.6; 12M 36.1/40.7/62.1/1.0/1.0. Text: large budget "induces premature 'early stopping'", small budget "increases perceptual hallucinations".

Fig 1 (p2): left, Accuracy on VisualProbe-Hard (Avg@32) vs test-time turn limit 4/8/16/24/32. Read off ‡: Mini-o3 ~25 / ~41 / ~47 / ~48 / 48.0 (final stated 48.0); Mini-o3 w/o over-turn mask ~38 flat; DeepEyes ~35 flat (35.1 stated). Right (printed labels): DeepEyes Correct 35.1%; w/o masking Correct 38.3% (38.1% in 0<=turns<4, 0.2% in a further bucket); Mini-o3 Correct 48.0% = 25.3% (0-4) + 15.3% (4-8) + 6.4% (8-16) + 0.8% (16-24) + 0.1% (24-32).
Fig 6 (p7): share of *correct* responses by turn range on VisualProbe-Hard: w/o masking 69.4 / 29.5 / 1.1 / 0 / 0 ; Mini-o3 53.3 / 32.4 / 12.2 / 1.9 / 0.1 for [1,4) [4,8) [8,16) [16,24) [24,32).
- Inconsistency (reported neutrally): Fig 1 right shows the w/o-masking correct responses almost all in the 0-4 bucket (38.1 of 38.3), Fig 6 shows 29.5% of them in [4,8). Mini-o3 shares also differ slightly between Fig 1 right (normalised by 48.0‡: 52.7/31.9/13.3/1.7/0.2) and Fig 6 (53.3/32.4/12.2/1.9/0.1). Fig 1's pixel budget is not stated.
- Where gain is absent: at the test limit of 4 turns Mini-o3 is ~25‡ on VisualProbe-Hard, below w/o-mask ~38‡ and DeepEyes ~35‡ (Fig 1 left read-off).
- Abstract claims accuracy "improving as the number of turns increases", trajectories "spanning tens of steps"; Fig 2 example 11 turns; Fig 11 caption: "after 18 reasoning turns" (p18).

Costs reported: avg turns (Tables 2, 3), training wall-clock 10 days -> ~3 days (16 -> 6 turn cap, Intro p4). No image/visual-token counts, no generated-token counts, no per-question crop counts other than turns, no extra-rollout accounting.

## 6. Controls NOT run / not reported (v1 main text + appendix searched)
- Matched-compute comparison between Mini-o3 and DeepEyes/Pixel Reasoner (they use their own data/training); DeepEyes trained with RL on its own data, not given the SFT/over-turn mask.
- Alternatives to masking (e.g. soft penalty, excluding over-turn from mean/std, longer training cap at same compute).
- Ablation of cold-start filtering rule, exemplar number/content, generator model, SFT set size, SFT-only result before RL.
- Seeds / variance across training runs.
- Evaluation of whether intermediate thoughts/bboxes in SFT trajectories are correct.

## 7. Limitations stated by authors
v1 has no limitations section (sections: Intro, Related Work, Approach, Experiment, Conclusion, Appendix A). Stated observations: over-large pixel budget induces early stopping and small budget increases perceptual hallucination (Table 3 text, p10); greedy decoding repeats (§3.4); higher train-turn budget gives higher ceiling but slower convergence (p10).
