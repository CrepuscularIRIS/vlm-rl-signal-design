> 阅读状态：已有详细阅读稿；独立核查未完成；arXiv v1与会议稿的差异保留。 本文件是研究笔记，不是作者原文或复现结果；涉及“我们的设置”的旧描述以本仓库 protocol 为准。机器路径已替换。

# SOURCE — Mini-o3

Read from: arXiv:2509.07969v1 (https://arxiv.org/pdf/2509.07969v1), sha256 d6bc21d118a7772fb735643bb72fb41d0182c9fb51d7beec8ec35bf537dcd036, 18 pages. Checked 2026-10-28: the abs page lists only [v1] (submitted 9 Sep 2025); v2 and v3 pdf URLs return 404.
Paper figures shown: none
Style: palette rust · layout hero-number

## Second text used (descriptive content only, never for numbers)
Local ICLR 2026 proceedings text `[local-source]`, sha256 aff5e56e2a617da9778528daadb3e60e19ae45f911b65291386c37612f70193e. Used ONLY for content absent from arXiv v1: the cold-start prompt template and in-context-example text (App. F), the general-VQA / Gemini note (App. D), the judge prompt (App. E). Every such statement is tagged **[ICLR]** in facts.md and 精读.zh.md. The poster uses none of it as a number.
- Numbers in Table 1, Table 2, Table 3, Eq 1-4 text, 6,000 trajectories / 6 exemplars, SFT and RL hyper-parameters: identical in both texts where compared (checked Tables 1-3 line by line).
- Differences noted, none used for a number: the ICLR text has App. A-F (prompt, examples, VQA, judge) whereas v1's appendix holds only three trajectory illustrations; ICLR App. D Table 4 lists VisualProbe hard/medium/easy for "Ours" as 48.2 / 50.5 / 67.3 while arXiv v1 Table 1 and Table 3 (2M) list 48.0 / 50.4 / 67.0; the ICLR text says 6 exemplars but displays 4 in-context examples (Fig 9-12).

## ‡ numbers (computed or read off a figure)
- Poster/精读 "约 25‡, 约 38‡" (Mini-o3 at test limit 4 turns, Mini-o3 w/o over-turn mask): read off Figure 1 left (p2); the y-axis has gridlines every 2 points, values are my reading, not printed. Mini-o3 48.0 and DeepEyes 35.1 are printed in Figure 1 right.
- The paper's own ‡ in Table 1 (DeepEyes, Pixel Reasoner re-evaluated with official code, Avg@32) is written as "（复测）" in 精读 and the poster, to keep ‡ reserved for my computed / read-off numbers.
- facts.md only: Fig 1 right segments normalised by 48.0 (52.7 / 31.9 / 13.3 / 1.7 / 0.2) = segment / 48.0 * 100, compared against Fig 6 values (53.3 / 32.4 / 12.2 / 1.9 / 0.1). Not used on the poster.
- facts.md: MME-RW margin 65.5 - 64.4 = 1.1, V* margin 88.2 - 88.0 = 0.2. Not printed in the paper; the poster/精读 print the two values instead of the difference.
- "8.6 points" (Hard, Table 2 ID4 vs ID1) is stated in the paper text (p10) and equals 44.4 - 35.8.

## Page / table provenance (printed page numbers of the arXiv PDF)
- Abstract, authors, affiliations: p1. Intro, 3-component recipe, 16 -> 6 turns / 10 -> 3 days: p2-4. Fig 1: p2. Fig 3 (framework): p4.
- §3.1 action space, two-phase training, judge reward, caps 6 turns / 32K: p5. Fig 4 (VisualProbe), Fig 5 (cold-start pipeline), §3.2 data: p6-7 ("retain only trajectories whose final answers are correct", "approximately 6,000 ... from 6 exemplars": p7).
- §3.3 max pixels, Eq 1-2, over-turn problem statement, Fig 6: p7. Eq 3-4, Fig 7, §3.4: p8.
- Table 1, §4.1 (SFT/RL/data/metric), §4.2: p9. Table 2, Table 3, ablation text: p10. Fig 8, conclusion: p10-11. Appendix A (Fig 9-11): p16-18.

## Not verified
- Identity of the generator VLM, exemplar contents (v1), the judge model, whether the generator sees ground truth, filter judging method, source of the 6,000 questions: not stated in v1 main text or appendix (searched). ICLR text does not name the generator for the VisualProbe trajectories either.
- Our A-SFT's filtering/teacher details were not part of the brief; the poster states only the brief's facts about our setting.
- Reading "over-turn trajectories are removed by the correct-answer filter in SFT generation" is mine; the paper only states the final-answer-correct rule and the turn-limit termination.
- Did not run any code from the paper. Fig 7 / Fig 5 redrawn by me, no paper figure is shown.
