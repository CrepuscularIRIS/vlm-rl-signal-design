> 阅读状态：已有详细阅读稿；原核查中断，不能标成全文独立复核通过。 本文件是研究笔记，不是作者原文或复现结果；涉及“我们的设置”的旧描述以本仓库 protocol 为准。机器路径已替换。

# SOURCE — CounterCredit

Read from: arXiv 2609.22910v1 (https://arxiv.org/pdf/2609.22910v1), downloaded PDF, 30 pages, sha256 f7bf7bb5a7a25b37fd4ef8432493c052d1f8c87c71d636e972c74cc44e8894a4 (paper.pdf in work dir).
Title in PDF: "When Should a VLM Look? Paying Only for Visual Calls That Were Needed and Used"; method name CounterCredit. Authors (title block): Kunyu Peng, Junming Liu, Ruiqi He, Qingzhuo Wang, Jianzhong Qi, Xianhui Liu.
Paper figures shown: none
Style: palette cobalt · layout blueprint

## ‡ notes
No computed numbers are used: every decimal in 精读.zh.md, facts.md and the poster appears in the paper. No figure reads (‡ none). The gaps 6.8 / 11.0 / 0.0 / −1.1 / 1.6 / 1.7 / 15.8 / 3.1 / 9.4 / 0.5 / 2.4 / 2.7 / 1.1 / 4.2 / 7.2 are quoted from the paper text (pp.8-9).

## Provenance (page / table)
- Method: §3 pp.3-6 (Eq.1-7, Prop.1-4), Fig.2 p.4, Alg.1 p.21, constants Table 5 p.17 and App. C pp.18-20, prompts/elicitation App. E pp.21-22.
- Data: §4.1 p.6, App. B p.18, App. C p.19.
- Results: Table 1 p.7; Table 2 p.8; Table 3 p.8; Table 4 p.9; Table 7-8 p.23; Table 9-10 p.24; Table 11-12 p.25; Table 13 p.26; Table 14 p.25 (appendix numbering as printed).
- Author limits: §5 p.10, App. A.8 pp.17-18, App. G p.30.
- Poster chart: Table 9 (V*) values; poster table: Table 3.

## Caveats
- Text/table: abstract "10% to 12%" vs §5 "9.8% to 11.8%" (Table 7 base rates 11.8 / 9.8 / 10.4). Table 4 caption says Avg is over seven benchmarks while four columns are shown; Avg values match Tables 1/11.
- "Per-token / per-call split of Â_j not stated" = searched §3.5, App. A.6, App. C, Alg.1 only by reading; no code was run or consulted.
- Our-setting statements in "与我们的关系" come from BRIEF_ICML.md.
- Poster renders were produced with a local long-timeout copy of poster_styles.render (host load made the library default 30 s timeout fail); lib not edited.
