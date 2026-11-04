"""Per-step training stats from rollouts.jsonl (1024 rows/step, no filtering). usage: python step_stats.py <rollouts.jsonl>"""
import json
import statistics as st
import sys

R = [json.loads(l) for l in open(sys.argv[1])]
n = len(R) // 1024
print(f"steps {n}/32")
print("step  acc(orig)  S(m=1)  gate  crops  gen_tok  cont_rows  cont_acc")
for s in range(1, n + 1):
    B = R[(s - 1) * 1024:s * 1024]
    o = [r for r in B if not r.get("cont")]
    m = [r for r in o if r.get("m")]
    c = [r for r in B if r.get("cont")]
    ca = f"{st.mean(r['correct'] for r in c):.3f}" if c else "-"
    print(f"{s:4d}  {st.mean(r['correct'] for r in o):.3f}     {st.mean(r['s'] for r in m):.3f}   "
          f"{st.mean(r.get('gate', 0) for r in m):.2f}  {st.mean(r['crops'] for r in o):.2f}   "
          f"{st.mean(r['gen_tokens'] for r in o):5.0f}    {len(c):4d}      {ca}")
