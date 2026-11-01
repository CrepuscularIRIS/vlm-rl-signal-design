"""Reaggregate existing public E1 exports. CPU only; no inference or scoring changes."""
from pathlib import Path
import csv, gzip, hashlib, json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/baseline_classes_20261031'
OUT.mkdir(exist_ok=True)
CONDITIONS = [
    ('B0-ans', 'I', 'Base / direct answer', 'none', 'no RL'),
    ('B0', 'I', 'Base / CoT', 'none', 'no RL'),
    ('B1', 'I', 'Base / native zoom prompt', 'native zoom', 'no RL'),
    ('B3-ans', 'II', 'A-SFT / direct answer', 'none', 'no RL'),
    ('B3-cot', 'II', 'A-SFT / CoT', 'none', 'no RL'),
    ('B3-zoom', 'II', 'A-SFT / native zoom prompt', 'native zoom', 'no RL'),
    ('K-GRPO', 'III', 'A-SFT + standard GRPO', 'caption v5.1', '4k x 8 / step32'),
    ('K-base', 'III', 'Base + standard GRPO', 'caption v5.1', '4k x 8 / step32'),
    ('B3', 'supplement', 'A-SFT / caption prompt', 'caption v5.1', 'no RL'),
    ('B2', 'supplement', 'Base / caption prompt', 'caption v5.1', 'no RL'),
    ('B4', 'historical', 'A-SFT + standard GRPO', 'caption v5.1', '6k x 16 / step47'),
    ('B5', 'historical', 'Base + standard GRPO', 'caption v5.1', '6k x 16 / step47'),
    ('B6', 'historical', 'Base + standard GRPO / no tools', 'none', '6k x 16 / step47'),
]
rows, strata, hashes = [], [], {}
reference = None
for cid, group, label, interface, budget in CONDITIONS:
    path = ROOT / 'data/rollouts' / (cid + '.jsonl.gz')
    data = [json.loads(line) for line in gzip.open(path, 'rt')]
    keys = {(r['item_id'], r['sample_idx']): (r['gold_letter'], r['bench']) for r in data}
    assert len(keys) == len(data) == 2208
    assert set(Counter(r['item_id'] for r in data).values()) == {4}
    assert len({r['item_id'] for r in data}) == 552
    if reference is None:
        reference = keys
    assert keys == reference
    assert all(r['correct'] in (0, 1) for r in data)
    n = len(data)
    row = dict(id=cid, baseline_class=group, label=label, interface=interface,
               rl_budget=budget, n=n, correct=sum(r['correct'] for r in data))
    row.update(accuracy=row['correct']/n,
               answer_tag_rate=sum(r['has_answer_tag'] for r in data)/n,
               crops=sum(r['crops'] for r in data)/n,
               gen_tokens=sum(r['gen_tokens'] for r in data)/n,
               processed_tokens=sum(r['prompt_tokens']+r['gen_tokens'] for r in data)/n,
               truncated=sum(bool(r['truncated']) for r in data)/n)
    rows.append(row)
    for source in sorted({r['bench'] for r in data}):
        part = [r for r in data if r['bench'] == source]
        c = sum(r['correct'] for r in part)
        strata.append(dict(id=cid, source=source, n=len(part), correct=c, accuracy=c/len(part)))
    hashes[cid] = hashlib.sha256(path.read_bytes()).hexdigest()
for name, records in [('summary', rows), ('by_source', strata)]:
    with (OUT / (name+'.csv')).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
(OUT/'PROVENANCE.json').write_text(json.dumps(dict(
    input_sha256=hashes, rows_per_condition=2208, questions=552, samples_per_question=4,
    same_keys_gold_source=True, operation='reaggregation of frozen correct fields; no inference',
    scorer_reference='scripts/scorer_reference.py',
    scorer_sha256=hashlib.sha256((ROOT/'scripts/scorer_reference.py').read_bytes()).hexdigest(),
    notes=['All failed rows retain their denominator.', 'E1 is not an independent blind test.',
           'Prompt/tool interfaces and historical RL budgets are explicitly separated.']), indent=2)+'\n')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios':[1.1, 1]})
colors = ['#3575ad', '#e69838']
for j, label in enumerate(['Base', 'A-SFT']):
    vals = [r['accuracy']*100 for r in rows[j*3:j*3+3]]
    bars = axes[0].bar([i+(j-.5)*.34 for i in range(3)], vals, width=.32, label=label, color=colors[j])
    axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
axes[0].set_xticks(range(3), ['Direct answer', 'CoT', 'Native zoom\nprompt'])
axes[0].set_title('Classes I and II: fixed checkpoints, different prompts')
axes[0].legend(frameon=False)
vals = [r['accuracy']*100 for r in rows[6:8]]
bars = axes[1].bar(['A-SFT + GRPO', 'Base + GRPO'], vals, color=[colors[1], colors[0]], width=.5)
axes[1].bar_label(bars, fmt='%.2f', padding=3)
axes[1].set_title('Class III: outcome-only GRPO\n4k x 8, final step32, caption v5.1')
for ax in axes:
    ax.set_ylim(0, 75); ax.set_ylabel('E1 accuracy (%)'); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Existing E1 baselines: 552 questions x 4 samples per condition', fontsize=13)
fig.text(.5,.015,'All failures retained. E1 is diagnostic, not blind. Native zoom uses model-generated tool text.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.94))
for ext in ['png','pdf']:
    fig.savefig(OUT/f'baseline_classes.{ext}', dpi=170)
print(json.dumps([{k:r[k] for k in ['id','n','correct','accuracy']} for r in rows],indent=2))
