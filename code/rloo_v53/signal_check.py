"""First-step signal check for grpo_v53_dl (user 10-27): group rewards, fork value gaps, violation sources, aux
gradient and extra cost, read from one run's logs. Counts only; no model calls.
  python signal_check.py OUT_DIR STDOUT_LOG [STEP]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import pstdev


def groups(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["uid"]].append(r)
    kinds = Counter()
    stds, advantages = [], []
    for g in by.values():
        c = sum(r["correct"] for r in g)
        kinds["all_correct" if c == len(g) else "all_wrong" if c == 0 else "mixed"] += 1
        rewards = [r["reward"] for r in g]
        stds.append(pstdev(rewards))
        if len(g) > 1:
            advantages += [r - (sum(rewards) - r) / (len(g) - 1) for r in rewards]
    n = len(rows)
    if not n:
        raise ValueError("No rollout rows for requested step")
    return {"questions": len(by), "trajectories": n, "groups": dict(kinds),
            "group_sizes": dict(Counter(len(g) for g in by.values())),
            "reward_std_mean": sum(stds) / len(stds), "nonzero_reward_variance_groups": sum(s > 0 for s in stds),
            "rloo_abs_adv_mean": sum(abs(a) for a in advantages) / max(1, len(advantages)),
            "acc": sum(r["correct"] for r in rows) / n, "reward_mean": sum(r["reward"] for r in rows) / n,
            "crops_mean": sum(r["crops"] for r in rows) / n, "terminal": dict(Counter(r["terminal"] for r in rows)),
            "first_kind": dict(Counter(str(r.get("first_kind")) for r in rows)),
            "violation_sources": dict(Counter(e for r in rows for e in r.get("errors", []))),
            "first_crops_mean": sum(r["crops"] > 0 for r in rows) / n,
            "later_crops_mean": sum(max(0, r["crops"] - 1) for r in rows) / n,
            "gen_tokens": sum(r["gen_tokens"] for r in rows), "proc_tokens": sum(r["proc_tokens"] for r in rows)}


def forks(lines: list[dict]) -> dict:
    states = [x for x in lines if "state_status" in x]
    fk = [x for x in lines if "action" in x]
    gap = {}
    for x in fk:
        gap[(x["pid"], x["call"], x["owner_row"])] = x["d"]
    d = list(gap.values())
    by_action = {a: [x for x in fk if x["action"] == a] for a in ("answer", "grounding")}
    return {"states": len(states), "state_status": dict(Counter(x["state_status"] for x in states)),
            "forks": len(fk), "d_sign": {"pos": sum(v > 0 for v in d), "neg": sum(v < 0 for v in d),
                                         "zero": sum(v == 0 for v in d)},
            "abs_d_mean": sum(abs(v) for v in d) / max(1, len(d)),
            "q": {a: sum(x["r"] for x in v) / max(1, len(v)) for a, v in by_action.items()},
            "y": {a: sum(x["y"] for x in v) / max(1, len(v)) for a, v in by_action.items()},
            "terminal": {a: dict(Counter(x["terminal"] for x in v)) for a, v in by_action.items()},
            "violation_sources": {a: dict(Counter(e for x in v for e in x.get("errors", []))) for a, v in by_action.items()},
            "fork_crops_mean": {a: sum(x["n_crops"] for x in v) / max(1, len(v)) for a, v in by_action.items()},
            "fork_gen_tokens": sum(x["gen_tokens"] for x in fk),
            "fork_proc_tokens": sum(x["proc_tokens"] for x in fk) if all("proc_tokens" in x for x in fk) else None}


def main() -> None:
    out, log = Path(sys.argv[1]), Path(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rows = [json.loads(x) for x in (out / "rollouts.jsonl").read_text().splitlines() if x.strip()]
    res = {"rollouts": groups([r for r in rows if r["step"] == step])}
    fork_lines = []
    for f in sorted(out.glob("forks.jsonl.*")):
        fork_lines += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    calls = [x for x in fork_lines if x.get("call") == step - 1]
    if calls:
        res["forks"] = forks(calls)
    text = log.read_text(errors="replace")
    res["log"] = {"dl_forks": re.findall(rf"\[dl_forks\] call={step - 1} .*", text),
                  "dl_aux": re.findall(rf"\[dl_aux\] update={step} .*", text),
                  "timing": re.findall(r"timing_s/\w+:\s*[\d.]+", text)[:40]}
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
