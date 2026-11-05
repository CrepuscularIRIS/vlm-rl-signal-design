"""Outcome-stratified evidence term on top of GRPO (2026-11-05, CAPTION_ADV_STRAT=1).

    A_i = (Y_i - mean_g Y) / (std_g Y + eps)  +  lam * (Q_i - mean_{j in H_i} Q_j)

The first term is stock EasyR1 GRPO on R = V*Y (run with CAPTION_SG_LAMBDA=0). Q_i = S from reward_sg (psi v2,
L1 index, CAPTION_SG_PSI). H_i = same group, same Y, evidence scorable (m = 1 and a valid final answer, not a
continuation row). The second term is 0 when i is not scorable or |H_i| < 2. It is centred only (not divided by a
std) and broadcast to every response token of the trajectory, like the GRPO term.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict


def enabled() -> bool:
    return os.environ.get("CAPTION_ADV_STRAT", "0") == "1"


def strat_term(y, q, scorable, uids, lam):
    """-> per-trajectory lam * (q_i - mean over same-uid, same-y, scorable peers); 0 if unscorable or alone."""
    strata = defaultdict(list)
    for i, (yi, ok, u) in enumerate(zip(y, scorable, uids)):
        if ok:
            strata[(u, int(yi))].append(i)
    out = [0.0] * len(y)
    for rows in strata.values():
        if len(rows) < 2:
            continue
        m = sum(q[i] for i in rows) / len(rows)
        for i in rows:
            out[i] = lam * (q[i] - m)
    return out


def install_advantage():
    """Wrap ray_trainer.compute_advantage (install BEFORE update_ledger so the ledger sees the final advantages)."""
    import torch
    import verl.trainer.ray_trainer as RT
    from reward_sg import process_score
    if getattr(RT.compute_advantage, "_strat", False):
        return
    original = RT.compute_advantage
    lam = float(os.environ.get("CAPTION_STRAT_LAMBDA", "0.1"))
    log_path = os.environ.get("CAPTION_STRAT_LOG")
    calls = {"n": 0}

    def compute_advantage(data, *args, **kwargs):
        data = original(data, *args, **kwargs)
        mm = data.non_tensor_batch["multi_modal_data"]
        uids = [str(u) for u in data.non_tensor_batch["uid"]]
        y = [int(s >= 1.0 - 1e-6) for s in data.batch["token_level_rewards"].sum(-1).tolist()]
        q, ok = [], []
        for m_ in mm:
            valid = m_.get("caption_terminal") == "answer" and not m_.get("caption_cont_views")
            m, s, _ = process_score(m_.get("caption_qid", ""), m_.get("caption_geometry_wh", (0, 0)),
                                    (m_.get("caption_recipe") or {}).get("boxes") or []) if valid else (0, 0.0, {})
            q.append(float(s))
            ok.append(bool(valid and m == 1))
        p = strat_term(y, q, ok, uids, lam)
        mask = data.batch["response_mask"]
        add = torch.tensor(p, dtype=data.batch["advantages"].dtype, device=data.batch["advantages"].device)
        data.batch["advantages"] = data.batch["advantages"] + add.unsqueeze(-1) * mask
        data.batch["returns"] = data.batch["advantages"].clone()
        calls["n"] += 1
        if log_path:
            g = defaultdict(list)
            for i, u in enumerate(uids):
                g[u].append(i)
            homo = [i for rows in g.values() if len({y[j] for j in rows}) == 1 for i in rows]
            tok = mask.sum(-1).tolist()
            adv = data.batch["advantages"]
            tot = float((adv.abs() * mask).sum())
            hom = float(sum((adv[i].abs() * mask[i]).sum() for i in homo))
            rec = {"call": calls["n"], "lam": lam, "n": len(y), "scorable": sum(ok),
                   "nonzero_term": sum(1 for x in p if x != 0.0), "mean_abs_term": sum(abs(x) for x in p) / len(p),
                   "homog_traj": len(homo), "homog_nonzero_term": sum(1 for i in homo if p[i] != 0.0),
                   "share_abs_adv_mass_homog": hom / tot if tot else None, "mean_tokens": sum(tok) / len(tok)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[strat] call={rec['call']} scorable={rec['scorable']}/{rec['n']} nonzero={rec['nonzero_term']} "
                  f"|term|={rec['mean_abs_term']:.4f} homog_mass={rec['share_abs_adv_mass_homog']:.3f}", flush=True)
        return data

    compute_advantage._strat = True
    RT.compute_advantage = compute_advantage
    print(f"[caption] advantage -> GRPO + outcome-stratified evidence term (lambda={lam})", flush=True)
