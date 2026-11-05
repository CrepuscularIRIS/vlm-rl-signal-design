"""SG reward v2 (mentor spec 2026-11-05): S in [0, 1], R = V * (Y + LAMBDA * m * S).

Per evidence node b (L1 box, normalized -> px) and the EXECUTED crops C (history; the original view is not a crop):
  prox_b   = 0.5 * ( max_{c in C} IoU(b, c) + |b ∩ (∪C)| / |b| )         in [0, 1]
  Prox     = max_{S in E_q} mean_{b in S} prox_b                          (aggregation over L2 evidence_sets kept from v1)
Gate (unchanged from v1): g = 1 once any key node is >= 90% covered by some presented view (original included) whose
  final-input scale gives h_kind >= U_GATE (uncalibrated px ramp: text 6->14, object 16->48).
  explore  = |∪C| / (W * H)                                               (all historical crops, no magnification filter)
  Psi      = ALPHA + (1 - ALPHA) * Prox   if g == 1
           = ALPHA * explore              if g == 0
  S        = clip(Psi_T - Psi_0, 0, 1)    (Psi_0: original view only, i.e. no crops)
CAPTION_SG_PSI=max (user 2026-11-05): Psi = max(Prox, ALPHA * explore) -- upper envelope of an approach score and an
  explore score; the argmax switches explore -> approach automatically once crops reach the targets. Approach has no
  ALPHA floor (with a floor, ALPHA*explore could never win). Both terms are non-decreasing in the crop history.
v1 differences: v1 Phi used scale-gated coverage h(detail * s(v)) per view and explore counted only crops with
s(v) >= 2 s(v0). Diagnostics keys: prox, prox0, gate, gate0, explore.
"""
import json
import math
import os
from pathlib import Path

ALPHA = 0.25
PSI_MODE = os.environ.get("CAPTION_SG_PSI", "gate")  # "gate": mentor pseudocode; "max": psi = max(prox, ALPHA*explore)
U_GATE = 0.5
COVER = 0.9
RAMP = {"text": (6.0, 14.0), "object": (16.0, 48.0)}  # final-input px; uncalibrated proxy
AREA_CAP = 1400 * 1400      # caption_env.present
MIN_SHOWN_SHORT = 448
MAX_UPSCALE = 4.0
MAX_PIXELS = 589824         # data.max_pixels (process_image cap)

_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = json.loads(Path(__file__).with_name("sg_index.json").read_text())
    return _INDEX


def view_scale(w: float, h: float) -> float:
    """Original px -> final input px for a view of w x h original pixels (caption_env.present + process_image)."""
    pw, ph = w, h
    if min(pw, ph) < MIN_SHOWN_SHORT:
        k = min(MIN_SHOWN_SHORT / min(pw, ph), MAX_UPSCALE)
        pw, ph = pw * k, ph * k
    if pw * ph > AREA_CAP:
        k = math.sqrt(AREA_CAP / (pw * ph))
        pw, ph = pw * k, ph * k
    if pw * ph > MAX_PIXELS:
        k = math.sqrt(MAX_PIXELS / (pw * ph))
        pw, ph = pw * k, ph * k
    return pw / w


def _cover(b, v):
    iw = min(b[2], v[2]) - max(b[0], v[0])
    ih = min(b[3], v[3]) - max(b[1], v[1])
    a = (b[2] - b[0]) * (b[3] - b[1])
    return iw * ih / a if iw > 0 and ih > 0 and a > 0 else 0.0


def _h(kind: str, p: float) -> float:
    if kind == "context":
        return 1.0
    p0, p1 = RAMP[kind]
    return min(1.0, max(0.0, (p - p0) / (p1 - p0)))


def _union_area(boxes):
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        iv = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        high = length = 0.0
        for lo, hi in iv:
            length += max(0.0, hi - max(lo, high))
            high = max(high, hi)
        total += (right - left) * length
    return total


def _views(W, H, crops):
    """[(box_px, scale)], v0 first."""
    return [((0.0, 0.0, float(W), float(H)), view_scale(W, H))] + \
           [(tuple(float(x) for x in c), view_scale(c[2] - c[0], c[3] - c[1])) for c in crops]


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _union_cover(b, crops):
    """|b ∩ (∪ crops)| / |b| (exact, via union of the crops clipped to b)."""
    clipped = [(max(b[0], c[0]), max(b[1], c[1]), min(b[2], c[2]), min(b[3], c[3])) for c in crops]
    clipped = [c for c in clipped if c[2] > c[0] and c[3] > c[1]]
    a = (b[2] - b[0]) * (b[3] - b[1])
    return min(1.0, _union_area(clipped) / a) if clipped and a > 0 else 0.0


def _psi(rec, W, H, views):
    """views = [(box_px, scale)], original first; crops = views[1:]. -> (psi, prox, gate, explore)."""
    nodes = rec["nodes"]
    crops = [box for box, _ in views[1:]]
    prox, gate = {}, False
    for nid, n in nodes.items():
        b = (n["bbox"][0] * W, n["bbox"][1] * H, n["bbox"][2] * W, n["bbox"][3] * H)
        detail = (b[3] - b[1]) / max(1, n.get("lines", 1)) if n["kind"] == "text" else min(b[2] - b[0], b[3] - b[1])
        if n.get("key") and not gate:
            gate = any(_cover(b, box) >= COVER and _h(n["kind"], detail * s) >= U_GATE for box, s in views)
        prox[nid] = 0.5 * (max((_iou(b, c) for c in crops), default=0.0) + _union_cover(b, crops)) if crops else 0.0
    proximity = max(sum(prox[i] for i in S) / len(S) for S in rec["sets"])
    explore = min(1.0, _union_area(crops) / (W * H)) if crops else 0.0
    if PSI_MODE == "max":   # automatic explore -> approach switch: upper envelope, no gate, no ALPHA floor on approach
        psi = max(proximity, ALPHA * explore)
    else:
        psi = ALPHA + (1 - ALPHA) * proximity if gate else ALPHA * explore
    return psi, proximity, gate, explore


def process_score(qid, wh, executed_boxes):
    """-> (m, S, diag). m = 0 when the question is not eligible (S = 0)."""
    rec = index().get(qid)
    W, H = wh
    if rec is None or not rec.get("m") or not W or not H:
        return 0, 0.0, {}
    views = _views(W, H, executed_boxes)
    psi_t, prox_t, g_t, e_t = _psi(rec, W, H, views)
    psi_0, prox_0, g_0, _ = _psi(rec, W, H, views[:1])
    S = max(0.0, min(1.0, psi_t - psi_0))
    state = "approach" if prox_t > 0 and prox_t >= ALPHA * e_t else ("explore" if e_t > 0 else "none")
    return 1, S, {"prox": round(prox_t, 4), "prox0": round(prox_0, 4), "gate": int(g_t), "gate0": int(g_0),
                  "explore": round(e_t, 4), "psi": round(psi_t, 4), "state": state}


def _selfcheck():
    """v2 expectations (run: python reward_sg.py). Behaviour changes vs v1 are printed, not hidden."""
    global _INDEX
    _INDEX = {
        "t": {"m": 1, "sets": [["c", "h"]], "nodes": {
            "c": {"bbox": [0.50, 0.50, 0.52, 0.505], "kind": "text", "lines": 1, "key": True},
            "h": {"bbox": [0.10, 0.50, 0.20, 0.505], "kind": "text", "lines": 1, "key": False}}},
        "off": {"m": 0, "sets": [["c"]], "nodes": {}},
    }
    W, H = 4000, 3000
    ok = 0
    m, s, d = process_score("t", (W, H), [])
    assert (m, s) == (1, 0.0) and d["prox"] == 0 and d["explore"] == 0, d; ok += 1     # no crop: nothing
    tight = [(1990, 1490, 2090, 1525)]                                                   # tight crop on the cell
    m, s_tight, d = process_score("t", (W, H), tight)
    assert d["gate"] == 1 and s_tight > ALPHA, (s_tight, d); ok += 1
    assert 0 <= d["prox"] <= 1, d; ok += 1
    both = tight + [(390, 1490, 810, 1525)]
    s_both = process_score("t", (W, H), both)[1]
    assert s_both > s_tight, (s_both, s_tight); ok += 1                                  # completing the set adds
    looser = [(1900, 1400, 2200, 1600)]                                                  # covers cell, lower IoU
    d_l = process_score("t", (W, H), looser)[2]
    d_t = process_score("t", (W, H), tight)[2]
    assert d_l["prox"] < d_t["prox"], (d_l, d_t); ok += 1                                # IoU term rewards tightness
    wander = [(0, 0, 1000, 750), (3000, 2250, 4000, 3000)]                               # away from the target
    m, s_w, d = process_score("t", (W, H), wander)
    assert d["gate"] == 0 and d["prox"] == 0 and abs(s_w - ALPHA * 0.125) < 1e-9, (s_w, d); ok += 1
    assert s_w < s_tight; ok += 1
    overlap = [(0, 0, 1000, 750), (500, 375, 1500, 1125)]                                # union area, not sum
    assert abs(process_score("t", (W, H), overlap)[2]["explore"] - (1000 * 750 * 2 - 500 * 375) / (W * H)) < 1e-4; ok += 1
    assert process_score("off", (W, H), tight) == (0, 0.0, {}); ok += 1
    assert process_score("missing", (W, H), tight)[0] == 0; ok += 1
    global PSI_MODE
    keep, PSI_MODE = PSI_MODE, "max"
    s_wm, d_wm = process_score("t", (W, H), wander)[1:]
    assert d_wm["state"] == "explore" and abs(s_wm - ALPHA * 0.125) < 1e-9, (s_wm, d_wm); ok += 1     # far crops: explore
    s_tm, d_tm = process_score("t", (W, H), wander + tight)[1:]
    assert d_tm["state"] == "approach" and s_tm > s_wm, (s_tm, d_tm); ok += 1                       # reaching target switches
    full_max = process_score("t", (W, H), [(0, 0, W, H)])
    PSI_MODE = keep
    # behaviour changes vs v1, reported for the record
    full = process_score("t", (W, H), [(0, 0, W, H)])
    _INDEX = {"s": {"m": 1, "sets": [["c"]], "nodes": {
        "c": {"bbox": [0.2, 0.2, 0.8, 0.4], "kind": "text", "lines": 1, "key": True}}}}
    leg = process_score("s", (600, 400), [(100, 60, 500, 180)])
    _INDEX = None
    print(f"reward_sg v2 selfcheck: {ok}/{ok} ok")
    print(f"  [v1->v2] full-image 'crop', gate closed: S={full[1]:.4f} explore={full[2]['explore']} (v1: 0, filtered)")
    print(f"  [max mode] full-image 'crop': S={full_max[1]:.4f} prox={full_max[2]['prox']} state={full_max[2]['state']}"
          " (union-coverage term = 1 for any crop containing the box)")
    print(f"  [v1->v2] original already legible (gate0=1), crop on target: S={leg[1]:.4f} prox={leg[2]['prox']} (v1: 0)")


if __name__ == "__main__":
    _selfcheck()
