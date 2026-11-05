"""SG reward (Design_New 2026-11-04 §4): scale-gated evidence acquisition score S in [0, 1].

R = V * (Y + LAMBDA * m * S)   (assembled in caption_contracts.assistant_reward_batch)

Per evidence node b (L1 box, normalized) and presented view v (original image v0 + executed crops):
  c(b, v)  = |b ∩ v| / |b|
  p_b(v)   = detail_b * s(v)        detail_b: text line height / object short side, original px;
                                    s(v): original px -> final model-input px (present() then max_pixels cap)
  a_b(v)   = 1[c >= 0.9] * h_kind(p_b(v))      context nodes: h = 1 (coverage only)
  z_b      = max_v a_b(v)           (v0 included)
  Phi      = max_{S in E_q} mean_{b in S} z_b
Gate: g = 1 once any key node has a_b(v) >= U_GATE in some view.
  E'      = union area of crops with s(v) >= 2 * s(v0), over image area  (effective new exploration)
  Psi     = ALPHA * E'                    if g == 0
          = ALPHA + (1 - ALPHA) * Phi     if g == 1
  S       = Psi_T - Psi_0                (Psi_0: v0 only)
h is an UNCALIBRATED scale proxy (linear ramp p0 -> p1, saturates at 1); constants frozen in SG_FREEZE.
"""
import json
import math
from pathlib import Path

ALPHA = 0.25
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


def _psi(rec, W, H, views):
    nodes = rec["nodes"]
    z, gate = {}, False
    for nid, n in nodes.items():
        b = (n["bbox"][0] * W, n["bbox"][1] * H, n["bbox"][2] * W, n["bbox"][3] * H)
        detail = (b[3] - b[1]) / max(1, n.get("lines", 1)) if n["kind"] == "text" else min(b[2] - b[0], b[3] - b[1])
        best = 0.0
        for box, s in views:
            if _cover(b, box) >= COVER:
                best = max(best, _h(n["kind"], detail * s))
        z[nid] = best
        if n.get("key") and best >= U_GATE:
            gate = True
    phi = max(sum(z[i] for i in S) / len(S) for S in rec["sets"])
    if gate:
        return ALPHA + (1 - ALPHA) * phi, phi, True, 0.0
    s0 = views[0][1]
    eff = [box for box, s in views[1:] if s >= 2 * s0]
    explore = min(1.0, _union_area(eff) / (W * H)) if eff else 0.0
    return ALPHA * explore, phi, False, explore


def process_score(qid, wh, executed_boxes):
    """-> (m, S, diag). m = 0 when the question is not eligible (S = 0)."""
    rec = index().get(qid)
    W, H = wh
    if rec is None or not rec.get("m") or not W or not H:
        return 0, 0.0, {}
    views = _views(W, H, executed_boxes)
    psi_t, phi_t, g_t, e_t = _psi(rec, W, H, views)
    psi_0, phi_0, g_0, _ = _psi(rec, W, H, views[:1])
    S = max(0.0, min(1.0, psi_t - psi_0))
    return 1, S, {"phi": round(phi_t, 4), "phi0": round(phi_0, 4), "gate": int(g_t), "gate0": int(g_0),
                  "explore": round(e_t, 4)}


def _selfcheck():
    """Counterexamples the score must satisfy (run: python reward_sg.py)."""
    global _INDEX
    _INDEX = {
        # one small text cell in a 4000x3000 image, plus its header (key = cell)
        "t": {"m": 1, "sets": [["c", "h"]], "nodes": {
            "c": {"bbox": [0.50, 0.50, 0.52, 0.505], "kind": "text", "lines": 1, "key": True},
            "h": {"bbox": [0.10, 0.50, 0.20, 0.505], "kind": "text", "lines": 1, "key": False}}},
        "off": {"m": 0, "sets": [["c"]], "nodes": {}},
    }
    W, H = 4000, 3000
    m, s, d = process_score("t", (W, H), [])
    assert (m, s) == (1, 0.0), (m, s, d)                         # no crop: no gain
    full = [(0, 0, W, H)]
    assert process_score("t", (W, H), full)[1] == 0.0            # full-image "crop": no magnification, no gain
    tight = [(1990, 1490, 2090, 1525)]                             # magnified crop on the cell
    m, s_tight, d = process_score("t", (W, H), tight)
    assert d["gate"] == 1 and s_tight > ALPHA, (s_tight, d)        # gate passed, partial set (header missing)
    both = tight + [(390, 1490, 810, 1525)]
    s_both = process_score("t", (W, H), both)[1]
    assert s_both > s_tight, (s_both, s_tight)                     # completing the evidence set adds progress
    one_big = [(380, 1480, 2100, 1530)]                             # one crop covering both nodes (exploration-neutral)
    s_big = process_score("t", (W, H), one_big)[1]
    assert s_big > 0 and abs(s_big - s_both) < 0.35, (s_big, s_both)
    wander = [(0, 0, 1000, 750), (3000, 2250, 4000, 3000)]          # magnified exploration away from the target
    m, s_w, d = process_score("t", (W, H), wander)
    assert d["gate"] == 0 and 0 < s_w <= ALPHA, (s_w, d)            # exploration credit capped at ALPHA
    assert s_w < s_tight
    assert process_score("off", (W, H), tight) == (0, 0.0, {})
    assert process_score("missing", (W, H), tight)[0] == 0
    # small image already legible at v0: gate open at start, S = 0 for any crop (no ritual cropping credit)
    _INDEX = {"s": {"m": 1, "sets": [["c"]], "nodes": {
        "c": {"bbox": [0.2, 0.2, 0.8, 0.4], "kind": "text", "lines": 1, "key": True}}}}
    m, s, d = process_score("s", (600, 400), [(100, 60, 500, 180)])
    assert d["gate0"] == 1 and s == 0.0, (s, d)
    _INDEX = None
    print("reward_sg selfcheck: 11/11 ok")


if __name__ == "__main__":
    _selfcheck()
