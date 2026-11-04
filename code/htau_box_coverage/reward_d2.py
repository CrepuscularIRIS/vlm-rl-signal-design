"""D2 reward: R = V·(Y + 0.25·m·Φ), Φ = 0.5·Φ_cover + 0.5·Φ_cite (frozen 2026-11-02, results/d2_offline_20261102).

Φ_cover  — per D1-B: mean over evidence nodes of (maxIoU(node, executed crops) + union-coverage)/2.
Φ_cite   — mean over nodes of max coverage by the views cited in the final answer turn
            (observation_k only; the original view is excluded). Citation = usage is a frozen
            assumption; cite-all headroom is a registered, deferred hacking observation.
Both use original-image pixel coordinates: executed boxes from caption_recipe, targets from index.json
(normalized, converted with caption_geometry_wh).
"""
import json
import re
from pathlib import Path
from statistics import mean

_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = json.loads((Path(__file__).with_name("index.json")).read_text())
    return _INDEX


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[0] < box[2] and box[1] < box[3] else None


def _union_area(boxes):
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        high = length = 0.0
        for lo, hi in intervals:
            length += max(0.0, hi - max(lo, high))
            high = max(high, hi)
        total += (right - left) * length
    return total


def _cover(target, box):
    i = _inter(target, box)
    return _area(i) / _area(target) if i else 0.0


def _iou(a, b):
    i = _inter(a, b)
    if not i:
        return 0.0
    u = _area(a) + _area(b) - _area(i)
    return _area(i) / u if u else 0.0



TAU = 0.005  # frozen 2026-11-02: V* offline sweep — discrimination .589, append-full-image attack .094, correct-traj zeroing .14


def process_score(qid, wh, executed_boxes, cited_views=None):
    """H-tau (frozen): node score over precision-gated crops. Crops whose best-node precision
    is below TAU do not count toward that node's IoU or coverage. cited_views is accepted for
    interface compatibility and unused (last-crop=usage made citation parsing unnecessary)."""
    rec = index().get(qid)
    if rec is None or not rec.get("targets"):
        return 0, 0.0
    W, H = wh
    crops = [(b[0] / W, b[1] / H, b[2] / W, b[3] / H) for b in executed_boxes]
    targets = rec["targets"]
    scores = []
    for t in targets:
        gate = [c for c in crops if (_inter(t, c) and _area(_inter(t, c)) / _area(c) >= TAU)]
        iou = max([_iou(t, c) for c in gate] or [0.0])
        cov = _union_area([_inter(t, c) for c in gate if _inter(t, c)]) / _area(t) if gate else 0.0
        scores.append(min(1.0, 0.5 * (iou + cov)))
    return 1, sum(scores) / len(scores)
