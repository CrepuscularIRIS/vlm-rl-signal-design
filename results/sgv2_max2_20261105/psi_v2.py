"""max2 的过程分 S（psi v2，max 模式）参考实现。

训练实际调用的是 code/methods/sg/v2/reward_sg.py（CAPTION_SG_PSI=max）；本文件只保留 max 模式用到的部分，
`python psi_v2.py` 会在随机样例上与它逐数值对照。

记号：框 = (x1, y1, x2, y2)，原图像素；crops = 已执行的裁图框（原图本身不算 crop）。
  prox_b  = 0.5 × (max_c IoU(b, c) + |b ∩ ∪crops| / |b|)            单个目标框的逼近分
  prox    = max_{S ∈ evidence_sets} mean_{b ∈ S} prox_b               取最好的一组证据
  explore = |∪crops| / (W × H)                                       全部历史裁图的并集面积比
  psi     = max(prox, ALPHA × explore)
  S       = clip(psi_T − psi_0, 0, 1)，psi_0 只有原图（无 crop）
奖励 r = V × (Y + 0.1 × m × S)，再做标准 GRPO 组内归一。
"""
from __future__ import annotations

from typing import Mapping, Sequence

Box = tuple[float, float, float, float]
ALPHA = 0.25


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a: Box, b: Box) -> Box | None:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[2] > box[0] and box[3] > box[1] else None


def union_area(boxes: Sequence[Box]) -> float:
    """矩形并集的精确面积：按 x 坐标切条，条内合并 y 区间。"""
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        covered = top = 0.0
        for lo, hi in spans:
            covered += max(0.0, hi - max(lo, top))
            top = max(top, hi)
        total += (right - left) * covered
    return total


def iou(a: Box, b: Box) -> float:
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    shared = area(inter)
    return shared / (area(a) + area(b) - shared)


def prox_one(target: Box, crops: Sequence[Box]) -> float:
    """单个目标框：最大 IoU 与被裁图并集覆盖比例的均值。"""
    if not crops:
        return 0.0
    pieces = [p for p in (intersect(target, c) for c in crops) if p]
    cover = min(1.0, union_area(pieces) / area(target)) if pieces and area(target) > 0 else 0.0
    return 0.5 * (max(iou(target, c) for c in crops) + cover)


def psi(targets: Mapping[str, Box], evidence_sets: Sequence[Sequence[str]],
        crops: Sequence[Box], image_wh: tuple[float, float]) -> float:
    """psi = max(prox, ALPHA × explore)。"""
    prox = {key: prox_one(box, crops) for key, box in targets.items()}
    approach = max(sum(prox[k] for k in s) / len(s) for s in evidence_sets)
    width, height = image_wh
    explore = min(1.0, union_area(crops) / (width * height)) if crops else 0.0
    return max(approach, ALPHA * explore)


def process_score(targets: Mapping[str, Box], evidence_sets: Sequence[Sequence[str]],
                  crops: Sequence[Box], image_wh: tuple[float, float]) -> float:
    """S = clip(psi_T − psi_0, 0, 1)。"""
    gain = psi(targets, evidence_sets, crops, image_wh) - psi(targets, evidence_sets, [], image_wh)
    return min(1.0, max(0.0, gain))


def _selfcheck() -> None:
    """与训练实现逐数值对照（2,000 个随机样例）。"""
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code' / 'methods' / 'sg' / 'v2'))
    import reward_sg  # 训练实际使用的实现

    reward_sg.PSI_MODE = 'max'
    rng = random.Random(0)

    def rand_box(w: float, h: float) -> Box:
        x1, y1 = rng.uniform(0, w - 60), rng.uniform(0, h - 60)
        return (x1, y1, rng.uniform(x1 + 56, w), rng.uniform(y1 + 56, h))

    for _ in range(2000):
        w, h = rng.choice([(1000, 800), (2250, 1500), (640, 480)])
        targets = {f'b{i}': rand_box(w, h) for i in range(rng.randint(1, 3))}
        keys = list(targets)
        sets = [keys] if rng.random() < .5 else [[k] for k in keys]
        crops = [rand_box(w, h) for _ in range(rng.randint(0, 6))]
        rec = {'m': 1, 'sets': sets, 'nodes': {k: {'bbox': [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
                                                   'kind': 'object', 'key': True} for k, b in targets.items()}}
        reward_sg._INDEX = {'q': rec}
        expected = reward_sg.process_score('q', (w, h), crops)[1]
        got = process_score(targets, sets, crops, (w, h))
        assert abs(expected - got) < 1e-9, (expected, got)
    print('psi_v2 selfcheck: 2000/2000 match reward_sg.py (CAPTION_SG_PSI=max)')


if __name__ == '__main__':
    _selfcheck()
