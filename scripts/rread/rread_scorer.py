# R-read 评分器：L2 读取一致性 Φ_read（冻结版 v1）
# 冻结协议: ~/vlm-rl-signal-design/design/R_READ_FREEZE.zh.md (commit 681d6a3)
# 规则: commit 文本中找字段的唯一明确读数, 规范化精确匹配; 缺失/枚举/否定=0; 每字段最多一次。
"""用法:
  python rread_scorer.py <traj.jsonl> [--out scores.jsonl]   # 轨迹评分
  python rread_scorer.py --selfcheck                          # 14 组成对反例
"""
import json, os, re, sys
from pathlib import Path

INDEX_PATH = os.environ.get("RREAD_INDEX") or Path(__file__).resolve().parents[2] / "data/reading_index_20261103.json"


def load_index(path: str | None = None) -> dict:
    return json.load(open(path or INDEX_PATH))


# ---------- 数字规范化（Decimal 规则: 负号/会计括号/百分号/千分位） ----------
def norm_num(s: str):
    s = re.sub(r"[$€£¥₹]", "", s.strip())
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("%"):  # % 是显示符号, 去符号等价, 不做百分数缩放(预登记行为)
        s = s[:-1]
    s = re.sub(r",(\d{3})", r"\1", s)
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _negated(text: str, pos: int) -> bool:
    return bool(re.search(r"\b(?:not|except|rather than|over)\s*$", text[max(0, pos - 25):pos].lower()))


def match_num(text: str, target: str) -> bool:
    nt = norm_num(target)
    if nt is None:
        return False
    for m in re.finditer(r"-?[\d][\d,.]*\d", text):
        if norm_num(m.group()) != nt or _negated(text, m.start()):
            continue
        # 选项枚举守卫: "30, 50, 70, 80" 中列出的值不计
        if re.search(r"\d[,.]\s*$", text[max(0, m.start() - 10):m.start()]) and \
           re.match(r"\s*[,.]", text[m.end():m.end() + 2] if m.end() < len(text) else " "):
            continue
        # 会计括号: 陈述中写 (144,489) 目标 144,489 → 负号语义由字段决定, v1 视为匹配
        return True
    return False


def match_txt(text: str, target: str) -> bool:
    tl = target.lower().strip()
    if len(tl) < 2:
        return False
    for m in re.finditer(re.escape(tl), text.lower()):
        s, e = m.start(), m.end()
        b = text[s - 1] if s > 0 else " "
        a = text[e] if e < len(text) else " "
        if re.match(r"[a-zA-Z0-9]", b) or re.match(r"[a-zA-Z0-9]", a):  # 子串守卫 MUSEUM≠MUSEUMS
            continue
        if _negated(text, s):
            continue
        return True
    return False


def match_field(text: str, target: str, kind: str) -> bool:
    return match_num(text, target) if kind == "number" else match_txt(text, target)


def phi_read(entry: dict, text: str):
    """返回 (phi, z_list, K)。entry = index[qid]; text = commit 文本。"""
    zs = []
    for f in entry["fields"]:
        zs.append(1.0 if match_field(text, f["value"], f["kind"]) else 0.0)
    return (sum(zs) / len(zs) if zs else 0.0), zs, len(zs)


def score_traj(traj_path: str, index: dict | None = None, out_path: str | None = None):
    index = index or load_index()
    rows, n_elig, n_stated = [], 0, 0
    for line in open(traj_path):
        r = json.loads(line)
        if r.get("uid") and "final_text" in r and "sample_idx" in r:
            rows.append(r)
    out = []
    for r in rows:
        e = index.get(r["uid"])
        if not e or e.get("m") != 1:
            continue
        phi, zs, K = phi_read(e, r["final_text"] or "")
        out.append({"uid": r["uid"], "sample_idx": r["sample_idx"], "ok": r.get("ok"),
                    "K": K, "phi": phi, "z": zs})
        n_elig += 1
    if out_path:
        with open(out_path, "w") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return out, n_elig


# ---------- 14 组成对反例（冻结协议 §成对反例） ----------
SELFHECK = [
    ("correct 35,649", "reads 35,649", "35,649", "number", True),
    ("wrong 35,650", "reads 35,650", "35,649", "number", False),
    ("enum no match", "options A, B, C, D", "35", "number", False),
    ("enum guarded", "options 30, 50, 70, 80", "50", "number", False),
    ("not 30", "it is not 30, but 50", "30", "number", False),
    ("but 50", "it is not 30, but 50", "50", "number", True),
    ("repeated once", "reads 35,649 and 35,649 and 35,649", "35,649", "number", True),
    ("MUSEUM exact", "sign reads MUSEUM", "MUSEUM", "text", True),
    ("Museum case", "sign reads Museum", "MUSEUM", "text", True),
    ("MUSEUMS guard", "word is MUSEUMS", "MUSEUM", "text", False),
    ("(144,489)", "cell shows (144,489)", "144,489", "number", True),
    ("neg target", "cell shows -144,489", "(144,489)", "number", True),
    ("5.2%", "margin is 5.2%", "5.2%", "number", True),
    ("no number", "I see a table", "35,649", "number", False),
]


def _selfcheck() -> int:
    fails = 0
    for name, text, target, kind, expect in SELFHECK:
        got = match_field(text, target, kind)
        if got != expect:
            print(f"FAIL {name}: got {got} expect {expect}")
            fails += 1
    print(f"selfcheck: {len(SELFHECK) - fails}/{len(SELFHECK)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--selfcheck":
        sys.exit(_selfcheck())
    traj = sys.argv[1]
    out_path = None
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    res, n = score_traj(traj, out_path=out_path)
    import collections
    by_uid = collections.defaultdict(list)
    for o in res:
        by_uid[o["uid"]].append(o["phi"])
    print(f"rows={len(res)} groups={len(by_uid)} mean_phi={sum(o['phi'] for o in res)/max(1,len(res)):.4f}")
    var_groups = sum(1 for v in by_uid.values() if len(set(v)) > 1)
    print(f"groups_with_phi_variance={var_groups}")
