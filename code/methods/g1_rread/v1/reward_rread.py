# G1 (R-read) 过程奖励: Φ_read = commit 文本读数 vs L2 读取索引规范化精确匹配
# 冻结: R = V*(Y + 0.25*m*Phi_read); 本模块只提供 (m, phi), 不含 Y/V。
# 索引 sidecar: 同目录 rread_index.json (reward-only, 不进 actor 输入)。
import json, os, re

_DIR = os.path.dirname(os.path.abspath(__file__))
_INDEX = None


def _index():
    global _INDEX
    if _INDEX is None:
        with open(os.path.join(_DIR, "rread_index.json")) as f:
            _INDEX = json.load(f)
    return _INDEX


# ---------- 冻结 v1 匹配器（与 obs/src/rread/rread_scorer.py 逐行同源, 预登记 14 反例） ----------
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
        if re.search(r"\d[,.]\s*$", text[max(0, m.start() - 10):m.start()]) and \
           re.match(r"\s*[,.]", text[m.end():m.end() + 2] if m.end() < len(text) else " "):
            continue
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
        if re.match(r"[a-zA-Z0-9]", b) or re.match(r"[a-zA-Z0-9]", a):
            continue
        if _negated(text, s):
            continue
        return True
    return False


def commit_text(full_response: str) -> str:
    """commit = 最后一次 </grounding> 之后的文本(无 grounding 则全文)。预登记规则。"""
    pos = full_response.rfind("</grounding>")
    return full_response[pos + len("</grounding>"):] if pos >= 0 else full_response


def process_score(qid: str, full_response: str):
    """返回 (m, phi)。m=0 → 不适用(题不在索引/资格不符), 奖励回落 V*Y。"""
    entry = _index().get(qid)
    if not entry or entry.get("m") != 1:
        return 0, 0.0
    text = commit_text(full_response or "")
    zs = [1.0 if (match_num(text, f["value"]) if f["kind"] == "number" else match_txt(text, f["value"]))
          else 0.0 for f in entry["fields"]]
    return 1, (sum(zs) / len(zs) if zs else 0.0)


# ---------- 预登记成对反例（训练前必过; 与冻结协议 §成对反例 对应） ----------
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
    ("commit only", "OBSERVE x<grounding>{}</grounding>OBSERVE 99<grounding>{}</grounding>reads 35,649", "35,649", "number", True),
    ("commit drops earlier", "reads 35,649<grounding>{}</grounding>final words", "35,649", "number", False),
]

if __name__ == "__main__":
    fails = 0
    for name, text, target, kind, expect in SELFHECK:
        got = match_num(text, target) if kind == "number" else match_txt(text, target)
        if kind == "number" and name in ("commit only", "commit drops earlier"):
            got = match_num(commit_text(text), target)  # commit 规则用例
        if got != expect:
            print(f"FAIL {name}: got {got} expect {expect}")
            fails += 1
    print(f"selfcheck: {len(SELFHECK) - fails}/{len(SELFHECK)} passed")
    sys = __import__("sys")
    sys.exit(1 if fails else 0)
