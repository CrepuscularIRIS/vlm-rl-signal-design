"""RL 题池 grpo_caption_v51_4k（4,096 题）逐题元数据 + hint 键 + 标注单位草案（规则版，待看图核对）。

输入（只读）：raw/rl4k_meta.jsonl（服务器 train.parquet 导出，sha256 见 README）、raw/match.json、
obs/benchmarks/funnel_v1_20261002/pool.jsonl、obs/src/caption_v51.py（task_hints）。
输出：items.jsonl、summary.json。用法：python3 build_types.py
"""
import collections
import json
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[2] / "obs/src"))
import caption_v51 as proto  # noqa: E402

ABSENT = re.compile(r"(does not|doesn't) (feature|contain|show|include|exist)|not (present|visible|shown|exist)|"
                    r"none of the|no such|cannot be (determined|found)|not mentioned", re.I)
COUNT_Q = re.compile(r"^\s*(how many|count\b|what is the (total )?number of)", re.I)
TABLE_Q = re.compile(r"\b(table|row|column|cell)\b", re.I)
CHART_Q = re.compile(r"\b(chart|graph|plot|axis|legend|bar|line|pie|curve|trend)\b", re.I)
REL_Q = re.compile(r"\b(left|right|above|below|closest|nearest|behind|in front of|next to|between|"
                   r"top|bottom|corner|beside|adjacent)\b", re.I)
MULTI_Q = re.compile(r"\b(which (of the following )?(two|three|corners)|compare|difference between|"
                     r"both|all of the|each of|highest|lowest|largest|smallest|most|least|rank)\b", re.I)

UNITS = {
    "single_box": "一个目标区域（单个物体 / 一段文字 / 一个元素）",
    "multi_set": "多个同类实例，无序（计数、找全部满足条件的）",
    "keyed_set": "带角色的多框：表格=单元格+行表头+列表头；图表=图例+数据标记+坐标轴刻度",
    "pair_relation": "两个对象 + 关系，关系可由两框坐标计算",
    "multi_compare": "多个候选之间比较 / 排序（每个候选一框）",
    "region_global": "需要整块图（学科示意图、乐谱、医学影像等），框=整幅或主图区",
    "absent": "正确答案为“不存在”类选项，无目标框；只能用探索",
    "unknown": "规则无法判断，需看图",
}


def option_text(question: str, letter: str) -> str:
    m = re.search(r"\(" + re.escape(letter) + r"\)\s*([^\n]*)", question)
    return m.group(1) if m else ""


def material_of(r: dict, m: dict) -> tuple[str | None, str | None, str]:
    if m["material"]:
        return m["material"], m["category"], "funnel"
    rel, q = r["rel"], r["question"]
    if r["source"] == "mme_realworld":
        top = rel.split("/")[2]
        if COUNT_Q.search(q):
            return "count", None, "path+question"
        return {"ocr_cc": "text", "diagram_and_table": "chart_table", "AutonomousDriving": "natural"}[top], None, "path"
    if r["source"] == "mmmu_pro":
        return "academic", "/".join(r["uid"].split("_")[1:-1]), "uid"
    return None, None, "none"


def draft_unit(r: dict, material: str | None, q_body: str) -> tuple[str, str]:
    gt_opt = option_text(r["question"], r["gt"])
    if ABSENT.search(gt_opt):
        return "absent", "gt 选项为不存在类"
    if material == "count" or COUNT_Q.search(q_body):
        return "multi_set", "计数"
    if material == "academic":
        return "region_global", "学科题"
    if material == "chart_table" or TABLE_Q.search(q_body) or CHART_Q.search(q_body):
        return "keyed_set", "表格/图表"
    if REL_Q.search(q_body):
        return "pair_relation", "关系词"
    if MULTI_Q.search(q_body):
        return "multi_compare", "比较/排序词"
    if material in ("text", "natural"):
        return "single_box", f"material={material}"
    return "unknown", "无规则命中"


def main() -> None:
    meta = [json.loads(x) for x in open(D / "raw/rl4k_meta.jsonl")]
    match = {m["uid"]: m for m in json.load(open(D / "raw/match.json"))}
    out, summ = [], collections.defaultdict(collections.Counter)
    for r in meta:
        m = match[r["uid"]]
        material, category, mat_src = material_of(r, m)
        q_body = r["question"].split("\n(A)")[0]
        unit, why = draft_unit(r, material, q_body)
        cat_l = (category or r["rel"]).lower().replace("_", " ")
        hints = proto.task_hints({"question": q_body, "category": cat_l, "material": material or "",
                                  "image_size": (r["w"], r["h"])}, limit=13)
        rec = dict(uid=r["uid"], source=r["source"], subdir="/".join(r["rel"].split("/")[1:-1]),
                   w=r["w"], h=r["h"], material=material, material_src=mat_src, category=category,
                   funnel_match=m["match"], hint_keys=hints, unit_draft=unit, unit_rule=why,
                   gt=r["gt"], gt_option=option_text(r["question"], r["gt"]))
        out.append(rec)
        summ["source×unit"][f"{r['source']}|{unit}"] += 1
        summ["source×material"][f"{r['source']}|{material}"] += 1
        summ["material_src"][mat_src] += 1
        summ["hint_any"].update(hints)
        summ["hint_none"][r["source"]] += not hints
        summ["long_side>=3000"][r["source"]] += max(r["w"], r["h"]) >= 3000
    with open(D / "items.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    json.dump({k: dict(v) for k, v in summ.items()} | {"units": UNITS}, open(D / "summary.json", "w"),
              ensure_ascii=False, indent=1)
    assert len(out) == 4096 and len({o["uid"] for o in out}) == 4096
    print(json.dumps({k: dict(v) for k, v in summ.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
