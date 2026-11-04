"""Write relation_canonical.jsonl from relation_computable.json + visual-check verdicts below."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RR = "$OBS/notes/rereview_20261104/relation_computable.json"

OK_OVERLAP = "标注方向正确，严格几何因框重叠无法判定"
OK = "标注方向正确，严格几何一致"
WRONG = "标注方向错误"
NA = "不是方向关系（NA）"

# (row index, uid) -> (dir_canonical, verdict, seen)
VISUAL = {
    (1, "mme_14871"): ("D", OK_OVERLAP, "b1=JEEP 标牌，b2=ABARTH 标牌；JEEP 在 ABARTH 下方偏右，两框 y 投影重叠 0.003（b1 y1=0.256，b2 y2=0.259）"),
    (3, "mme_15377"): ("U", OK_OVERLAP, "b1=Alexandria 地名，b2=210 路标；Alexandria 在 210 上方，两框 y 投影重叠 0.001，x 投影 b2 位于 b1 范围内"),
    (6, "mme_15892"): ("RU", OK_OVERLAP, "b1=Montpellier，b2=Narbonne；Montpellier 在 Narbonne 右上方，y 不重叠，x 投影重叠（b1 x1=0.859 < b2 x2=0.871），中心 dx=+0.031"),
    (10, "mme_16113"): ("R", OK_OVERLAP, "b1=Dragon Insurance 标题，b2=绿色恐龙图；标题在恐龙右侧（与恐龙头部同高），x 投影重叠（b1 x1=0.751 < b2 x2=0.764），y 投影重叠"),
    (22, "test_Economics_56"): ("RU", OK_OVERLAP, "b1=红点 B 及标签，b2=黑点 A 及标签；B 在 A 右上方，两框在 x、y 两轴都重叠（含字母标签）"),
    (25, "worldbench_1022"): ("NA", NA, "b1=单个蜜蜂糖（放大可见一只黑眼、一只空凹眼），b2=整组蜜蜂糖；b1 位于 b2 框内中部，关系为群体内位置"),
    (26, "worldbench_108"): ("U", OK, "b1=CREATE 菜单项，b2=选中的 POST 标签；CREATE 在 POST 正上方，x 投影重叠；关系文本含 above，norm 为 None"),
    (28, "worldbench_12"): ("U", OK, "b1=绿色光标点，b2=Twitter Tweet 按钮；光标点在按钮正上方，x 投影重叠；关系文本含『上方』，norm 为 None"),
    (29, "worldbench_1268"): ("LU", WRONG, "b1=空黑箱，b2=装鱼黑箱；图中 b1 在 b2 上方偏左（中心 dx=-0.067，dy=-0.112），y 不重叠，x 投影重叠（0.514–0.542）；标注仅 L"),
    (30, "worldbench_1274"): ("NA", NA, "b1=举两张纸的深色外套男子（放大核对），b2=整个合影群体（几乎全图）；b1 位于 b2 框内左部，关系为群体内位置"),
    (34, "worldbench_1552"): ("D", OK, "b1=下方空白格（Maint Technician 列），reference=n1（L1 negatives 中的上方空白格，Maint Supert 列），不在 L1 boxes；b1 在 n1 下方偏左，y 不重叠（b1 y1=0.386 > n1 y2=0.315），x 边界相接（b1 x2=n1 x1=0.642）"),
    (36, "worldbench_1938"): ("NA", NA, "b1=Obsolete Software 复选框，b2=Tools 侧栏按钮；b1 在 b2 右侧，b1 中心 y=0.4835 与 b2 中心 y=0.495 相近；关系文本为垂直中心对齐"),
    (37, "worldbench_343"): ("RD", WRONG, "b2=あまんきみこ 标牌与 い 标牌之间的书组，b3=上一层左端『あ』标牌；图中 b2 在 b3 右下方（中心 dx=+0.024，dy=+0.095），y 边界相接（b3 y2=b2 y1=0.365），x 投影重叠 0.001；标注仅 R"),
    (38, "worldbench_343"): ("L", OK, "b2=同上书组，reference=n4（L1 negatives 中的『む』标签，放大核对框住 む 字），不在 L1 boxes；b2 在 n4 左侧，x 不重叠（b2 x2=0.415 < n4 x1=0.586），y 投影重叠"),
    (39, "worldbench_4"): ("U", OK, "b1=标题中的 Zoom，b2=覆盖 Zoom 链接的黑色矩形；b1 在 b2 上方，b2 左边缘 x=0.192 落在 b1 x 范围 [0.145,0.233] 内；关系文本方向词为『上方』，norm=LU 中的 L 对应文本『左边缘』；图中 Add guests 处红框为原图自带"),
    (42, "worldbench_51"): ("NA", NA, "b1=Address 行，b2=Chassis Id 行；两框左边缘同为 x=0.106，b1 在 b2 上方；关系文本为 left-aligned（左对齐）"),
    (43, "worldbench_54"): ("D", OK, "b1=CLS 旁的 CORE 标签，b2=All 标签；CORE 在 All 下方，b1 中心 x=0.405 落在 b2 x 范围 [0.384,0.409] 内；关系文本含 below，norm 为 None"),
    (44, "worldbench_569"): ("NA", NA, "b1=标题 CUTTING THROUGH ROCKS，b2=Sundance 奖项标识；b1 在 b2 下方，b1 右边缘 x=0.622 落在 b2 x 范围 [0.483,0.679] 内；关系文本为右边缘与 b2 水平范围的对齐"),
}


def main():
    rows = json.load(open(RR))
    seen = set()
    with open(os.path.join(HERE, "relation_canonical.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            key = (i, r["uid"])
            if key in VISUAL:
                d, verdict, note = VISUAL[key]
                src, cn = "visual_check", f"{verdict}｜{note}"
                seen.add(key)
            else:
                assert r["norm"] and set(r["norm"]) <= set(r["geo_strict"] or ""), r
                d, src, cn = r["norm"], "rule+geometry", f"norm={r['norm']} geo_strict={r['geo_strict']}"
            f.write(json.dumps({"uid": r["uid"], "subject": r["subject"], "reference": r["reference"],
                                "relation_raw": r["relation"], "dir_canonical": d, "source": src,
                                "check_note": cn}, ensure_ascii=False) + "\n")
    assert seen == set(VISUAL), set(VISUAL) - seen
    print(len(rows), "rows;", len(seen), "visual")


if __name__ == "__main__":
    main()
