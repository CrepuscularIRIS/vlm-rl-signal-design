# 准入管线: 轨迹 → Φ_read 评分 → 实数据冒烟（reward_rread vs rread_scorer 一致性）→ ν 统计
"""用法: python run_admission.py <traj.jsonl> <out_scores.jsonl>"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rread_scorer import load_index, phi_read  # noqa: E402
import reward_rread  # noqa: E402

traj_path, out_path = sys.argv[1], sys.argv[2]
index = load_index()

rows = [json.loads(x) for x in open(traj_path)]
print(f"rows={len(rows)}")

out, mism, n_elig = [], 0, 0
for r in rows:
    e = index.get(r["uid"])
    if not e or e.get("m") != 1:
        continue
    n_elig += 1
    phi, zs, K = phi_read(e, r["final_text"] or "")
    vy = float(bool(r.get("ok")) and r.get("end") == "answer")  # 任务信号 V·Y
    out.append({"uid": r["uid"], "sample_idx": r["sample_idx"], "ok": r.get("ok"),
                "vy": vy, "K": K, "phi": phi, "z": zs})
    # 冒烟: 训练侧路径 = 全 turns 拼接 → commit 提取（与 trainer decode 同构）
    full = "".join(t["text"] for t in (r.get("turns") or []))
    m, phi_train = reward_rread.process_score(r["uid"], full)
    if m != 1 or abs(phi_train - phi) > 1e-9:
        mism += 1
        if mism <= 3:
            print(f"  MISMATCH {r['uid']} s{r['sample_idx']}: scorer={phi:.3f} train={phi_train:.3f} m={m}")

with open(out_path, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"eligible={n_elig} smoke_mismatches={mism}")
