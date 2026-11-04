"""Stop an EasyR1 run once validation accuracy has plateaued (rule fixed before the run starts).

Rule: validation every 25 steps (440 val questions). From step 150 on, after each validation compare the mean of
the latest 3 validation accuracies with the mean of the 3 before them; gain < 0.01 => plateau. Then wait until the
checkpoint of that step is fully saved (checkpoint_tracker.json last_global_step >= that step) and stop the run's own
process tree. EasyR1 keeps the best-validation checkpoint as well. max_steps in the launch is the hard ceiling.

usage: python plateau_stop.py TAG LOG CKPT_DIR LAUNCHER_PID STATUS_JSON
"""
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

MIN_STEP, WINDOW, MIN_GAIN, POLL = 150, 3, 0.01, 300
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^\s*\([A-Za-z]+ pid=\d+(?:, ip=[0-9.]+)?\)\s*")  # Ray "(Runner pid=123) "


def val_series(log: str) -> list[tuple[int, float]]:
    out, step, in_val = [], None, False
    for line in open(log, errors="replace"):
        s = PREFIX.sub("", ANSI.sub("", line)).strip()
        m = re.fullmatch(r"Step (\d+)", s)
        if m:
            step, in_val = int(m.group(1)), False
        elif s == "val:":
            in_val = True
        elif in_val and s.startswith("accuracy_reward:") and step is not None:
            out.append((step, float(s.split(":")[1])))
            in_val = False
    return sorted(dict(out).items())


def descendants(pid: int) -> list[int]:
    kids = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                ppid = int(open(p / "stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(p.name))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_tree(pid: int, ray_tmp: str) -> list[int]:
    pids = descendants(pid) + [pid]
    for p in Path("/proc").iterdir():  # Ray workers of this run only (their cmdline carries this run's RAY_TMPDIR)
        if p.name.isdigit():
            try:
                if ray_tmp in open(p / "cmdline").read():
                    pids.append(int(p.name))
            except OSError:
                pass
    pids = sorted(set(pids) - {os.getpid()})
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        time.sleep(20)
    return pids


def main() -> None:
    tag, log, ckpt, launcher, status_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    status = {"tag": tag, "rule": f"step>={MIN_STEP}: mean(last {WINDOW} val) - mean(prev {WINDOW}) < {MIN_GAIN}"}
    while True:
        vals = [(s, a) for s, a in val_series(log) if s > 0]
        status.update(vals=vals, updated=time.strftime("%F %T"))
        if not alive(launcher):
            status["result"] = "run ended by itself (max_steps or error)"
            break
        if len(vals) >= 2 * WINDOW and vals[-1][0] >= MIN_STEP:
            last = sum(a for _, a in vals[-WINDOW:]) / WINDOW
            prev = sum(a for _, a in vals[-2 * WINDOW:-WINDOW]) / WINDOW
            status["last_gain"] = round(last - prev, 4)
            if last - prev < MIN_GAIN:
                target = vals[-1][0]
                tracker = Path(ckpt) / "checkpoint_tracker.json"
                while alive(launcher):
                    if tracker.exists() and json.loads(tracker.read_text()).get("last_global_step", -1) >= target:
                        break
                    time.sleep(30)
                time.sleep(60)
                status.update(result=f"plateau at step {target} (gain {last - prev:+.4f})",
                              stopped_pids=stop_tree(launcher, f"/tmp/ray_v52_{tag}"))
                break
        Path(status_path).write_text(json.dumps(status, indent=1))
        time.sleep(POLL)
    status["finished"] = time.strftime("%F %T")
    Path(status_path).write_text(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
