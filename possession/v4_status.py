"""Progress line for the ablation stage, computed from the fold files present (called by every training process)."""
import os, time
from pathlib import Path

HERE = Path(__file__).parent
PLAN = [(c, s) for c in ("full", "nomet", "nosc", "nobd") for s in (0, 1, 2)] + [("full", 3), ("full", 4)]
TOTAL = len(PLAN) * 5


def update():
    d = HERE / "logs" / "v4"
    done = len(list(d.glob("*.npz")))
    t0f = HERE / "logs" / "v4_t0.txt"
    if not t0f.exists():
        t0f.write_text(str(time.time()))
    t0 = float(t0f.read_text())
    el = (time.time() - t0) / 60
    eta = el / max(done, 1) * (TOTAL - done) if done else 0
    pct = 25 + int(60 * done / TOTAL)
    (HERE / "logs" / "v4_progress.txt").write_text(f"{pct}% | stage 4/6: ablation runs | jobs done {done}/{TOTAL} | ETA {eta:.0f} min\n")


PLAN_C = [(c, s) for c in ("cA", "cB") for s in range(10, 15)]
TOTAL_C = len(PLAN_C) * 5


def update_c():
    """Progress for the confirmation run (seeds 10-14, configs cA = IDSSE+Metrica, cB = IDSSE only; both no SkillCorner)."""
    d = HERE / "logs" / "v4"
    done = len(list(d.glob("cA_s1?_*.npz"))) + len(list(d.glob("cB_s1?_*.npz")))
    t0f = HERE / "logs" / "v4c_t0.txt"
    if not t0f.exists():
        t0f.write_text(str(time.time()))
    el = (time.time() - float(t0f.read_text())) / 60
    eta = el / max(done, 1) * (TOTAL_C - done) if done else 0
    pct = int(100 * done / TOTAL_C)
    (HERE / "logs" / "v4c_progress.txt").write_text(f"{pct}% | stage 1/1: confirmation LOMO | jobs {done}/{TOTAL_C} | ETA {eta:.0f} min\n")
