"""Simulator check (IDSSE TUNE, CPU): label-free input statistics of clean IDSSE, matched2 and matched3, next to the real-ATD
aggregates we already have. python -m possession.v21_simstats -> logs/v21/simstats.json

Real references (aggregates only): ball seen 0.593, missing runs median 10.7 s / mean 17.2 s / p10 0.47 s, seen runs median
13.8 s / mean 25.3 s (atd_stats, HANDOVER); P(missing|D) 0.702, P(missing|alive) 0.101 with v2's D as proxy (ball_loss_check).
Public competitor measurements, unverified by us: dead-time ball seen moves ~4.7 m/s, visible-still ball 5-15% of ball frames,
live ball acceleration p50/p90/p99 6.6/18.7/33.8 m/s^2. atd_stats3 (human) measures the same quantities on the real tracking."""
import json

import numpy as np

from . import degrade_m3 as M3, degrade_matched2 as DM2
from .data import FPS, TUNE, load_idsse, pm
from .pipeline import HERE

STILL_W, STILL_M = 25, 0.5


def ball_stats(halves):
    seen_l, dead_l, spd_l, still_l, acc_l, miss_runs, seen_runs = [], [], [], [], [], [], []
    for h in halves:
        b, dead = h["ball"].astype(float), h["truth"] == pm.D
        seen = ~np.isnan(b[:, 0])
        seen_l.append(seen); dead_l.append(dead)
        v = np.full(len(b), np.nan)
        v[5:] = np.linalg.norm(b[5:] - b[:-5], axis=1) * FPS / 5
        spd_l.append(v)
        disp = np.full(len(b), np.nan)
        disp[STILL_W:] = np.linalg.norm(b[STILL_W:] - b[:-STILL_W], axis=1)
        still_l.append(disp)
        a = np.full(len(b), np.nan)
        a[1:-1] = np.linalg.norm(b[2:] - 2 * b[1:-1] + b[:-2], axis=1) * FPS ** 2
        acc_l.append(np.where(dead, np.nan, a))
        for mask, out in ((~seen, miss_runs), (seen, seen_runs)):
            out += [(e - s) / FPS for s, e in M3.runs(mask)]
    seen, dead = np.concatenate(seen_l), np.concatenate(dead_l)
    spd, disp, acc = map(np.concatenate, (spd_l, still_l, acc_l))
    mr, sr = np.array(miss_runs), np.array(seen_runs)
    fin = lambda x: x[np.isfinite(x)]
    return {
        "seen": float(seen.mean()), "p_miss_dead": float((~seen[dead]).mean()), "p_miss_alive": float((~seen[~dead]).mean()),
        "miss_run_med_s": float(np.median(mr)) if len(mr) else None, "miss_run_mean_s": float(mr.mean()) if len(mr) else None,
        "miss_run_p10_s": float(np.percentile(mr, 10)) if len(mr) else None,
        "seen_run_med_s": float(np.median(sr)), "seen_run_mean_s": float(sr.mean()),
        "dead_seen_speed_mean": float(np.mean(fin(spd[dead & seen]))),
        "dead_seen_still_share": float(np.mean(fin(disp[dead & seen]) < STILL_M)),
        "seen_still_share": float(np.mean(fin(disp[seen]) < STILL_M)),
        "live_acc_p50_p90_p99": [float(x) for x in np.percentile(fin(acc), (50, 90, 99))],
        "dead_share": float(dead.mean()),
    }


def main():
    clean = {m: load_idsse(m) for m in TUNE}
    res = {"clean": ball_stats([h for m in TUNE for h in clean[m]])}
    res["matched2"] = ball_stats([h for m in TUNE for h in DM2.degrade_match(clean[m], m, 0)])
    res["matched3"] = ball_stats([h for m in TUNE for h in M3.degrade_match(clean[m], m, 0)])
    for k in (1, 2, 3):
        res[f"matched3_dr{k}"] = ball_stats([h for m in TUNE for h in M3.degrade_match(clean[m], m, 1000 + k, M3.variant(k))])
    for name, r in res.items():
        print(f"{name:13s} " + " ".join(f"{k}={v if not isinstance(v, float) else round(v, 3)}" for k, v in r.items()))
    out = HERE / "logs" / "v21"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "simstats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
