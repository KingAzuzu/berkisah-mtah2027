"""Simulator vs target table for the matched profile (IDSSE TUNE matches only)."""
import sys
import numpy as np
from . import atd_stats as A, degrade_matched as DM
from .data import TUNE, HALF_LEN, HALF_WID, load_idsse

T = {"seen": 0.5931, "seen_run": dict(mean=25.3, p10=2.2, p25=5.5, p50=13.8, p75=30.3, p90=61.7),
     "miss_run": dict(mean=17.2, p10=0.47, p25=6.1, p50=10.7, p75=21.1, p90=33.8, p99=104.8),
     "obj": 20.49, "trk": dict(mean=106.8, p10=2.6, p25=7.9, p50=35.5, p75=145, p90=285),
     "step": 0.203 / 25 * 25, "out": 0.0268}


def pooled(seed=0, matches=TUNE, clean=False):
    balls, xys, blocks, n = [], [], [], 0
    for m in matches:
        hs = load_idsse(m)
        hs = hs if clean else DM.degrade_match(hs, m, seed)
        for i, h in enumerate(hs):
            balls.append(h["ball"]); xys.append(h["xy"])
            blocks.append((i, n, n + len(h["ball"]) - 1)); n += len(h["ball"])
    team = np.repeat(np.array([0, 1], np.int8), 23)
    return A.compute_stats(np.concatenate(balls), np.concatenate(xys), team, blocks, (2 * HALF_LEN, 2 * HALF_WID))


def main():
    st = pooled()
    print("ball seen", st["ball_seen_fraction"], "target", T["seen"])
    for key, tk in (("ball_seen_run_s", "seen_run"), ("ball_missing_run_s", "miss_run"), ("slot_tracklet_s", "trk")):
        print(key, "count", st[key]["count"])
        for q, v in T[tk].items():
            print(f"  {q:5s} sim {st[key][q]:8.2f}  target {v:8.2f}  ratio {st[key][q] / v:5.2f}")
    print("objects", st["objects_per_frame"]["all"]["mean"], "target", T["obj"], "home/away",
          st["objects_per_frame"]["home"]["mean"], st["objects_per_frame"]["away"]["mean"], "p10", st["objects_per_frame"]["all"]["p10"])
    print("slots used", st["slots_ever_active"])
    print("ball step median m/frame", st["ball_step_m_both_frames_seen"]["p50"], "target 0.203")
    print("ball speed med/p99", st["ball_speed_ms_both_frames_seen"]["p50"], st["ball_speed_ms_both_frames_seen"]["p99"], "target 5.1/24.6")
    print("share >40", st["share_speed_over_40ms"], "target 0.0023")
    print("outside 1m", st["share_seen_outside_pitch_1m"], "target", T["out"])
    print("still speed med/p90", st["ball_speed_ms_nearly_stationary"]["p50"], st["ball_speed_ms_nearly_stationary"]["p90"], "target 0.32/1.75")


if __name__ == "__main__":
    main()
