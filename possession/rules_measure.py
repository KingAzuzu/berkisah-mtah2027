"""WS2 step 1: how does DFL BallPossession behave around changes and in flight? Clean IDSSE, tuning matches only.

    python -m possession.rules_measure

Prints aggregate numbers only (pooled over the 5 tuning matches) and writes possession/logs/rules_measure.json.
"""
import json

import numpy as np

from .data import FPS, TUNE, load_idsse, pm
from .lomo import LOG_DIR

NEAR = 1.5
FAR = 3.0
OFFS = np.arange(-50, 51, 5)


def nearest(half):
    d = np.linalg.norm(half["xy"] - half["ball"][:, None, :], axis=2)
    with np.errstate(all="ignore"):
        dh = np.fmin.reduce(np.where(half["team"] == 0, d, np.nan), axis=1)
        da = np.fmin.reduce(np.where(half["team"] == 1, d, np.nan), axis=1)
    return dh, da


def main():
    agree_by_dist = {}   # distance bin -> [n, n_nearest_team_equals_truth]
    flight = {"n_alive": 0, "n_far": 0, "far_eq_last": 0, "far_eq_next": 0, "far_n_both": 0,
              "far_last_eq_next": 0, "far_truth_eq_last_when_sides_agree": 0, "far_sides_agree": 0,
              "far_truth_eq_last_when_sides_differ": 0, "far_sides_differ": 0}
    prof_new = {o: [] for o in OFFS}   # distance of nearest NEW-team player to ball at offset from truth change
    prof_old = {o: [] for o in OFFS}
    lag_new_arrive, lag_old_leave, n_ch = [], [], 0
    n_ch_via_d = 0
    holds_after = []   # frames before change that old team was last within NEAR, relative to change
    for mid in TUNE:
        for h in load_idsse(mid):
            tr = h["truth"]
            dh, da = nearest(h)
            dmin = np.fmin(dh, da)
            near_team = np.where(dh < da, pm.H, pm.A)
            alive = tr != pm.D
            # agreement of nearest team with truth by distance bin
            for lo, hi in ((0, 1), (1, 1.5), (1.5, 3), (3, 6), (6, 10), (10, 1e9)):
                m = alive & (dmin >= lo) & (dmin < hi)
                k = f"{lo}-{hi}"
                a = agree_by_dist.setdefault(k, [0, 0])
                a[0] += int(m.sum()); a[1] += int((near_team[m] == tr[m]).sum())
            # flight: alive frames with nearest > FAR. last / next close (<NEAR) player's team
            close = alive & (dmin < NEAR)
            idx = np.arange(len(tr))
            last = np.maximum.accumulate(np.where(close, idx, -1))
            nxt = np.minimum.accumulate(np.where(close, idx, len(tr) + 10)[::-1])[::-1]
            far = alive & (dmin >= FAR)
            flight["n_alive"] += int(alive.sum()); flight["n_far"] += int(far.sum())
            ok = far & (last >= 0) & (nxt < len(tr))
            lt = near_team[np.clip(last, 0, None)]
            nt = near_team[np.clip(nxt, 0, len(tr) - 1)]
            flight["far_n_both"] += int(ok.sum())
            flight["far_eq_last"] += int((ok & (tr == lt)).sum())
            flight["far_eq_next"] += int((ok & (tr == nt)).sum())
            flight["far_last_eq_next"] += int((ok & (lt == nt)).sum())
            ag = ok & (lt == nt)
            flight["far_sides_agree"] += int(ag.sum())
            flight["far_truth_eq_last_when_sides_agree"] += int((ag & (tr == lt)).sum())
            di = ok & (lt != nt)
            flight["far_sides_differ"] += int(di.sum())
            flight["far_truth_eq_last_when_sides_differ"] += int((di & (tr == lt)).sum())
            # changes: direct H<->A on alive frames
            ch = np.flatnonzero((tr[1:] != tr[:-1]) & (tr[1:] != pm.D) & (tr[:-1] != pm.D)) + 1
            n_ch_via_d += int(np.sum((tr[1:] != tr[:-1]) & (tr[1:] == pm.D)))
            for t in ch:
                new, old = tr[t], tr[t - 1]
                n_ch += 1
                dn, do = (dh, da) if new == pm.H else (da, dh)
                for o in OFFS:
                    if 0 <= t + o < len(tr):
                        prof_new[o].append(dn[t + o]); prof_old[o].append(do[t + o])
                w = slice(max(0, t - 5 * FPS), min(len(tr), t + 5 * FPS))
                # first frame at/after t-5s where new team within NEAR, relative to t
                arr = np.flatnonzero(dn[w] < NEAR)
                arr = arr[(arr + w.start) >= t - 5 * FPS]
                if len(arr):
                    after = arr[(arr + w.start) >= t]
                    before = arr[(arr + w.start) < t]
                    # arrival: first near frame of the new team in the run that contains or follows t
                    cand = after[0] + w.start - t if len(after) else None
                    lag_new_arrive.append(cand if cand is not None else np.nan)
                else:
                    lag_new_arrive.append(np.nan)
                arr = np.flatnonzero(do[w] < NEAR) + w.start
                bef = arr[arr < t]
                lag_old_leave.append(bef[-1] - t if len(bef) else np.nan)  # last frame old team within NEAR
    lagn, lago = np.array(lag_new_arrive, float), np.array(lag_old_leave, float)

    def q(x):
        x = x[~np.isnan(x)]
        return {"n": int(len(x)), "q10_q50_q90_frames": [float(v) for v in np.percentile(x, [10, 50, 90])]}

    out = {
        "n_direct_changes_HA": n_ch, "n_alive_to_dead_changes": n_ch_via_d,
        "nearest_team_equals_truth_by_dist_m": {k: {"n": v[0], "acc": v[1] / max(v[0], 1)} for k, v in agree_by_dist.items()},
        "flight": flight,
        "median_dist_new_team_by_offset": {int(o): float(np.nanmedian(prof_new[o])) for o in OFFS},
        "median_dist_old_team_by_offset": {int(o): float(np.nanmedian(prof_old[o])) for o in OFFS},
        "new_team_first_within_1.5m_after_change_frames": q(lagn),
        "old_team_last_within_1.5m_before_change_frames": q(lago),
    }
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "rules_measure.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
