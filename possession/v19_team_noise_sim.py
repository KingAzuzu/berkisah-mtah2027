"""v19 (CPU): how much do team-label errors in the tracking cost v10? Decides whether a jersey-colour team classifier is worth building.

Real competition tracking (team_label_check, human run): 13.5% of frames have > 11 home players, 5.5% > 11 away.
Simulation on matched2 TUNE halves, frozen v10 (in-sample; deltas only, base from v17: DFL 0.7329, merged 0.8241):
in each 10 s block, with prob q, one random player of a random team is moved to a new slot with the other team's label for that block.
q in (0.05, 0.13). Variant "near" (worst case, added after the random run): the swapped player is the one nearest the ball in the block. Decision fixed beforehand: if the q 0.13 drop is < 0.02 on both truths, the jersey idea is closed.
python -m possession.v19_team_noise_sim -> logs/v19/team_noise.json"""
import json
import numpy as np
from . import m2_lib as L
from .data import TUNE
from .pipeline import v10_predictor
from .score import score
from .v8_convention import merge
from .v7_decoder import LOG

NEAR_BALL = False
BASE = {"dfl": 0.7329, "merged": 0.8241}


def corrupt(h, q, seed, near=False):
    rng = np.random.default_rng(seed)
    xy, team = h["xy"].copy(), np.asarray(h["team"]).copy()
    T = len(xy)
    new_xy, new_team = [], []
    for s in range(0, T, 250):
        if rng.random() >= q:
            continue
        e = min(s + 250, T)
        on = np.flatnonzero(~np.isnan(xy[s:e, :, 0]).all(0))
        if not len(on):
            continue
        if near:
            b = h["ball"][s:e]
            d = np.nanmean(np.linalg.norm(xy[s:e, on] - b[:, None, :], axis=2), 0) if (~np.isnan(b[:, 0])).any() else None
            k = on[np.nanargmin(d)] if d is not None and np.isfinite(d).any() else rng.choice(on)
        else:
            k = rng.choice(on)
        col = np.full((T, 2), np.nan)
        col[s:e] = xy[s:e, k]
        xy[s:e, k] = np.nan
        new_xy.append(col); new_team.append(1 - team[k])
    if new_xy:
        xy = np.concatenate([xy, np.stack(new_xy, 1)], 1)
        team = np.r_[team, new_team].astype(team.dtype)
    return dict(h, xy=xy, team=team)


def task(arg):
    q, near = arg
    _, predict = v10_predictor()
    data = L.data2()
    preds = {m: [predict(corrupt(h, q, 7000 * j + i, near)) for i, h in enumerate(data[m])] for j, m in enumerate(TUNE)}
    out = {}
    for tn in ("dfl", "merged"):
        s = [float(score([dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in data[m]], preds[m])["spell_f1_mean"])
             for m in TUNE]
        out[tn] = {"mean": float(np.mean(s)), "per_match": s, "delta": float(np.mean(s) - BASE[tn])}
    return q, out


if __name__ == "__main__":
    import sys
    from multiprocessing import Pool
    if len(sys.argv) > 1 and sys.argv[1] == "near":
        NEAR_BALL = True
    with Pool(2) as pool:
        res = dict(pool.map(task, [(0.05, NEAR_BALL), (0.13, NEAR_BALL)]))
    for q, r in res.items():
        print(f"q {q}: dfl {r['dfl']['mean']:.4f} ({r['dfl']['delta']:+.4f}), merged {r['merged']['mean']:.4f} ({r['merged']['delta']:+.4f})")
    r = res[0.13]
    print("jersey idea:", "worth building" if min(-r["dfl"]["delta"], -r["merged"]["delta"]) >= 0.02 else "CLOSED (drop < 0.02)")
    (LOG / "v19").mkdir(exist_ok=True)
    json.dump({str(k): v for k, v in res.items()}, open(LOG / "v19" / ("team_noise_near.json" if NEAR_BALL else "team_noise.json"), "w"), indent=1)
