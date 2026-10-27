"""Test B: rest-defence surplus at open-play possession loss vs. how fast the opponent attacks (see PREREGISTRATION.md)."""
import json
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # tiny problems: BLAS threads only thrash
    os.environ[v] = "1"
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import idsse

FPS = 25
ZONE_X = -52.5 + 35.0  # losing team (attacks +x) own final third: x <= -17.5 in that team's frame
SEED, N_BOOT = 0, 1000


def losses_in_match(mid):
    m = idsse.load_match(mid)
    shots = idsse.load_events(mid)["shots"]
    shot_fr = [(s["team"],) + tuple(idsse.to_frame(m, s["time"]) or (None, None)) for s in shots]
    home, away = m["home"], m["away"]
    rows, short_runs = [], 0
    for hi, h in enumerate(m["halves"]):
        poss, st, T = h["poss"], h["status"], h["T"]
        chg = np.where((poss[1:] != poss[:-1]) & np.isin(poss[1:], (1, 2)) & np.isin(poss[:-1], (1, 2))
                       & (st[1:] == 1) & (st[:-1] == 1))[0] + 1  # open play: ball alive on both sides of the change
        # x-positions in the frame where each possible losing team attacks +x (computed once per half)
        fr = {X: (idsse.ball_xy(h, X), idsse.team_xy(h, home, X)[..., 0], idsse.team_xy(h, away, X)[..., 0]) for X in (home, away)}
        for i in chg:
            X, Y = (home, away) if poss[i - 1] == 1 else (away, home)
            ball, xh, xa = fr[X]
            xX, xY = (xh, xa)[X == away][i], (xh, xa)[Y == away][i]
            gX, gY = h["teams"][X]["gk"], h["teams"][Y]["gk"]
            bx = float(ball[i, 0])
            S = int((xX[~gX] < bx).sum() - (xY[~gY] < bx).sum())  # NaN compares False: off-pitch players not counted
            end = min(T, i + 10 * FPS + 1)
            keep = (poss[i:end] == (1 if Y == home else 2)) & (st[i:end] == 1)
            stop = int(np.argmin(keep)) if not keep.all() else len(keep)  # first frame where Y no longer keeps live possession
            y1 = int((ball[i:i + stop, 0] <= ZONE_X).any())
            run = int(np.argmax(poss[i:] != poss[i])) if (poss[i:] != poss[i]).any() else T - i
            short_runs += run < 5
            y2 = int(any(t == Y and hh == hi and i <= f <= i + 15 * FPS for t, hh, f in shot_fr))
            rows.append((mid, S, bx, y1, y2, int(bx <= ZONE_X)))
    return rows, short_runs


def fit(X, y):
    return LogisticRegression(C=np.inf, max_iter=2000).fit(X, y)


def boot_ci(d, ycol, rng):
    mids = np.unique(d[:, 0])
    cs = []
    for _ in range(N_BOOT):
        pick = rng.choice(mids, len(mids))
        sub = np.concatenate([d[d[:, 0] == m] for m in pick])
        y = sub[:, ycol].astype(int)
        if len(np.unique(y)) < 2:
            continue
        cs.append(fit(sub[:, [1, 2]].astype(float), y).coef_[0, 0])
    return float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)), len(cs)


def lomo(d, ycol):
    full, base = [], []
    for m in np.unique(d[:, 0]):
        tr, te = d[:, 0] != m, d[:, 0] == m
        y = d[:, ycol].astype(int)
        Xf = d[:, [1, 2]].astype(float)
        if len(np.unique(y[te])) < 2:
            continue
        full.append(roc_auc_score(y[te], fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]))
        base.append(roc_auc_score(y[te], fit(Xf[tr][:, 1:], y[tr]).predict_proba(Xf[te][:, 1:])[:, 1]))
    return full, base


def analyse(d, ycol, rng):
    y, X = d[:, ycol].astype(int), d[:, [1, 2]].astype(float)
    coef = fit(X, y).coef_[0]
    lo, hi, nb = boot_ci(d, ycol, rng)
    full, base = lomo(d, ycol)
    return {"coef_S": float(coef[0]), "coef_ballx": float(coef[1]), "ci95_S": [lo, hi], "n_boot_ok": nb,
            "auc_full_by_match": full, "auc_base_by_match": base,
            "auc_full_mean": float(np.mean(full)), "auc_base_mean": float(np.mean(base)),
            "auc_gain": float(np.mean(full) - np.mean(base)), "n_folds": len(full)}


def main():
    with Pool(7) as pool:
        res = pool.map(losses_in_match, list(idsse.MATCHES))
    d = np.array([r for rs, _ in res for r in rs], dtype=object)
    out = {"n_losses": len(d), "per_match": {mid: int((d[:, 0] == mid).sum()) for mid in idsse.MATCHES},
           "base_rate_Y1": float(d[:, 3].astype(int).mean()), "base_rate_Y2": float(d[:, 4].astype(int).mean()),
           "share_loss_already_in_zone": float(d[:, 5].astype(int).mean()),
           "possession_runs_shorter_than_5_frames": int(sum(s for _, s in res)),
           "S_mean": float(d[:, 1].astype(float).mean()), "S_sd": float(d[:, 1].astype(float).std()),
           "corr_S_ballx": float(np.corrcoef(d[:, 1].astype(float), d[:, 2].astype(float))[0, 1])}
    out["Y1"] = analyse(d, 3, np.random.default_rng(SEED))
    out["Y2"] = analyse(d, 4, np.random.default_rng(SEED + 1))
    out["Y1_exploratory_excl_loss_already_in_zone"] = analyse(d[d[:, 5].astype(int) == 0], 3, np.random.default_rng(SEED + 2))
    json.dump(out, open(idsse.ROOT / "results_b.json", "w"), indent=1)
    np.save(idsse.ROOT / "b_losses.npy", d[:, 1:].astype(float))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
