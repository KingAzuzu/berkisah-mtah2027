"""Candidate (A): stage-2 HistGradientBoosting (H/A/D) on stage-1 outputs + window context, then the same Viterbi.
LOMO under degrade profile matched2, TUNE only. Held-out match m: stage-1 p_dead from the model trained on the other 4;
stage-2 is trained on the other 4 matches using OUT-OF-FOLD stage-1 p_dead (model excluding both m and that match).
    python -m possession.ctx_stack
"""
import itertools
import json
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import dead, integrate_run as IR, m2_lib as L, pipeline as pl, safety
from .data import FPS, TUNE
from .score import score

WIN_S = (0.5, 2.0, 6.0)
SUB = 5


def _cm(x, w):
    """Centred window mean (nan-safe) over +-w frames; x (T,)."""
    v = np.isfinite(x)
    a = np.where(v, x, 0.0)
    c = np.r_[0, np.cumsum(a)]
    n = np.r_[0, np.cumsum(v)]
    T = len(x)
    lo, hi = np.clip(np.arange(T) - w, 0, T), np.clip(np.arange(T) + w + 1, 0, T)
    cnt = n[hi] - n[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def base_series(half):
    ball, xy, team = half["ball"], half["xy"], half["team"]
    T = len(ball)
    miss = np.isnan(ball[:, 0]).astype(float)
    cols = {"miss": miss}
    sp = np.full(xy.shape[:2], np.nan)
    sp[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) * FPS
    sp[sp > 12] = np.nan
    for k, nm in ((0, "h"), (1, "a")):
        s = team == k
        with np.errstate(all="ignore"):
            cols["cx_" + nm] = np.nanmean(xy[:, s, 0], axis=1)
            cols["sp_" + nm] = np.nanmean(sp[:, s], axis=1)
            cols["n_" + nm] = np.isfinite(xy[:, s, 0]).sum(1).astype(float)
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, axis=1)
    ok = np.isfinite(d[np.arange(T), j]) & (miss == 0)
    cols["near_a"] = np.where(ok, team[j].astype(float), np.nan)
    cols["near_d"] = np.where(ok, d[np.arange(T), j], np.nan)
    return cols


def features(half, e_rules, team_ev, p_dead):
    T = len(p_dead)
    s1 = {"ev_h": team_ev[:, 0], "ev_a": team_ev[:, 1], "ev_diff": team_ev[:, 0] - team_ev[:, 1],
          "ev_D": e_rules[:, 2], "lp": dead.logit(p_dead)}
    base = base_series(half)
    cols = [s1[k] for k in s1] + [base[k] for k in ("miss", "cx_h", "cx_a", "sp_h", "sp_a", "near_a", "near_d")]
    for w in WIN_S:
        n = int(round(w * FPS))
        for k in ("ev_diff", "ev_D", "lp"):
            cols.append(_cm(s1[k], n))
        for k in ("miss", "cx_h", "cx_a", "sp_h", "sp_a", "near_a", "near_d"):
            cols.append(_cm(base[k], n))
        # future minus past of team evidence (direction of change)
        e = s1["ev_diff"]
        cols.append(np.r_[e[n:], np.full(n, np.nan)] - np.r_[np.full(n, np.nan), e[:-n]])
        cols.append(np.r_[s1["lp"][n:], np.full(n, np.nan)] - np.r_[np.full(n, np.nan), s1["lp"][:-n]])
    return np.stack(cols, axis=1).astype(np.float32)


def build(data, P, pd_by_half):
    return [features(data[i], P[i]["e_rules"], P[i]["team"], pd_by_half[i]) for i in range(len(data))]


def fit_stage2(Xs, ys):
    X = np.concatenate([x[::SUB] for x in Xs])
    y = np.concatenate([a[::SUB] for a in ys])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6, min_samples_leaf=100,
                                          l2_regularization=1.0, random_state=0).fit(X, y)


def decode_lp(model, X, cell):
    lp = np.log(np.clip(model.predict_proba(X), 1e-6, 1))   # columns = classes 0,1,2 = H,A,D
    return pl.decode(lp, cell)


def _fold(m, data, P, F_out_m, Xtr, ytr, cells, e_r):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        mod2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6, min_samples_leaf=100,
                                              l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=15, random_state=0).fit(Xtr, ytr)
        out = {}
        for c, (cell, w) in cells.items():
            preds = []
            for X, er in zip(F_out_m, e_r):
                lp = np.log(np.clip(mod2.predict_proba(X), 1e-6, 1))
                preds.append(pl.decode(lp if w is None else er + w * lp, cell))
            out[c] = score(data, preds)["spell_f1_mean"]
    return out


def main():
    from joblib import Parallel, delayed
    data, P = L.data2(), L.prep2()
    path = L.CACHE / "ctx_stack_feats.pkl"
    if path.exists():
        F_out, TR = joblib.load(path)
    else:
        mods, keep = L.r_models(P)
        outer = {m: IR.predict_p(mods[m], P, m, keep) for m in TUNE}
        inner = {}
        for a, b in itertools.combinations(TUNE, 2):
            mod = IR.train_on(P, [k for k in TUNE if k not in (a, b)], keep)
            inner[(a, b)] = IR.predict_p(mod, P, b, keep)
            inner[(b, a)] = IR.predict_p(mod, P, a, keep)
        F_out = {m: build(data[m], P[m], outer[m]) for m in TUNE}
        TR = {}
        for m in TUNE:
            Xs, ys = [], []
            for k in TUNE:
                if k != m:
                    Xs += build(data[k], P[k], inner[(m, k)])
                    ys += [h["truth"] for h in data[k]]
            TR[m] = (np.concatenate([x[::SUB] for x in Xs]), np.concatenate([a[::SUB] for a in ys]))
        joblib.dump((F_out, TR), path)
    cells = {"R_cell": (L.PARAMS, None),
             "sw10_5": ({**L.PARAMS, "SWITCH_TEAM": 10.0, "SWITCH_DEAD": 5.0}, None),
             "sw5_2.5": ({**L.PARAMS, "SWITCH_TEAM": 5.0, "SWITCH_DEAD": 2.5}, None),
             "blend0.5_R_cell": (L.PARAMS, 0.5), "blend1_R_cell": (L.PARAMS, 1.0)}
    e_rs = {m: [pl.combine(h["team"], pd, L.PARAMS["ALPHA"]) for h, pd in zip(P[m], IR.predict_p(
        IR.train_on(P, [k for k in TUNE if k != m], pl.keep_columns(P[TUNE[0]][0]["names"], "ball")), P, m,
        pl.keep_columns(P[TUNE[0]][0]["names"], "ball")))] for m in TUNE}
    outs = Parallel(n_jobs=5)(delayed(_fold)(m, data[m], P[m], F_out[m], TR[m][0], TR[m][1], cells, e_rs[m]) for m in TUNE)
    res = {c: dict(zip(TUNE, [o[c] for o in outs])) for c in cells}
    Rm = json.load(open(pl.HERE / "logs" / "m2_R.json"))["per_match"]
    summ = {}
    for c in cells:
        v = [res[c][m] for m in TUNE]
        summ[c] = {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "per_match": dict(zip(TUNE, v)),
                   "wins": int(sum(res[c][m] > Rm[m] for m in TUNE))}
        print(c, "mean %.4f [%.4f, %.4f] wins %d/5" % (summ[c]["mean"], min(v), max(v), summ[c]["wins"]))
    json.dump(summ, open(pl.HERE / "logs" / "ctx_stack.json", "w"), indent=1)


if __name__ == "__main__":
    main()
