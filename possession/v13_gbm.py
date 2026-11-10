"""v13 B1: HistGB second model on pooled features + temporal context, LOMO on TUNE, fused 0.7*TCN(v8)+0.3*GBM then v8 decoder.
python -m possession.v13_gbm -> logs/v13/gbm_{match}.npz, score_B1.json"""
import os, json, gc
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from . import m2_lib as L, tcn_lib as T, v4_lib as V, v7_decoder as E
from .data import TUNE, idsse
from .v8_convention import merge
from .score import score

OUT = os.path.join(os.path.dirname(__file__), "logs", "v13")
PROG = os.path.join(OUT, "progress.txt")
MAXR, W = 400_000, 0.3


def ctx(X):
    X = X.astype(np.float32)
    col = np.nanmean(X, 0); col = np.where(np.isnan(col), 0, col)
    X = np.where(np.isnan(X), col, X)
    n = len(X); cs = np.vstack([np.zeros((1, X.shape[1]), np.float64), np.cumsum(X, 0, dtype=np.float64)])
    def win(a, b):  # mean over [t+a, t+b] clipped
        lo = np.clip(np.arange(n) + a, 0, n); hi = np.clip(np.arange(n) + b + 1, 0, n)
        return ((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)[:, None]).astype(np.float32)
    f = [X] + [win(-k, k) for k in (2, 10, 30)]
    f.append(win(0, 10) - win(-10, 0))
    return np.concatenate(f, 1)


def main():
    data = L.data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    labs = {m: [T.labels5(dict(h, truth=merge(h["truth"], 2))) for h in data[m]] for m in TUNE}
    cache = {}
    for m in TUNE:
        Xs, ys = [], []
        for i in range(2):
            for v in range(-1, len(dr[m])):
                f = canon[m][i] if v < 0 else dr[m][v][i]
                Xs.append(ctx(f)); ys.append(labs[m][i])
        cache[m] = (np.concatenate(Xs), np.concatenate(ys))
    for k, m in enumerate(TUNE):
        fn = os.path.join(OUT, f"gbm_{m}.npz")
        if os.path.exists(fn):
            continue
        X = np.concatenate([cache[o][0] for o in TUNE if o != m]); y = np.concatenate([cache[o][1] for o in TUNE if o != m])
        ok = y >= 0; X, y = X[ok], y[ok]
        if len(y) > MAXR:
            idx = np.random.default_rng(0).choice(len(y), MAXR, replace=False); X, y = X[idx], y[idx]
        clf = HGB(max_depth=6, max_iter=300, learning_rate=0.08, early_stopping=False, random_state=0).fit(X, y)
        res = {}
        for i in range(2):
            p = clf.predict_proba(ctx(canon[m][i]))
            lp = np.full((len(p), 3), -20.0); lp[:, clf.classes_] = np.log(np.clip(p, 1e-6, 1))
            res[f"lp{i}"] = lp
        np.savez(fn, **res)
        del X, y, clf; gc.collect()
        open(PROG, "w").write(f"B1 GBM fold {k+1}/5\n")
    c = joblib.load(E.CACHE)
    out = {}
    for name, w in (("v8", 0.0), ("fused", W), ("gbm_alone", 1.0)):
        for tn in ("merged", "dfl"):
            s = []
            for m in TUNE:
                g = np.load(os.path.join(OUT, f"gbm_{m}.npz"))
                fs = [np.load(E.LOG / "v8" / f"m2_s{sd}_{m}.npz") for sd in range(20, 25)]
                P = []
                for i in range(2):
                    n = len(c[m][i]["truth"])
                    lt = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
                    b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
                    lg = V.upsample(g[f"lp{i}"], n)
                    lp = (1 - w) * lt + w * lg
                    P.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
                hv = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]]
                s.append(float(score(hv, P)["spell_f1_mean"]))
            out[f"{name} vs {tn}"] = {"mean": float(np.mean(s)), "per_match": s}
    a, r = np.array(out["fused vs merged"]["per_match"]), np.array(out["v8 vs merged"]["per_match"])
    out["gate"] = {"wins": int((a > r).sum()), "pass": bool(a.mean() >= 0.7663 and (a > r).sum() >= 4)}
    json.dump(out, open(os.path.join(OUT, "score_B1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    open(PROG, "w").write(f"B1 scored {a.mean():.4f} {'PASS' if out['gate']['pass'] else 'FAIL'}\n")


if __name__ == "__main__":
    main()
