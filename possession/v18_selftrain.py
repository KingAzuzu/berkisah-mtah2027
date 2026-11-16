"""v18: transductive self-training on the unlabeled held-out match (RULES.md allows algorithmic pseudo-labels on the provided game).

Student = v8 recipe (cB, SHORT_W 3, 1500 steps, seeds 20-24), trained on the 4 labelled TUNE matches (merged K2 labels, canon + DR)
PLUS the held-out match m with pseudo-labels = v8 LOMO Viterbi decode for m (nets that never saw m), canonical features only
(on BAR-ATH there is one version of the data). Same seeds as v8, so the difference is the pseudo-data.
Gate fixed beforehand (vs merged truth): mean >= 0.7663 (v8 0.7563 + 0.01) AND beats v8 on >= 4/5 matches. DFL info only.
python -m possession.v18_selftrain train SEED   -> logs/v18/st_s{SEED}_{MATCH}.npz
python -m possession.v18_selftrain eval         -> logs/v18/eval.json"""
import os
import sys
import time
import json
import numpy as np
import joblib

from . import m2_lib as L, tcn_lib as T, v4_lib as V, v7_decoder as E
from .data import TUNE, idsse
from .v8_convention import merge

OUT = E.LOG / "v18"


def pseudo(c, m):
    fs = [np.load(E.LOG / "v8" / f"m2_s{s}_{m}.npz") for s in range(20, 25)]
    out = []
    for i in range(2):
        n = len(c[m][i]["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        out.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
    return out


def train(seed):
    os.makedirs(OUT, exist_ok=True)
    data, c = L.data2(), joblib.load(E.CACHE)
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    labs = {m: [T.labels5(dict(h, truth=merge(h["truth"], 2))) for h in data[m]] for m in TUNE}
    for m in TUNE:
        f = OUT / f"st_s{seed}_{m}.npz"
        if f.exists():
            continue
        t0 = time.time()
        items = [{"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))], "y": labs[k][i]}
                 for k in TUNE if k != m for i in range(2)]
        ps = pseudo(c, m)
        items += [{"src": "idsse", "feats": [canon[m][i]], "y": T.labels5(dict(data[m][i], truth=ps[i]))} for i in range(2)]
        mod = V.train4(items, seed=seed, steps=1500, short_w=3.0)
        res = {}
        for i in range(2):
            res[f"lp{i}"], res[f"b{i}"] = V.predict4(mod, canon[m][i])
        np.savez(f, **res)
        print(seed, m, f"{time.time() - t0:.0f}s", flush=True)


def evaluate():
    from .v10_hedge import nets, score_vs, decode
    P8, PS = decode(nets(E.LOG / "v8", "m2")), decode(nets(OUT, "st"))
    res = {}
    for name, P in (("v8", P8), ("v18", PS)):
        res[name] = {}
        for tn in ("merged", "dfl"):
            s, cnt = score_vs(P, tn)
            res[name][tn] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
            print(f"{name:4s} {tn:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt, flush=True)
    wins = sum(a > b for a, b in zip(res["v18"]["merged"]["per_match"], res["v8"]["merged"]["per_match"]))
    res["gate"] = bool(res["v18"]["merged"]["mean"] >= 0.7663 and wins >= 4)
    print(f"wins {wins}/5 gate {'PASS' if res['gate'] else 'FAIL'}")
    json.dump(res, open(OUT / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    train(int(sys.argv[2])) if sys.argv[1] == "train" else evaluate()
