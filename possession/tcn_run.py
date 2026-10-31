"""LOMO for candidate B (TCN) vs R. python -m possession.tcn_run"""
import json, sys, time
import numpy as np
import torch
from . import m2_lib as L, pipeline as pl, integrate_run as IR, tcn_lib as T
from .data import TUNE
from .score import score

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
data, P = L.data2(), L.prep2()
mods, keep = L.r_models(P)
R = L.r_preds(P, mods, keep)
import joblib
from .data import idsse
_fp = idsse.DATA / "cache" / "tcn_feats_matched2.pkl"
if _fp.exists():
    feats = joblib.load(_fp)
else:
    feats = {m: [T.frame_features(h) for h in data[m]] for m in TUNE}
    joblib.dump(feats, _fp)
labs = {m: [T.labels5(h) for h in data[m]] for m in TUNE}
res = {"alone": {}, "fusion": {}, "R": json.load(open(pl.HERE / "logs" / "m2_R.json"))["per_match"]}
for m in TUNE:
    t0 = time.time()
    tr = [k for k in TUNE if k != m]
    mod = T.train([f for k in tr for f in feats[k]], [l for k in tr for l in labs[k]], steps=STEPS)
    lp = [T.predict_logp(mod, feats[m][i], len(data[m][i]["truth"])) for i in range(2)]
    lp2 = [T.predict_logp(mod, feats[m][i], len(data[m][i]["truth"])) for i in range(2)]
    assert all(np.array_equal(a, b) for a, b in zip(lp, lp2)), "CPU inference not identical"
    pds = R[m][0]
    eR = [pl.combine(P[m][i]["team"], pds[i], L.PARAMS["ALPHA"]) for i in range(2)]
    for name, es in (("alone", lp), ("fusion", [0.5 * a + 0.5 * b for a, b in zip(lp, eR)])):
        pred = [pl.decode(e, L.PARAMS) for e in es]
        r = score(data[m], pred)
        res[name][m] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"]}
    print(m, f"{time.time()-t0:.0f}s", {k: round(res[k][m]["spell"], 4) for k in ("alone", "fusion")},
          "R", round(res["R"][m], 4), flush=True)
    if m == TUNE[0]:
        n = sum(p.numel() for p in mod["net"].parameters()); print("params", n)
        res["n_params"] = n
for k in ("alone", "fusion"):
    s = np.array([res[k][m]["spell"] for m in TUNE]); mf = np.mean([res[k][m]["macro"] for m in TUNE])
    w = sum(res[k][m]["spell"] > res["R"][m] for m in TUNE)
    res[k]["summary"] = {"spell_mean": s.mean(), "min": s.min(), "max": s.max(), "macro_mean": mf, "wins": int(w)}
    print(k, res[k]["summary"])
Rm = np.mean(list(res["R"].values())); print("R mean", Rm)
json.dump(res, open(pl.HERE / "logs" / f"tcn_lomo_{STEPS}.json", "w"), indent=1, default=float)
