"""Seed check + ensemble for v3 (TCN + v2.1 fusion), LOMO on TUNE, matched2 degrade. python -m possession.tcn_v3_seeds"""
import json, time
import numpy as np
import joblib
from . import m2_lib as L, pipeline as pl, tcn_lib as T
from .data import TUNE, idsse
from .score import score

SEEDS = (0, 1, 2)
STEPS = 1500
data, P = L.data2(), L.prep2()
mods, keep = L.r_models(P)
R = L.r_preds(P, mods, keep)
feats = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
labs = {m: [T.labels5(h) for h in data[m]] for m in TUNE}
res = {f"s{s}": {} for s in SEEDS}; res["ens"] = {}; res["R"] = json.load(open(pl.HERE / "logs" / "m2_R.json"))["per_match"]
for m in TUNE:
    t0 = time.time()
    tr = [k for k in TUNE if k != m]
    lps = []
    for s in SEEDS:
        mod = T.train([f for k in tr for f in feats[k]], [l for k in tr for l in labs[k]], seed=s, steps=STEPS)
        lps.append([T.predict_logp(mod, feats[m][i], len(data[m][i]["truth"])) for i in range(2)])
    eR = [pl.combine(P[m][i]["team"], R[m][0][i], L.PARAMS["ALPHA"]) for i in range(2)]
    cands = {f"s{s}": lps[j] for j, s in enumerate(SEEDS)}
    cands["ens"] = [np.mean([l[i] for l in lps], 0) for i in range(2)]
    for name, lp in cands.items():
        pred = [pl.decode(0.5 * a + 0.5 * b, L.PARAMS) for a, b in zip(lp, eR)]
        r = score(data[m], pred)
        res[name][m] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"]}
    print(m, f"{time.time()-t0:.0f}s", {k: round(res[k][m]["spell"], 4) for k in cands}, flush=True)
for k in list(res)[:-1] if False else [f"s{s}" for s in SEEDS] + ["ens"]:
    s = np.array([res[k][m]["spell"] for m in TUNE])
    res[k]["summary"] = {"mean": s.mean(), "min": s.min(), "max": s.max(), "macro": np.mean([res[k][m]["macro"] for m in TUNE])}
    print(k, res[k]["summary"])
json.dump(res, open(pl.HERE / "logs" / "tcn_v3_seeds.json", "w"), indent=1, default=float)
