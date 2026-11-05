"""Score E2 (short-spell x3, seeds 20-24, cB) with the v4 decoder on the LOMO folds. IDSSE TUNE only."""
import json
import numpy as np, joblib
from . import v6_e1 as E, v4_lib as V
from .data import TUNE
from .score import score
c = joblib.load(E.CACHE)
D = E.LOG / "v6"
o = json.load(open(D / "e1.json"))
v4 = np.array(o["v4"])
res, cnt = [], []
for m in TUNE:
    fs = [np.load(D / f"c_s{s}_{m}.npz") for s in range(20, 25)]
    pred = []
    for i in range(2):
        n = len(c[m][i]["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        pred.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
    r = score(c[m], pred)
    res.append(r["spell_f1_mean"]); cnt.append((r["n_pred_spells"], r["n_gt_spells"]))
res = np.array(res)
out = {"c": res.tolist(), "mean": float(res.mean()), "min": float(res.min()), "max": float(res.max()), "wins": int((res > v4).sum()),
       "v4": v4.tolist(), "counts": cnt, "pass": bool(res.mean() >= 0.6839 and (res > v4).sum() >= 4)}
json.dump(out, open(D / "c.json", "w"), indent=1)
print(out)
