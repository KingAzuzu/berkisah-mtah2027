"""Score v7 step 3 (short-spell factor F, seeds 20-24) with the v6 decoder (fusion 0.5, Viterbi 20/5, hook 0.6), LOMO, IDSSE TUNE only.
python -m possession.v7_short_eval F -> logs/v7/short_F.json. Gate: mean >= 0.6960 AND >= 4/5 wins vs v6."""
import sys, json
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .data import TUNE
from .score import score

F = int(sys.argv[1])
c = joblib.load(E.CACHE)
D = E.LOG / "v7"
v6 = np.array(json.load(open(D / "dec.json"))["v4"])   # reference point = v6 decoder on v6 nets
res, mac, cnt = [], [], []
for m in TUNE:
    fs = [np.load(D / f"sF{F}_s{s}_{m}.npz") for s in range(20, 25)]
    pred = []
    for i in range(2):
        n = len(c[m][i]["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        pred.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
    r = score(c[m], pred)
    res.append(r["spell_f1_mean"]); mac.append(r["macro_f1"]); cnt.append((int(r["n_pred_spells"]), int(r["n_gt_spells"])))
res = np.array(res)
out = {"factor": F, "per_match": res.tolist(), "mean": float(res.mean()), "min": float(res.min()), "max": float(res.max()),
       "macro_mean": float(np.mean(mac)), "wins": int((res > v6).sum()), "v6": v6.tolist(), "counts": cnt,
       "pass": bool(res.mean() >= 0.6960 and (res > v6).sum() >= 4)}
json.dump(out, open(D / f"short_{F}.json", "w"), indent=1)
print(out)
