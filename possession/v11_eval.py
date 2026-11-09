"""Score v11 variants (python -m possession.v11_eval V1|V2|D) vs merged truth (K 2, primary) and DFL truth, LOMO, seeds 20-24.
Gate fixed: mean vs merged >= 0.7663 AND beats v8 per-match list on >= 4/5. -> logs/v11/score_{VAR}.json"""
import sys, json
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .v8_convention import merge
from .data import TUNE
from .score import score

V8 = [.763, .719, .786, .734, .779]
VAR = sys.argv[1]
c = joblib.load(E.CACHE)
out = {}
P = {}
for m in TUNE:
    fs = [np.load(E.LOG / "v11" / f"{VAR}_s{s}_{m}.npz") for s in range(20, 25)]
    p = []
    for i in range(2):
        n = len(c[m][i]["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        p.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
    P[m] = p
for tn in ("merged", "dfl"):
    s = []
    for m in TUNE:
        halves = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]]
        s.append(score(halves, P[m])["spell_f1_mean"])
    out[tn] = {"mean": float(np.mean(s)), "per_match": s}
    print(VAR, tn, f"{np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s))
a = np.array(out["merged"]["per_match"]); wins = int((a > np.array(V8)).sum())
out["gate"] = {"wins_vs_v8": wins, "mean_ok": bool(a.mean() >= 0.7663), "pass": bool(a.mean() >= 0.7663 and wins >= 4)}
print(out["gate"])
json.dump(out, open(E.LOG / "v11" / f"score_{VAR}.json", "w"), indent=1)
