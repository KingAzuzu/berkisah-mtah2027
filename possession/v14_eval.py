"""Score C3 (aug nets) vs v8 with the v8 decoder, merged + DFL truth. Gate: merged mean >= 0.7663 AND beats v8 on >= 4/5. -> logs/v14/score_C3.json"""
import json
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .v8_convention import merge
from .data import TUNE
from .score import score

c = joblib.load(E.CACHE)


def dec(pref, D):
    out = {}
    for m in TUNE:
        fs = [np.load(D / f"{pref}_s{s}_{m}.npz") for s in range(20, 25)]
        p = []
        for i in range(2):
            n = len(c[m][i]["truth"])
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
            p.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
        out[m] = p
    return out


P = {"v8": dec("m2", E.LOG / "v8"), "C3": dec("aug", E.LOG / "v14")}
res = {}
for tn in ("merged", "dfl"):
    for name, Pp in P.items():
        s = []
        for m in TUNE:
            halves = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]]
            s.append(score(halves, Pp[m])["spell_f1_mean"])
        res[f"{name} vs {tn}"] = {"mean": float(np.mean(s)), "per_match": s}
        print(f"{name:3s} vs {tn:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s))
a, r = np.array(res["C3 vs merged"]["per_match"]), np.array(res["v8 vs merged"]["per_match"])
res["gate"] = {"wins": int((a > r).sum()), "pass": bool(a.mean() >= 0.7663 and (a > r).sum() >= 4)}
print("gate", res["gate"])
json.dump(res, open(E.LOG / "v14" / "score_C3.json", "w"), indent=1)
open(E.LOG / "v14" / "progress.txt", "w").write(f"scored {a.mean():.4f} {'PASS' if res['gate']['pass'] else 'FAIL'}\n")
