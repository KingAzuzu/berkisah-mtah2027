"""Score v8 (merged-label nets, seeds 20-24) with the v6 decoder, LOMO, IDSSE TUNE. Compares v6, v6+merge, v8, v8+merge on merged
truth (K 2) and DFL truth. Gate: v8 or v8+merge (best, vs merged) mean >= 0.7410 AND >= 4/5 wins vs v6+merge. -> logs/v8/eval.json"""
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


P6, P8 = dec("e2", E.LOG / "v6"), dec("m2", E.LOG / "v8")
cands = {"v6": (P6, False), "v6+merge": (P6, True), "v8": (P8, False), "v8+merge": (P8, True)}
res = {}
for tn in ("merged", "dfl"):
    for name, (P, mg) in cands.items():
        s, cnt = [], []
        for m in TUNE:
            halves = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]]
            r = score(halves, [merge(x, 2) if mg else x for x in P[m]])
            s.append(r["spell_f1_mean"]); cnt.append(int(r["n_pred_spells"]))
        res[f"{name} vs {tn}"] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
        print(f"{name:9s} vs {tn:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt)
ref = np.array(res["v6+merge vs merged"]["per_match"])
for k in ("v8", "v8+merge"):
    a = np.array(res[f"{k} vs merged"]["per_match"])
    res[f"{k} gate"] = {"wins": int((a > ref).sum()), "pass": bool(a.mean() >= 0.7410 and (a > ref).sum() >= 4)}
    print(k, "gate", res[f"{k} gate"])
json.dump(res, open(E.LOG / "v8" / "eval.json", "w"), indent=1)
