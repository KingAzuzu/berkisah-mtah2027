"""Score v12 phase TTA vs phase 0 only, with the v8 decoder (0.5*TCN + 0.5*eR, Viterbi 20/5, hook 0.6). python -m possession.v12_eval LABELS
Phase 0 here = freshly retrained nets (should reproduce logs/v8 m2_s*). Gate (v8 labels): TTA merged mean >= 0.7663 AND >= 4/5 wins vs phase 0."""
import sys, json
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .v8_convention import merge
from .data import TUNE
from .score import score

LAB = sys.argv[1]
c = joblib.load(E.CACHE)
D = E.LOG / "v12"


def shifted(a, k, n):
    u = V.upsample(a, n - k)
    return np.concatenate([np.repeat(u[:1], k, axis=0), u]) if k else u


def preds(phases):
    out = {}
    for m in TUNE:
        fs = [np.load(D / f"{LAB}_s{s}_{m}.npz") for s in range(20, 25)]
        p = []
        for i in range(2):
            n = len(c[m][i]["truth"])
            lp = np.mean([shifted(f[f"lp{i}_k{k}"], k, n) for f in fs for k in phases], 0)
            b = np.mean([shifted(f[f"b{i}_k{k}"][:, None], k, n)[:, 0] for f in fs for k in phases], 0)
            p.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
        out[m] = p
    return out


res = {}
P = {"phase0": preds([0]), "TTA5": preds([0, 1, 2, 3, 4])}
for tn in ("merged", "dfl"):
    for name, pr in P.items():
        s, cnt = [], []
        for m in TUNE:
            hv = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]]
            r = score(hv, pr[m]); s.append(r["spell_f1_mean"]); cnt.append(int(r["n_pred_spells"]))
        res[f"{name} vs {tn}"] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
        print(f"{name:7s} vs {tn:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt)
a, r0 = np.array(res["TTA5 vs merged"]["per_match"]), np.array(res["phase0 vs merged"]["per_match"])
res["gate"] = {"wins": int((a > r0).sum()), "pass": bool(a.mean() >= 0.7663 and (a > r0).sum() >= 4)}
print("gate", res["gate"])
json.dump(res, open(D / f"eval_{LAB}.json", "w"), indent=1)
