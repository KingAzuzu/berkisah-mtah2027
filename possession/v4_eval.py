"""Score v4 LOMO folds. python -m possession.v4_eval -> logs/v4_eval.json and a table.
Per config: mean over the first N seeds of log-probs (+ boundary probs), fused 50/50 with R (v2.1 evidence), Viterbi
(plain, and with the boundary hook). v3 reference per match = 3-seed ensemble in logs/tcn_v3_seeds.json."""
import json, sys
from pathlib import Path
import numpy as np

from . import m2_lib as L, pipeline as pl, v4_lib as V
from .data import TUNE
from .score import score

D = pl.HERE / "logs" / "v4"
data, P = L.data2(), L.prep2()
mods, keep = L.r_models(P)
R = L.r_preds(P, mods, keep)
eR = {m: [pl.combine(P[m][i]["team"], R[m][0][i], L.PARAMS["ALPHA"]) for i in range(2)] for m in TUNE}
p = pl.base_params(L.PARAMS)
v3 = json.load(open(pl.HERE / "logs" / "tcn_v3_seeds.json"))["ens"]


def run(cfg, seeds, hook):
    per = {}
    for m in TUNE:
        fs = [np.load(D / f"{cfg}_s{s}_{m}.npz") for s in seeds]
        pred = []
        for i in range(2):
            n = len(data[m][i]["truth"])
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            e = 0.5 * lp + 0.5 * eR[m][i]
            if hook:
                b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
                pred.append(V.viterbi_scaled(e / p.get("TEMP", 1.0), p["SWITCH_TEAM"] / p.get("TEMP", 1.0),
                                             p["SWITCH_DEAD"] / p.get("TEMP", 1.0), 1 - V.HOOK * b))
            else:
                pred.append(pl.decode(e, L.PARAMS))
        r = score(data[m], pred)
        per[m] = (r["spell_f1_mean"], r["macro_f1"])
    return per


def row(name, per):
    s = np.array([per[m][0] for m in TUNE])
    w = sum(per[m][0] > v3[m]["spell"] for m in TUNE)
    return {"name": name, "mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()), "wins_vs_v3": int(w),
            "macro": float(np.mean([per[m][1] for m in TUNE])), "per_match": {m: float(per[m][0]) for m in TUNE}}


if __name__ == "__main__":
    out = [row("v3 (ref, 3 seeds)", {m: (v3[m]["spell"], v3[m]["macro"]) for m in TUNE})]
    plan = [("full", (0, 1, 2)), ("nomet", (0, 1, 2)), ("nosc", (0, 1, 2)), ("nobd", (0, 1, 2)), ("full", (0, 1, 2, 3, 4)), ("nosc", (0, 1, 2, 3, 4))]
    for cfg, seeds in plan:
        if not all((D / f"{cfg}_s{s}_{m}.npz").exists() for s in seeds for m in TUNE):
            continue
        for hook in ((False, True) if cfg != "nobd" else (False,)):
            out.append(row(f"{cfg} x{len(seeds)} seeds{' +hook' if hook else ''}", run(cfg, seeds, hook)))
            r = out[-1]
            print(f"{r['name']:28s} {r['mean']:.4f} [{r['min']:.4f},{r['max']:.4f}] wins {r['wins_vs_v3']}/5 macro {r['macro']:.4f} "
                  + " ".join(f"{r['per_match'][m]:.3f}" for m in TUNE), flush=True)
    json.dump(out, open(pl.HERE / "logs" / "v4_eval.json", "w"), indent=1)
