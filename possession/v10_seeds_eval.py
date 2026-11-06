"""v10 10-seed v8 (seeds 20-29) vs 5-seed v8 (20-24): v8 decoder, LOMO, merged truth (gate) and DFL (info).
Gate FIXED: mean vs merged >= 0.7613 AND >= 4/5 wins vs v8 5-seed. python -m possession.v10_seeds_eval -> logs/v10/seeds.json"""
import json
import numpy as np
from . import v7_decoder as E
from .v10_hedge import nets, decode, score_vs

N5, N10 = nets(E.LOG / "v8", "m2", range(20, 25)), nets(E.LOG / "v8", "m2", range(20, 30))
P5, P10 = decode(N5), decode(N10)
res = {}
for tn in ("merged", "dfl"):
    for n, P in (("seeds5", P5), ("seeds10", P10)):
        s, cnt = score_vs(P, tn)
        res[f"{n} {tn}"] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
        print(n, tn, f"{np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt)
a, b = np.array(res["seeds10 merged"]["per_match"]), np.array(res["seeds5 merged"]["per_match"])
res["gate"] = {"wins": int((a > b).sum()), "pass": bool(a.mean() >= 0.7613 and (a > b).sum() >= 4)}
print(res["gate"])
json.dump(res, open(E.LOG / "v10" / "seeds.json", "w"), indent=1)
