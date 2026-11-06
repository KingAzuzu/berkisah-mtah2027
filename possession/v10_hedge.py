"""v10 hedge (CPU, decision aid, no gate): v6, v8, merge variants and the v6/v8 log-prob average, scored vs DFL and merged(K2) truth.
LOMO log-probs on IDSSE TUNE. python -m possession.v10_hedge -> logs/v10/hedge.json"""
import json
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .v8_convention import merge
from .data import TUNE
from .score import score

c = joblib.load(E.CACHE)


def nets(D, pref, seeds=range(20, 25)):
    """{m: [(lp, b) per half]} mean over seeds, upsampled."""
    out = {}
    for m in TUNE:
        fs = [np.load(D / f"{pref}_s{s}_{m}.npz") for s in seeds]
        h = []
        for i in range(2):
            n = len(c[m][i]["truth"])
            h.append((V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n),
                      V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]))
        out[m] = h
    return out


def evid(N, m, i):
    return 0.5 * N[m][i][0] + 0.5 * c[m][i]["eR"]


def decode(N):
    return {m: [V.viterbi_scaled(evid(N, m, i), 20.0, 5.0, 1 - 0.6 * N[m][i][1]) for i in range(2)] for m in TUNE}


def score_vs(P, tn, mg=0):
    s, cnt = [], []
    for m in TUNE:
        halves = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]]
        r = score(halves, [merge(x, mg) if mg else x for x in P[m]])
        s.append(float(r["spell_f1_mean"])); cnt.append(int(r["n_pred_spells"]))
    return s, cnt


if __name__ == "__main__":
    N6, N8 = nets(E.LOG / "v6", "e2"), nets(E.LOG / "v8", "m2")
    NA = {m: [((N6[m][i][0] + N8[m][i][0]) / 2, (N6[m][i][1] + N8[m][i][1]) / 2) for i in range(2)] for m in TUNE}
    P6, P8, PA = decode(N6), decode(N8), decode(NA)
    cands = {"v6": (P6, 0), "v8": (P8, 0), "v6+merge1": (P6, 1), "v6+merge2": (P6, 2),
             "avg(v6,v8)": (PA, 0), "avg+merge1": (PA, 1)}
    res = {}
    for name, (P, mg) in cands.items():
        res[name] = {}
        for tn in ("dfl", "merged"):
            s, cnt = score_vs(P, tn, mg)
            res[name][tn] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
            print(f"{name:11s} vs {tn:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt, flush=True)
    best = {tn: max(res[n][tn]["mean"] for n in res) for tn in ("dfl", "merged")}
    for n in res:
        res[n]["regret"] = {tn: best[tn] - res[n][tn]["mean"] for tn in best}
        res[n]["max_regret"] = max(res[n]["regret"].values())
        print(f"{n:11s} regret dfl {res[n]['regret']['dfl']:.4f} merged {res[n]['regret']['merged']:.4f} max {res[n]['max_regret']:.4f}")
    mm = min(res, key=lambda n: res[n]["max_regret"])
    res["_minimax"] = mm
    print("minimax-regret:", mm)
    json.dump(res, open(E.LOG / "v10" / "hedge.json", "w"), indent=1)
    open(E.LOG / "v10" / "progress_B.txt", "a").write("")
