"""v7 D: decoder retune on saved v6 LOMO log-probs (E2 nets, seeds 20-24), nested LOMO, wider grid than v6 E1
(E1 picks sat at the grid edges). IDSSE TUNE only, CPU, no retraining.
Gate fixed before running: nested mean >= 0.6960 (v6 0.6860 + 0.01) AND beats v6 on >= 4/5 matches.
python -m possession.v7_decoder -> logs/v7/dec.json"""
import json, time, itertools
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import joblib

from .data import TUNE, idsse

LOG = Path(__file__).parent / "logs"
CACHE = idsse.DATA / "cache" / "v7_comp.pkl"
ST, SD, HK, FW = (4, 6, 8, 12, 16, 20), (1, 2, 3, 5, 8), (0.6, 0.8, 0.9, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9)
GRID = list(itertools.product(ST, SD, HK, FW))
V4 = (20, 5, 0.6, 0.5)
C = None


def build():
    from . import m2_lib as L, pipeline as pl, v4_lib as V
    data, P = L.data2(), L.prep2()
    mods, keep = L.r_models(P)
    R = L.r_preds(P, mods, keep)
    D = pl.HERE / "logs" / "v6"
    out = {}
    for m in TUNE:
        fs = [np.load(D / f"e2_s{s}_{m}.npz") for s in range(20, 25)]
        hs = []
        for i in range(2):
            n = len(data[m][i]["truth"])
            hs.append({"lp": V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n),
                       "eR": pl.combine(P[m][i]["team"], R[m][0][i], L.PARAMS["ALPHA"]),
                       "b": V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0],
                       "truth": data[m][i]["truth"]})
        out[m] = hs
    joblib.dump(out, CACHE)
    return out


def init(c):
    global C
    C = c


def job(a):
    from . import v4_lib as V
    from .score import score
    gi, m = a
    st, sd, hk, fw = GRID[gi]
    pred = []
    for h in C[m]:
        e = fw * h["lp"] + (1 - fw) * h["eR"]
        pred.append(V.viterbi_scaled(e, st, sd, 1 - hk * h["b"]))
    r = score(C[m], pred)
    return gi, m, r["spell_f1_mean"], r["macro_f1"], int(r["spells"]["n_pred"]) if "spells" in r else None, r


def main():
    t0 = time.time()
    c = joblib.load(CACHE) if CACHE.exists() else build()
    jobs = [(g, m) for g in range(len(GRID)) for m in TUNE]
    res = {}
    with Pool(10, initializer=init, initargs=(c,)) as pool:
        for k, (g, m, s, mac, npred, r) in enumerate(pool.imap_unordered(job, jobs)):
            if k == 0:
                print("result keys", list(r.keys()), flush=True)
            res[(g, m)] = (s, mac, r)
            el = time.time() - t0
            pct = int(100 * (k + 1) / len(jobs))
            (LOG / "v7_dec_progress.txt").write_text(f"{pct}% | v7 decoder grid {k+1}/{len(jobs)} | ETA {int(el/(k+1)*(len(jobs)-k-1)/60)}min\n")
    S = np.array([[res[(g, m)][0] for m in TUNE] for g in range(len(GRID))])   # (grid, match)
    vi = GRID.index(V4)
    # spell counts: find n_pred / n_gt in the evaluate dict
    def counts(g, m):
        r = res[(g, m)][2]
        sp = r.get("spell") or r.get("spells") or {}
        return sp.get("n_pred"), sp.get("n_gt")
    nested, sel = [], []
    for j, m in enumerate(TUNE):
        others = [k for k in range(len(TUNE)) if k != j]
        g = int(np.argmax(S[:, others].mean(1)))
        nested.append(S[g, j]); sel.append(g)
    nested = np.array(nested)
    v4 = S[vi]
    out = {"v4_grid_point": V4, "v4": v4.tolist(), "v4_mean": float(v4.mean()), "nested": nested.tolist(), "nested_mean": float(nested.mean()),
           "nested_min": float(nested.min()), "nested_max": float(nested.max()), "wins": int((nested > v4).sum()),
           "selected": [GRID[g] for g in sel], "matches": list(TUNE),
           "counts_v4": [counts(vi, m) for m in TUNE], "counts_nested": [counts(g, m) for g, m in zip(sel, TUNE)],
           "best_global": {"point": GRID[int(np.argmax(S.mean(1)))], "mean": float(S.mean(1).max())},
           "grid_mean": {str(GRID[g]): float(S[g].mean()) for g in range(len(GRID))},
           "grid_per_match": {str(GRID[g]): S[g].tolist() for g in range(len(GRID))}}
    (LOG / "v7").mkdir(exist_ok=True)
    json.dump(out, open(LOG / "v7" / "dec.json", "w"), indent=1)
    print("v6 %.4f nested %.4f [%.4f,%.4f] wins %d/5" % (v4.mean(), nested.mean(), nested.min(), nested.max(), out["wins"]))
    print("selected", out["selected"]); print("counts v4", out["counts_v4"]); print("counts nested", out["counts_nested"])
    print("best global", out["best_global"])


if __name__ == "__main__":
    main()
