"""v7 ceiling diagnostic: v6 recipe (cB, SHORT_W 3, 1500 steps) trained+tested on CLEAN IDSSE (no degradation, no DR), LOMO over TUNE.
python -m possession.v7_ceiling feats | train SEED | score. IDSSE TUNE only. Nothing frozen."""
import sys, time, os, json
from pathlib import Path
import numpy as np
import joblib

from . import tcn_lib as T, v4_lib as V
from .data import TUNE, idsse, load_idsse
from .score import score

OUT = Path(__file__).parent / "logs" / "v7"
CF = idsse.DATA / "cache" / "tcn_feats_clean.pkl"
SEEDS = (20, 21)


def clean():
    return {m: load_idsse(m) for m in TUNE}


def feats():
    d = clean()
    joblib.dump({m: [T.frame_features(h) for h in d[m]] for m in TUNE}, CF)


def train(seed):
    d, cf = clean(), joblib.load(CF)
    labs = {m: [T.labels5(h) for h in d[m]] for m in TUNE}
    OUT.mkdir(exist_ok=True)
    for m in TUNE:
        f = OUT / f"ceil_s{seed}_{m}.npz"
        if f.exists():
            continue
        t0 = time.time()
        items = [{"src": "idsse", "feats": [cf[k][i]], "y": labs[k][i]} for k in TUNE if k != m for i in range(2)]
        mod = V.train4(items, seed=seed, steps=1500, short_w=3.0)
        res = {}
        for i in range(2):
            res[f"lp{i}"], res[f"b{i}"] = V.predict4(mod, cf[m][i])
        np.savez(f, **res)
        print(seed, m, f"{time.time()-t0:.0f}s", flush=True)


def decode(files, d, m):
    fs = [np.load(f) for f in files]
    halves, pred = d[m], []
    for i, h in enumerate(halves):
        n = len(h["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        pred.append(V.viterbi_scaled(lp, 20, 5, 1 - 0.6 * b))
    return score(halves, pred), pred


def spells(c):
    return int((np.diff(np.r_[0, c]) != 0).sum() - 0) if False else None


def main():
    from . import m2_lib as L
    dc, dd = clean(), L.data2()
    log, out = OUT.parent, {}
    for name, d, pat in (("clean", dc, lambda s, m: OUT / f"ceil_s{s}_{m}.npz"), ("degraded", dd, lambda s, m: log / "v6" / f"e2_s{s}_{m}.npz")):
        out[name] = {}
        for m in TUNE:
            r, _ = decode([pat(s, m) for s in SEEDS], d, m)
            out[name][m] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                            "sp": [r["n_pred_spells"], r["n_gt_spells"]]}
            print(name, m, round(r["spell_f1_mean"], 4), round(r["macro_f1"], 4), out[name][m]["sp"], flush=True)
    for k in out:
        print(k, "mean spell %.4f macro %.4f" % (np.mean([v["spell"] for v in out[k].values()]), np.mean([v["macro"] for v in out[k].values()])))
    json.dump(out, open(OUT / "ceil_scores.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    a = sys.argv[1]
    feats() if a == "feats" else train(int(sys.argv[2])) if a == "train" else main()
