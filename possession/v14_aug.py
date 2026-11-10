"""C3: stronger augmentation of TRAINING halves only. python -m possession.v14_aug prep | smoke | train SEED
4 extra variants per training half = canonical / DR degrade j (cycle) + stretch s~U(.9,1.1) + 0.30 m xy noise + 10 s player dropout p .10.
Stretched variants have another length, so each is its own item (same weight rule as other items). Outputs logs/v14/aug_s{seed}_{match}.npz."""
import sys, os, time, gc
import numpy as np
import joblib

from . import tcn_lib as T, v4_lib as V, degrade_matched2 as DM2
from .v4_prep import degrade_variant
from .v8_convention import merge
from .data import TUNE, idsse, load_idsse

NV, SIG, SEG = 4, 0.30, 250
CACHE = idsse.DATA / "cache" / "tcn_feats_aug.pkl"
OUT = os.path.join(os.path.dirname(__file__), "logs", "v14")
PROG = os.path.join(OUT, "progress.txt")


def prog(s):
    os.makedirs(OUT, exist_ok=True)
    open(PROG, "w").write(s + "\n")
    print(s, flush=True)


def interp(a, u):
    i0 = np.floor(u).astype(int); i1 = np.minimum(i0 + 1, len(a) - 1); w = (u - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[i0] * (1 - w) + a[i1] * w


def augment(h, rng):
    n = len(h["truth"]); s = rng.uniform(0.9, 1.1)
    u = np.arange(int((n - 1) / s) + 1) * s
    xy, ball = interp(h["xy"], u), interp(h["ball"], u)
    truth = h["truth"][np.minimum(np.rint(u).astype(int), n - 1)]
    xy = xy + rng.normal(0, SIG, xy.shape)
    for a in range(0, len(xy), SEG):
        xy[a:a + SEG, rng.random(xy.shape[1]) < 0.10] = np.nan
    return dict(h, xy=xy, ball=ball, truth=truth)


def prep():
    out = {}
    for mi, m in enumerate(TUNE):
        hv = load_idsse(m)
        out[m] = [[], []]
        for j in range(NV):
            deg = DM2.degrade_match(hv, m, 0) if j == 0 else degrade_variant(hv, m, j - 1)
            for i, h in enumerate(deg):
                a = augment(h, np.random.default_rng([1414, mi, j, i]))
                out[m][i].append((T.frame_features(a), T.labels5(dict(a, truth=merge(a["truth"], 2)))))
            del deg
        del hv; gc.collect()
        prog(f"prep match {mi+1}/5")
    joblib.dump(out, CACHE)


def fit(seed, folds, steps, tag):
    from .m2_lib import data2
    data = data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    aug = joblib.load(CACHE)
    labs = {m: [T.labels5(dict(h, truth=merge(h["truth"], 2))) for h in data[m]] for m in TUNE}
    del data; gc.collect()
    for m in folds:
        f = os.path.join(OUT, f"{tag}_s{seed}_{m}.npz")
        if os.path.exists(f):
            continue
        items = []
        for k in TUNE:
            if k == m:
                continue
            for i in range(2):
                items.append({"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))], "y": labs[k][i]})
                for ff, ll in aug[k][i]:
                    items.append({"src": "idsse", "feats": [ff], "y": ll})
        if tag == "smoke":
            for it in items:
                for ff in it["feats"]:
                    assert np.isfinite(ff).any() and ff.shape[1] == 26 and len(ff) == len(it["y"]), (ff.shape, len(it["y"]))
            print("smoke shapes ok", len(items), flush=True)
        mod = V.train4(items, seed=seed, steps=steps, short_w=3.0)
        res = {}
        for i in range(2):
            res[f"lp{i}"], res[f"b{i}"] = V.predict4(mod, canon[m][i])
            assert np.isfinite(res[f"lp{i}"]).all()
        np.savez(f, **res)
        del items, mod; gc.collect()


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "prep":
        prep()
    elif st == "smoke":
        fit(20, TUNE[:1], 60, "smoke"); prog("smoke ok")
    else:
        seed = int(sys.argv[2])
        fit(seed, TUNE, 1500, "aug")
        prog(f"train seed {seed} ({seed-19}/5 seeds done)")
