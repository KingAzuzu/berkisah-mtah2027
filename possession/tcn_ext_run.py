"""TCN LOMO on IDSSE TUNE with extra training data (Metrica games 1-3, degraded matched2) and pitch-mirror augmentation.
python -m possession.tcn_ext_run CONFIG SEED [STEPS]    CONFIG in {base, mir, ext, extmir}
Evaluation unchanged vs tcn_run.py: held-out IDSSE match never in training; scored by the organiser metric."""
import json, sys, time
import numpy as np
import joblib
from . import m2_lib as L, pipeline as pl, tcn_lib as T, extdata as E, degrade_matched2 as DM2
from .data import TUNE, idsse
from .score import score

cfg, seed = sys.argv[1], int(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
EXTRA, MIR = cfg.startswith("ext"), cfg.endswith("mir")
MIRROR_COLS = [0, 1, 10, 11, 20, 21]   # centre x/y of both teams, ball x/y


def mirror(X):
    X = X.copy()
    X[:, MIRROR_COLS] *= -1
    return X


data, P = L.data2(), L.prep2()
mods, keep = L.r_models(P)
R = L.r_preds(P, mods, keep)
feats = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
labs = {m: [T.labels5(h) for h in data[m]] for m in TUNE}
ext_f, ext_l = [], []
if EXTRA:
    fp = idsse.DATA / "external" / "tcn_feats_ext_matched2.pkl"
    if fp.exists():
        ext_f, ext_l = joblib.load(fp)
    else:
        for g, hv in joblib.load(idsse.DATA / "external" / "metrica_halves.pkl").items():
            for h in DM2.degrade_match(hv, g, 0):
                ext_f.append(T.frame_features(h)); ext_l.append(T.labels5(h))
        joblib.dump((ext_f, ext_l), fp)
res = {"config": cfg, "seed": seed, "steps": STEPS, "spell": {}, "fusion": {}, "macro": {}}
for m in TUNE:
    t0 = time.time()
    tr = [k for k in TUNE if k != m]
    F = [f for k in tr for f in feats[k]] + ext_f
    Y = [l for k in tr for l in labs[k]] + ext_l
    if MIR:
        F, Y = F + [mirror(f) for f in F], Y + Y
    mod = T.train(F, Y, seed=seed, steps=STEPS)
    lp = []
    for i in range(2):
        a = T.predict_logp(mod, feats[m][i], len(data[m][i]["truth"]))
        if MIR:
            a = 0.5 * (a + T.predict_logp(mod, mirror(feats[m][i]), len(data[m][i]["truth"])))
        lp.append(a)
    eR = [pl.combine(P[m][i]["team"], R[m][0][i], L.PARAMS["ALPHA"]) for i in range(2)]
    for name, es in (("alone", lp), ("fusion", [0.5 * a + 0.5 * b for a, b in zip(lp, eR)])):
        r = score(data[m], [pl.decode(e, L.PARAMS) for e in es])
        res["spell" if name == "alone" else "fusion"][m] = r["spell_f1_mean"]
        if name == "alone":
            res["macro"][m] = r["macro_f1"]
    print(cfg, seed, m, f"{time.time()-t0:.0f}s", round(res["spell"][m], 4), round(res["fusion"][m], 4), flush=True)
json.dump(res, open(pl.HERE / "logs" / f"tcn_ext_{cfg}_s{seed}.json", "w"), indent=1, default=float)
