"""v11 E1: v8 recipe + external data with merged labels (Metrica merge K2 at 5 Hz; SkillCorner team-only, unmerged). python -m possession.v11_ext V1|V2 SEED.
Orig doc: v8 candidate: v6 recipe (cB, SHORT_W 3, seeds 20-24, LOMO on TUNE) trained on touch-merged labels (v8_convention.merge, K = 2 s).
Only the label changes. python -m possession.v8_train SEED -> logs/v8/m2_s{SEED}_{MATCH}.npz. Scored by v8_eval.
Gate fixed before running (vs merged truth): mean >= 0.7410 (v6+merge 0.7310 + 0.01) AND beats v6+merge on >= 4/5 matches."""
import sys, time
import numpy as np
import joblib

from . import m2_lib as L, tcn_lib as T, v4_lib as V
from .data import TUNE, idsse

VAR, seed = sys.argv[1], int(sys.argv[2])   # VAR V1 = IDSSE+Metrica, V2 = +SkillCorner
cfg, SHORT_W = "cB", 3.0
STEPS = 1500
USE = {"V1": (1, 1, 0), "V2": (1, 1, 1), "full": (1, 1, 1), "nomet": (1, 0, 1), "nosc": (1, 1, 0), "nobd": (1, 1, 1), "cA": (1, 1, 0), "cB": (1, 0, 0)}[VAR]   # idsse, met, sc
OUT = V.T.__file__.replace("tcn_lib.py", "logs/v11")
import os
os.makedirs(OUT, exist_ok=True)

data = L.data2()
canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
from .v8_convention import merge
labs = {m: [T.labels5(dict(h, truth=merge(h["truth"], 2))) for h in data[m]] for m in TUNE}
ext = idsse.DATA / "external"
met_f, met_l = joblib.load(ext / "tcn_feats_ext_matched2.pkl") if USE[1] else ([], [])
sc = joblib.load(ext / "tcn_feats_sc.pkl") if USE[2] else {}
for m in TUNE:
    f = os.path.join(OUT, f"{VAR}_s{seed}_{m}.npz")
    if os.path.exists(f):
        continue
    t0 = time.time()
    items = []
    for k in TUNE:
        if k == m:
            continue
        for i in range(2):
            items.append({"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))], "y": labs[k][i]})
    for ff, ll in zip(met_f, met_l):
        ll = merge(ll, 2, fps=5)   # Metrica labels are 5 Hz; merge K=2 s -> 10 steps
        items.append({"src": "met", "feats": [ff], "y": ll})
    for mid, (fs, ls) in sc.items():
        for ff, ll in zip(fs, ls):
            items.append({"src": "sc", "feats": [ff], "y": ll})
    mod = V.train4(items, seed=seed, steps=STEPS, short_w=SHORT_W)
    res = {}
    for i in range(2):
        lp, b = V.predict4(mod, canon[m][i])
        res[f"lp{i}"], res[f"b{i}"] = lp, b
    np.savez(f, **res)
    print(VAR, seed, m, f"{time.time()-t0:.0f}s", flush=True)
