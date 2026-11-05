"""v7 step 3: v6 recipe (cB, seeds 20-24, LOMO on TUNE) with short-spell loss factor 5 or 8 instead of 3. IDSSE TUNE only.
python -m possession.v7_short FACTOR SEED -> logs/v7/sF{FACTOR}_s{SEED}_{MATCH}.npz. Scored by v7_short_eval.
Gate fixed before running: a factor passes if fused-decoder (v6 decoder) LOMO mean >= 0.6960 (v6 + 0.01) AND beats v6 on >= 4/5 matches."""
import sys, time
import numpy as np
import joblib

from . import m2_lib as L, tcn_lib as T, v4_lib as V
from .data import TUNE, idsse

cfg, SHORT_W, seed = "cB", float(sys.argv[1]), int(sys.argv[2])
STEPS = 1500
USE = {"full": (1, 1, 1), "nomet": (1, 0, 1), "nosc": (1, 1, 0), "nobd": (1, 1, 1), "cA": (1, 1, 0), "cB": (1, 0, 0)}[cfg]   # idsse, met, sc
OUT = V.T.__file__.replace("tcn_lib.py", "logs/v7")
import os
os.makedirs(OUT, exist_ok=True)

data = L.data2()
canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
labs = {m: [T.labels5(h) for h in data[m]] for m in TUNE}
ext = idsse.DATA / "external"
met_f, met_l = joblib.load(ext / "tcn_feats_ext_matched2.pkl") if USE[1] else ([], [])
sc = joblib.load(ext / "tcn_feats_sc.pkl") if USE[2] else {}
for m in TUNE:
    f = os.path.join(OUT, f"sF{int(SHORT_W)}_s{seed}_{m}.npz")
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
    print(cfg, SHORT_W, seed, m, f"{time.time()-t0:.0f}s", flush=True)
