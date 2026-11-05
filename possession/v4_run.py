"""v4 LOMO fold runner. python -m possession.v4_run CONFIG SEED [STEPS]
CONFIG in {full, nomet, nosc, nobd}. For each held-out IDSSE TUNE match (never in training; training = other 4 + sources)
saves 5 Hz log-probs and boundary probs of both halves to logs/v4/CONFIG_sSEED_MATCH.npz. Updates logs/v4_progress.txt."""
import sys, time
import numpy as np
import joblib

from . import m2_lib as L, tcn_lib as T, v4_lib as V
from .data import TUNE, idsse
from . import v4_status

cfg, seed = sys.argv[1], int(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
USE = {"full": (1, 1, 1), "nomet": (1, 0, 1), "nosc": (1, 1, 0), "nobd": (1, 1, 1)}[cfg]   # idsse, met, sc
OUT = V.T.__file__.replace("tcn_lib.py", "logs/v4")
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
    f = os.path.join(OUT, f"{cfg}_s{seed}_{m}.npz")
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
    mod = V.train4(items, seed=seed, steps=STEPS, bd_lambda=0.0 if cfg == "nobd" else V.BD_LAMBDA)
    res = {}
    for i in range(2):
        lp, b = V.predict4(mod, canon[m][i])
        res[f"lp{i}"], res[f"b{i}"] = lp, b
    np.savez(f, **res)
    print(cfg, seed, m, f"{time.time()-t0:.0f}s", flush=True)
    v4_status.update()
