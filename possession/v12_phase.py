"""v12 idea A: phase-shift TTA. The TCN reads 5-frame block means (25 -> 5 Hz); the held-out half is also run with the input shifted by
k = 1..4 frames (features recomputed on the shifted half), outputs are upsampled to 25 fps, shifted back and averaged with phase 0.
Fold nets are not saved by v8_train, so they are retrained here with the identical recipe/seeds and used at once for all phases.
python -m possession.v12_phase LABELS SEED   (LABELS = v8 merged-K2 labels | v6 raw labels)  -> logs/v12/{LABELS}_s{SEED}_{MATCH}.npz
Gate fixed before running (v8 labels, vs merged truth): TTA mean >= 0.7663 (v8 0.7563 + 0.01) AND beats v8 on >= 4/5 matches."""
import sys, time
import numpy as np
import joblib

from . import m2_lib as L, tcn_lib as T, v4_lib as V
from .data import TUNE, idsse

cfg, SHORT_W, LAB, seed = "cB", 3.0, sys.argv[1], int(sys.argv[2])
STEPS = 1500
USE = {"full": (1, 1, 1), "nomet": (1, 0, 1), "nosc": (1, 1, 0), "nobd": (1, 1, 1), "cA": (1, 1, 0), "cB": (1, 0, 0)}[cfg]   # idsse, met, sc
OUT = V.T.__file__.replace("tcn_lib.py", "logs/v12")
import os
os.makedirs(OUT, exist_ok=True)

data = L.data2()
canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
from .v8_convention import merge
labs = {m: [T.labels5(dict(h, truth=merge(h["truth"], 2)) if LAB == "v8" else h) for h in data[m]] for m in TUNE}
ext = idsse.DATA / "external"
met_f, met_l = joblib.load(ext / "tcn_feats_ext_matched2.pkl") if USE[1] else ([], [])
sc = joblib.load(ext / "tcn_feats_sc.pkl") if USE[2] else {}
for m in TUNE:
    f = os.path.join(OUT, f"{LAB}_s{seed}_{m}.npz")
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
        h = data[m][i]
        for k in range(5):
            X = canon[m][i] if k == 0 else T.frame_features({"xy": h["xy"][k:], "ball": h["ball"][k:], "team": h["team"]})
            lp, b = V.predict4(mod, X)
            res[f"lp{i}_k{k}"], res[f"b{i}_k{k}"] = lp, b
    np.savez(f, **res)
    import glob
    n_done = len(glob.glob(OUT + "/" + LAB + "_s*.npz"))
    open(os.path.join(OUT, "progress.txt"), "w").write("A: %s seed %d fold %s done (%d/25)" % (LAB, seed, m, n_done))
    print(cfg, SHORT_W, LAB, seed, m, f"{time.time()-t0:.0f}s", flush=True)
