"""Predicted/true spell counts for v4 and the E1 nested picks (reads logs/v6/e1.json)."""
import json
from multiprocessing import Pool
from . import v6_e1 as E
from .data import TUNE
import joblib

def f(a):
    m, pt = a
    from . import v4_lib as V
    from .score import score
    st, sd, hk, fw = pt
    pred = [V.viterbi_scaled(fw * h["lp"] + (1 - fw) * h["eR"], st, sd, 1 - hk * h["b"]) for h in E.C[m]]
    r = score(E.C[m], pred)
    return r["n_pred_spells"], r["n_gt_spells"], r["spell_f1_mean"]

if __name__ == "__main__":
    o = json.load(open(E.LOG / "v6" / "e1.json"))
    c = joblib.load(E.CACHE)
    jobs = [(m, E.V4) for m in TUNE] + [(m, tuple(p)) for m, p in zip(TUNE, o["selected"])]
    with Pool(10, initializer=E.init, initargs=(c,)) as p:
        r = p.map(f, jobs)
    o["counts_v4"], o["counts_nested"] = r[:5], r[5:]
    json.dump(o, open(E.LOG / "v6" / "e1.json", "w"), indent=1)
    print("v4", r[:5]); print("nested", r[5:])
