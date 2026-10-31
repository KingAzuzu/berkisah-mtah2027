"""Reference R: v2.1 (with-ball dead model, params_v2_1) LOMO under degrade profile 'matched2' (TUNE only).
Cached on disk (IDSSE-derived, public). Used by error_analysis.py and ctx_models.py."""
import json
import joblib
import numpy as np
from . import degrade_matched2 as DM2, integrate_run as IR, pipeline as pl
from .data import TUNE, idsse, load_idsse
from .score import score

CACHE = idsse.DATA / "cache"
PARAMS = json.loads((pl.HERE / "params_v2_1.json").read_text())


def data2(seed=0):
    path = CACHE / f"matched2_seed{seed}.pkl"
    if path.exists():
        return joblib.load(path)
    d = {m: DM2.degrade_match(load_idsse(m), m, seed) for m in TUNE}
    joblib.dump(d, path)
    return d


def prep2():
    path = CACHE / "matched2_prep.pkl"
    if path.exists():
        return joblib.load(path)
    P = IR.prep_all(data2())
    joblib.dump(P, path)
    return P


def r_models(P):
    """Dead models for each held-out match (trained on the other 4)."""
    keep = pl.keep_columns(P[TUNE[0]][0]["names"], "ball")
    return {m: IR.train_on(P, [k for k in TUNE if k != m], keep) for m in TUNE}, keep


def r_preds(P, mods, keep, params=PARAMS):
    out = {}
    for m in TUNE:
        pds = IR.predict_p(mods[m], P, m, keep)
        out[m] = (pds, IR._decode_cell([h["team"] for h in P[m]], [h["e_rules"] for h in P[m]], pds, params))
    return out


if __name__ == "__main__":
    data, P = data2(), prep2()
    mods, keep = r_models(P)
    rp = r_preds(P, mods, keep)
    sp = []
    for m in TUNE:
        r = score(data[m], rp[m][1]); sp.append(r["spell_f1_mean"])
        print(m, round(r["spell_f1_mean"], 4), round(r["macro_f1"], 4))
    print("R spellF1 %.4f [%.4f, %.4f]" % (np.mean(sp), min(sp), max(sp)))
    joblib.dump({m: rp[m] for m in TUNE}, CACHE / "matched2_R.pkl")
    json.dump({"spell_mean": float(np.mean(sp)), "spell_min": min(sp), "spell_max": max(sp),
               "per_match": dict(zip(TUNE, sp))}, open(pl.HERE / "logs" / "m2_R.json", "w"), indent=1)
