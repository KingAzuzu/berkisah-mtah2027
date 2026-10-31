"""LOMO under the matched degrade profile (TUNE matches only). No retune of switch costs / alpha (scope cut).
    python -m possession.matched_run
(a) v1 safety; (b) v2 as frozen: dead model trained on the OLD-profile degraded data of the other 4 matches;
(c) dead model retrained on matched data of the other 4 matches, players-only and with-ball.
A held-out match never enters any model that scores it."""
import json
import joblib
import numpy as np
from joblib import Parallel, delayed
from . import degrade_matched as DM, integrate_run as IR, pipeline as pl, safety
from .data import TUNE, idsse
from .run_idsse import load
from .score import score

OUT = IR.LOG_DIR / "matched_run.json"


def matched_data(seed=0):
    from .data import load_idsse
    path = idsse.DATA / "cache" / f"matched_seed{seed}.pkl"
    if path.exists():
        return joblib.load(path)
    d = {m: DM.degrade_match(load_idsse(m), m, seed) for m in TUNE}
    joblib.dump(d, path)
    return d


def run_models(Ptr, Pte, variant, params):
    """LOMO: model for m trained on Ptr of the other 4, applied to Pte[m]."""
    keep = pl.keep_columns(Ptr[TUNE[0]][0]["names"], variant)
    res = {}
    for m in TUNE:
        mod = IR.train_on(Ptr, [k for k in TUNE if k != m], keep)
        pds = IR.predict_p(mod, Pte, m, keep)
        preds = IR._decode_cell([h["team"] for h in Pte[m]], [h["e_rules"] for h in Pte[m]], pds, params)
        res[m] = preds
    return res


def main():
    params = pl.load_params()
    data = matched_data()
    Pm = IR.prep_all(data)
    Po = IR.prep_all(IR.load_cond("seed0"))
    out = {}
    v1p = json.loads((pl.HERE / "params.json").read_text())
    out["a_v1"] = {m: score(data[m], [safety.predict(h, v1p) for h in data[m]]) for m in TUNE}
    cands = {"b_v2_frozen": run_models(Po, Pm, "players", params),
             "c_players_retrained": run_models(Pm, Pm, "players", params),
             "c_ball_retrained": run_models(Pm, Pm, "ball", params)}
    for k, pr in cands.items():
        out[k] = {m: score(data[m], pr[m]) for m in TUNE}
    summ = {}
    for k, r in out.items():
        sp = [r[m]["spell_f1_mean"] for m in TUNE]
        summ[k] = {"spell_mean": float(np.mean(sp)), "spell_min": min(sp), "spell_max": max(sp),
                   "macro_mean": float(np.mean([r[m]["macro_f1"] for m in TUNE])),
                   "spell_per_match": dict(zip(TUNE, sp)),
                   "macro_per_match": {m: r[m]["macro_f1"] for m in TUNE}}
        print(k, f"spell {summ[k]['spell_mean']:.4f} [{min(sp):.4f},{max(sp):.4f}] macro {summ[k]['macro_mean']:.4f}",
              " ".join(f"{x:.3f}" for x in sp), flush=True)
    OUT.write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
