"""v21 = the v21_plan winner, frozen: TCN21 seed ensemble trained on matched3 (+ DR), dead-ball model trained on matched3, fused
as v10 (0.5 * mean log-prob + 0.5 * rules/dead evidence), boundary-hook Viterbi. Recipe, dead variant and decoder costs are read
from params_v21.json, written by `freeze` from the LOMO result (possession/results/v21.md).

Inference (CPU, float32, 1 thread, deterministic): python -m possession.bar_ath --method v21
Freeze the v21_select winner (IDSSE TUNE only, CPU or GPU): python -m possession.tcn_v21 freeze
Locked check, once, after freezing:             python -m possession.tcn_v21 locked"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from . import tcn21_lib as L, v4_lib as V

HERE = Path(__file__).parent
SEEDS = (20, 21, 22, 23, 24)


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v21.json").read_text())


def load_models():
    return joblib.load(HERE / "models" / "dead_v21.joblib")


def load_nets(params=None):
    params = params or load_params()
    return [L.load(HERE / "models" / f"tcn_v21_s{s}.pt") for s in params["seeds"]]


def predict(half, params, models, nets):
    from . import pipeline as pl
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    n = len(half["ball"])
    X = L.features(half, params["fset"])
    outs = [L.predict21(m, X) for m in nets]
    lp = V.upsample(np.mean([o[0] for o in outs], 0), n)
    b = V.upsample(np.mean([o[1] for o in outs], 0)[:, None], n)[:, 0]
    e = params["fusion"] * lp + (1 - params["fusion"]) * pl.evidence(half, params, models)
    from .v21_decoder import viterbi_batch
    cfg = np.array([[params["SWITCH_TEAM"], params["SWITCH_DEAD_IN"], params["SWITCH_DEAD_OUT"], params["SAME_RESTART"]]])
    return viterbi_batch(e, cfg, 1 - params["hook"] * b)[0]


def predictor():
    p = load_params()
    models, nets = load_models(), load_nets(p)
    return p, lambda half: predict(half, p, models, nets)


def freeze_selected():
    """Freeze the winner written by v21_select (logs/v21/select.json)."""
    from . import v21_lomo as H
    w = json.load(open(H.LOG / "select.json"))["winner"]
    d = w["decoder"]
    freeze(w["recipe"], w["tsim"], w["variant"], d["st"], d["sd_in"], d["sd_out"], d["same"], selected=w["name"])


def freeze(recipe, tsim, variant, st=20.0, sd_in=5.0, sd_out=5.0, same=0.0, selected="manual"):
    from . import dead, pipeline as pl, tcn_lib as T, v21_lomo as H
    from .data import TUNE
    r = H.RECIPES[recipe]
    nin = 26 if r["fset"] == "f26" else 32
    D = {m: H.load_data(m) for m in TUNE}
    # dead model on all 5 TUNE matches, canonical degrade of profile tsim
    names = D[TUNE[0]]["evals"][tsim][0]["names"]
    keep = pl.keep_columns(names, variant)
    X = np.concatenate([h["X"][::5][:, keep] for m in TUNE for h in D[m]["evals"][tsim]])
    y = np.concatenate([h["y"][::5] for m in TUNE for h in D[m]["evals"][tsim]])
    joblib.dump({"model": dead.make_model().fit(X, y), "keep": keep, "features": names}, HERE / "models" / "dead_v21.joblib")
    print("saved dead_v21", variant, flush=True)
    items = []
    for m in TUNE:
        for i in range(2):
            yy = T.labels5({"truth": D[m]["evals"]["m3"][i]["truth"]})
            items.append({"feats": [D[m]["feats"][(r["sim"], v)][i][:len(yy), :nin] for v in range(H.K + 1)], "y": yy})
    for s in SEEDS:
        mod = L.train21([dict(it) for it in items], seed=s, crop=r["crop"], dil=r["dil"], nin=nin)
        L.save(mod, HERE / "models" / f"tcn_v21_s{s}.pt", {"recipe": recipe, "trained_on": list(TUNE), "profile": "matched3+DR"})
        print("saved seed", s, flush=True)
    base = {k: v for k, v in pl.base_params().items() if isinstance(v, (int, float, str))}
    p = {**base, "ALPHA": 0.5, "SWITCH_TEAM": st, "SWITCH_DEAD_IN": sd_in, "SWITCH_DEAD_OUT": sd_out, "SAME_RESTART": same,
         "DECODER": "viterbi4", "variant": variant, "dead_profile": tsim, "seeds": list(SEEDS), "fusion": 0.5, "hook": V.HOOK,
         "fset": r["fset"], "recipe": recipe, "dil": list(r["dil"]), "steps": 1500,
         "sources": f"IDSSE TUNE ({r['sim']} canonical + 6 DR variants), DFL labels, mirror aug + TTA",
         "selection": f"v21_plan gates on matched3 LOMO, winner {selected}; see results/v21.md"}
    (HERE / "params_v21.json").write_text(json.dumps(p, indent=1))


def locked():
    """One-time check on the locked matches degraded with matched3 (refuses to rerun)."""
    from . import degrade_m3 as M3
    from .data import LOCKED, load_idsse
    from .score import score
    out = HERE / "logs" / "locked_v21.json"
    if out.exists():
        sys.exit("refusing: the locked check has already been run (logs/locked_v21.json)")
    p, predict_half = predictor()
    res = {}
    for m in LOCKED:
        halves = M3.degrade_match(load_idsse(m), m, 0)
        preds = [predict_half({k: h[k] for k in ("ball", "xy", "team")}) for h in halves]
        r = score(halves, preds)
        res[m] = {"spell": float(r["spell_f1_mean"]), "macro": float(r["macro_f1"]), "n_pred": int(r["n_pred_spells"])}
        print(m, res[m], flush=True)
    res["mean_spell"] = float(np.mean([res[m]["spell"] for m in LOCKED]))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze_selected()
    elif sys.argv[1] == "locked":
        locked()
