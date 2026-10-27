"""One-time evaluation of the frozen v2 method on the locked IDSSE matches (data.LOCKED).

Run once, after the method was frozen (params_v2.json, models/dead_v2.joblib). Writes logs/locked_v2.json.
Refuses to run again if that log exists, so the locked matches cannot drift into tuning.
"""
import json
import sys
from pathlib import Path

from . import pipeline, safety
from .data import LOCKED
from .run_idsse import load
from .score import score

LOG = Path(__file__).with_name("logs") / "locked_v2.json"

if __name__ == "__main__":
    if LOG.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{LOG} exists: the locked check was already run once")
    params, models = pipeline.load_params(), pipeline.load_models()
    ref = json.loads(Path(__file__).with_name("params.json").read_text())
    out = {}
    for m in LOCKED:
        halves = load(m)
        r_ref = score(halves, [safety.predict(h, ref) for h in halves])
        r_v2 = score(halves, [pipeline.predict(h, params, models) for h in halves])
        out[m] = {k: {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"],
                      "n_gt": r["n_gt_spells"]} for k, r in (("reference", r_ref), ("v2", r_v2))}
        print(m, json.dumps(out[m]))
    LOG.write_text(json.dumps(out, indent=1))
