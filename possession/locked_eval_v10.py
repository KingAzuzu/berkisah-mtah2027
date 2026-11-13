"""One-time evaluation of the frozen submission method v10 on the locked IDSSE matches (data.LOCKED).

    python -m possession.locked_eval_v10

v10 was frozen and chosen as the submission (HANDOVER, 11 Nov 2026) before this runs. v6 and v8 are scored next to it for the
record only: the choice is NOT revisited from these numbers. Degradation profile 'matched2' (seed 0), the same profile as the
TUNE evaluation. Scored against the DFL truth and against the touch-merged truth (v8_convention.merge(t, 2)).
Writes logs/locked_v10.json and refuses to run again if it exists.
"""
import json
from pathlib import Path

import numpy as np

from . import degrade_matched2 as DM2, pipeline
from .data import LOCKED, load_idsse
from .score import score
from .v8_convention import merge

LOG = Path(__file__).with_name("logs") / "locked_v10.json"

if __name__ == "__main__":
    if LOG.exists():
        raise SystemExit(f"{LOG} exists: the locked check was already run once")
    data = {m: DM2.degrade_match(load_idsse(m), m, 0) for m in LOCKED}
    preds = {}
    for name, fn in (("v6", pipeline.v6_predictor), ("v8", pipeline.v8_predictor), ("v10", pipeline.v10_predictor)):
        _, predict = fn()
        preds[name] = {m: [np.asarray(predict(h), np.int8) for h in data[m]] for m in LOCKED}
        print(name, "predicted", flush=True)
    out = {}
    for m in LOCKED:
        merged = [dict(h, truth=merge(np.asarray(h["truth"]), 2)) for h in data[m]]
        out[m] = {}
        for name in preds:
            r, rm = score(data[m], preds[name][m]), score(merged, preds[name][m])
            out[m][name] = {"spell_dfl": r["spell_f1_mean"], "spell_merged": rm["spell_f1_mean"], "macro_dfl": r["macro_f1"],
                            "n_pred": r["n_pred_spells"], "n_gt_dfl": r["n_gt_spells"], "n_gt_merged": rm["n_gt_spells"]}
    for name in preds:
        a = [out[m][name] for m in LOCKED]
        print(f"{name:<4} spellF1 DFL " + " ".join(f"{x['spell_dfl']:.4f}" for x in a) + f"  mean {np.mean([x['spell_dfl'] for x in a]):.4f}"
              f" | merged " + " ".join(f"{x['spell_merged']:.4f}" for x in a) + f"  mean {np.mean([x['spell_merged'] for x in a]):.4f}"
              f" | macroF1 DFL {np.mean([x['macro_dfl'] for x in a]):.4f}")
    LOG.write_text(json.dumps(out, indent=1))
