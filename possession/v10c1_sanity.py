"""Sanity check for v10c1 (no held-out score exists: all 7 IDSSE matches are in its training set).
Pass criteria fixed before running: v10c1 in-sample spellF1 on every match >= v10 - 0.02 (nothing broke), and the frame agreement
with v10 on every match >= 0.95 (same method, more data). python -m possession.v10c1_sanity -> logs/v10c1_sanity.json
"""
import json
from pathlib import Path

import numpy as np

from . import degrade_matched2 as DM2, m2_lib as L, pipeline
from .data import LOCKED, TUNE, load_idsse
from .score import score

if __name__ == "__main__":
    data = {**L.data2(), **{m: DM2.degrade_match(load_idsse(m), m, 0) for m in LOCKED}}
    P = {}
    for name, fn in (("v10", pipeline.v10_predictor), ("v10c1", pipeline.v10c1_predictor)):
        _, f = fn()
        P[name] = {m: [np.asarray(f(h), np.int8) for h in data[m]] for m in (*TUNE, *LOCKED)}
    out, ok = {}, True
    for m in (*TUNE, *LOCKED):
        a, b = score(data[m], P["v10"][m])["spell_f1_mean"], score(data[m], P["v10c1"][m])["spell_f1_mean"]
        agr = float(np.mean(np.concatenate(P["v10"][m]) == np.concatenate(P["v10c1"][m])))
        out[m] = {"v10": a, "v10c1": b, "agree": agr, "in_sample_v10": m in TUNE}
        ok &= b >= a - 0.02 and agr >= 0.95
        print(f"{m}  v10 {a:.4f}  v10c1 {b:.4f}  agree {agr:.4f}" + ("  (locked: in-sample for v10c1 only)" if m in LOCKED else ""))
    out["pass"] = bool(ok)
    print("PASS" if ok else "FAIL")
    Path(__file__).with_name("logs").joinpath("v10c1_sanity.json").write_text(json.dumps(out, indent=1))
