"""Part A: throw-in team rule vs the DFL ThrowIn Team attribute, degraded IDSSE tuning matches.

    python -m possession.restarts_throwin

Headline = the pre-set rule restarts.RULE (W 25 frames, margin 1.0 m) at the DFL event frame.
Sensitivity (other W / margin, and jittered frames) is reported beside it, not used to move the bar (>= 90%).
One match at a time; no model, nothing fitted.
"""
import gc
import json
from pathlib import Path

import numpy as np

from . import restarts as R
from .data import TUNE, idsse
from .run_idsse import load

LOG = Path(__file__).with_name("logs") / "restarts_throwin.json"
BAR = 0.90
SENS = [(12, 1.0), (25, 0.5), (25, 1.0), (25, 2.0), (50, 1.0), (50, 2.0)]
JIT = [0.5, 1.5]


def main():
    res = {"bar": BAR, "rule": R.RULE, "per_match": {}, "sens": {}, "jitter": {}}
    sens = {f"W{w}_m{mg}": [0, 0, 0] for w, mg in SENS}   # correct, wrong, none
    jit = {s: [0, 0, 0] for s in JIT}
    tot = [0, 0, 0]
    for mid in TUNE:
        m = idsse.load_match(mid)
        halves = load(mid)
        for hi, h in enumerate(halves):
            assert len(h["ball"]) == m["halves"][hi]["T"]
        ev = [e for e in R.events_by_half(mid, m) if e["type"] == "ThrowIn"]
        del m
        c = [0, 0, 0]
        rng = np.random.default_rng(0)
        for e in ev:
            h = halves[e["half"]]
            def tally(acc, pred):
                k = 2 if pred is None else 0 if pred == e["team"] else 1
                acc[k] += 1
            tally(c, R.throw_team(h, e["frame"]))
            for w, mg in SENS:
                tally(sens[f"W{w}_m{mg}"], R.throw_team(h, e["frame"], W=w, margin=mg))
            for s in JIT:
                f = int(np.clip(round(e["frame"] + rng.normal(0, s * R.FPS)), 0, len(h["ball"]) - 1))
                tally(jit[s], R.throw_team(h, f, W=max(R.RULE["W"], int(round(2 * s * R.FPS)))))
        n = len(ev)
        res["per_match"][mid] = {"n": n, "correct": c[0], "wrong": c[1], "none": c[2], "acc": c[0] / n}
        tot = [a + b for a, b in zip(tot, c)]
        print(f"{mid} n={n} correct {c[0]} wrong {c[1]} none {c[2]} acc {c[0] / n:.3f}", flush=True)
        del halves
        gc.collect()
    n = sum(tot)
    res["overall"] = {"n": n, "correct": tot[0], "wrong": tot[1], "none": tot[2], "acc": tot[0] / n,
                      "acc_among_found": tot[0] / max(1, tot[0] + tot[1]), "none_rate": tot[2] / n,
                      "pass": tot[0] / n >= BAR}
    res["sens"] = {k: {"correct": v[0], "wrong": v[1], "none": v[2], "acc": v[0] / n} for k, v in sens.items()}
    res["jitter"] = {str(k): {"correct": v[0], "wrong": v[1], "none": v[2], "acc": v[0] / n} for k, v in jit.items()}
    print(json.dumps(res["overall"]))
    for k, v in res["sens"].items():
        print("sens", k, v)
    for k, v in res["jitter"].items():
        print("jitter sigma", k, v)
    LOG.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
