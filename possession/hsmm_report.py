"""Writes the duration-histogram and per-match comparison for WS1 (reads logs/hsmm.json, logs/reference_safety.json)."""
import json
import time
from pathlib import Path

import numpy as np

from . import hsmm, safety
from .data import TUNE
from .hsmm_experiment import PARAMS, evid

L = Path(__file__).with_name("logs")
BINS = [0, 25, 50, 125, 250, 500, 1000, 10**9]  # frames: <1s,1-2,2-5,5-10,10-20,20-40,>40s
NAMES = ["<1s", "1-2s", "2-5s", "5-10s", "10-20s", "20-40s", ">40s"]


def hist(lens):
    return np.histogram(lens, BINS)[0]


def main():
    h = json.loads((L / "hsmm.json").read_text())
    ref = json.loads((L / "reference_safety.json").read_text())
    p = {**safety.PARAMS, **PARAMS}
    truth = {j: [] for j in range(3)}
    refl = {j: [] for j in range(3)}
    for m in TUNE:
        halves, ev = evid(m)
        for hf, e in zip(halves, ev):
            for src, arr in ((truth, hf["truth"]), (refl, safety.viterbi(e, p["SWITCH_TEAM"], p["SWITCH_DEAD"]))):
                st, ln = hsmm._runs(arr)
                for s, n in zip(st, ln):
                    src[int(s)].append(int(n))
    out = {"bins": NAMES, "state": {}}
    for j, nm in enumerate("HAD"):
        out["state"][nm] = {"truth": hist(truth[j]).tolist(), "reference": hist(refl[j]).tolist(),
                            "hsmm": hist(h["selected_run_lengths_frames"][str(j)]).tolist()}
    # runtime and determinism on the longest tuning half
    halves, ev = evid(TUNE[0])
    model = hsmm.fit([q["truth"] for m in TUNE[1:] for q in evid(m)[0]])
    times, same = [], True
    for e in ev:
        t0 = time.time()
        a = hsmm.decode(e, model, 3.0, 2.0)
        times.append((len(e), round(time.time() - t0, 2)))
        same &= bool(np.array_equal(a, hsmm.decode(e, model, 3.0, 2.0)))
    out["runtime_s_per_half"] = times
    out["deterministic"] = same
    out["reference_held"] = ref["held_out"]
    out["ref_mean"] = [ref["lomo_spell_mean"], ref["lomo_macro_mean"]]
    (L / "hsmm_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
