"""Descriptive stats of the degrade2 perturbations (missing rate over dead frames, ball noise, moved tracklets)."""
import numpy as np
from . import degrade2
from .data import TUNE, pm
from .run_idsse import load

for m in TUNE:
    hs = load(m)
    dm = degrade2.apply("dead_missing0.7", hs, m)
    wt = degrade2.apply("wrong_team0.02", hs, m)
    def dead_miss(hh):
        d = np.concatenate([h["truth"] == pm.D for h in hh]); b = np.concatenate([np.isnan(h["ball"][:, 0]) for h in hh])
        return b[d].mean(), b.mean()
    mv = sum(int(np.sum(np.any(np.nan_to_num(a["xy"]) != np.nan_to_num(b["xy"]), axis=(0, 2)))) for a, b in zip(hs, wt))
    print(m, "dead-frame missing before/after", [round(x, 3) for x in dead_miss(hs)[:1]], round(dead_miss(dm)[0], 3),
          "all-frame missing", round(dead_miss(hs)[1], 3), round(dead_miss(dm)[1], 3), "slots changed by wrong_team", mv)
