"""Freeze v2.1: with-ball dead model trained on all 5 TUNE matches under the matched degrade profile; params as v2."""
import json
import joblib
import numpy as np
import sklearn
from . import dead, integrate_run as IR, pipeline as pl
from .data import TUNE
from .matched_run import matched_data

P = IR.prep_all(matched_data())
names = P[TUNE[0]][0]["names"]
keep = pl.keep_columns(names, "ball")
model = IR.train_on(P, list(TUNE), keep)
joblib.dump({"model": model, "keep": keep, "features": [names[i] for i in keep], "variant": "ball", "sub": IR.SUB,
             "trained_on": list(TUNE), "profile": "matched", "windows_s": list(dead.WINDOWS_S),
             "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__}},
            pl.HERE / "models" / "dead_v2_1.joblib", compress=3)
p = {**pl.load_params(), "variant": "ball", "selection": "v2 params unchanged; dead model with ball features retrained on matched-degraded TUNE data"}
(pl.HERE / "params_v2_1.json").write_text(json.dumps(p, indent=1))
