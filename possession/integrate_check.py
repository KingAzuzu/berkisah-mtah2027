"""Step 4 checks: pipeline.predict reproduces the logged LOMO score for a fold, and the frozen model is deterministic."""
import json

import joblib
import numpy as np

from . import pipeline as pl
from .data import TUNE
from .integrate_run import CACHE
from .lomo import LOG_DIR
from .run_idsse import load
from .score import score

rep = json.loads((LOG_DIR / "integrate_s1_players_seed0.json").read_text())
m = TUNE[0]
h = rep["held_out"][m]
models = joblib.load(CACHE / f"outer_seed0_players_{m}.joblib")
data = load(m)
preds = [pl.predict(x, h["cell"], models) for x in data]
r = score(data, preds)
print(f"fold {m}: pipeline spellF1 {r['spell_f1_mean']:.6f} vs logged {h['spell']:.6f}; macro {r['macro_f1']:.6f} vs {h['macro']:.6f}")
assert abs(r["spell_f1_mean"] - h["spell"]) < 1e-9

params, final = pl.load_params(), pl.load_models()
a = pl.predict(data[0], params, final)
b = pl.predict(data[0], params, final)
print("frozen model, two runs identical:", bool(np.array_equal(a, b)), "codes", np.bincount(a, minlength=3).tolist())
print("frozen params:", json.dumps(params))
