"""python -m possession.test_v2_1 : pipeline.predict with v2_1 on one IDSSE half is deterministic (same output twice)."""
import numpy as np
from . import pipeline as pl
from .data import TUNE, load_idsse
from .degrade_matched import degrade_match

m = TUNE[0]
h = degrade_match(load_idsse(m), m)[0]
h = {k: (v[:20000] if k != "team" else v) for k, v in h.items()}
p, mod = pl.load_params(pl.HERE / "params_v2_1.json"), pl.load_models(pl.HERE / "models" / "dead_v2_1.joblib")
a, b = pl.predict(h, p, mod), pl.predict(h, p, mod)
assert a.shape == (20000,) and np.array_equal(a, b), "not deterministic"
print("ok", np.bincount(a, minlength=3))
