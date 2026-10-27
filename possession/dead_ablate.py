"""WS4 ablation: LOMO frame AUC / D F1 with ball-derived features removed (players only)."""
import json
import numpy as np
from . import dead
from .data import TUNE
from .dead_experiment import fit, predict, auc_f1
from .lomo import LOG_DIR
from .run_idsse import load

data = {m: load(m) for m in TUNE}
F, Y = {}, {}
for m in TUNE:
    b = [dead.build_features(h) for h in data[m]]
    F[m], names = [x[0] for x in b], b[0][1]
    Y[m] = [dead.labels(h) for h in data[m]]
ball = ("b_", "d_nearest", "n_near", "d_gap", "d_min_team")
keep = [i for i, n in enumerate(names) if not n.startswith(ball)]
print("features kept", len(keep), "of", len(names))
out = {}
for m in TUNE:
    rest = [k for k in TUNE if k != m]
    mod = fit([x[:, keep] for k in rest for x in F[k]], [y for k in rest for y in Y[k]])
    p = np.concatenate(predict(mod, [x[:, keep] for x in F[m]]))
    auc, f1 = auc_f1(np.concatenate(Y[m]), p)
    out[m] = {"auc": auc, "f1_D": f1}
    print(m, out[m], flush=True)
(LOG_DIR / "dead_ablate.json").write_text(json.dumps(out, indent=1))
