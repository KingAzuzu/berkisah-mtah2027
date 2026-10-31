"""v3 = TCN (seed ensemble) + v2.1 evidence fused 50/50, then Viterbi with params_v2_1 switch costs.

Inference: python -m possession.bar_ath --method v3 (torch imported lazily, CPU, float32, 1 thread,
deterministic algorithms). Freeze (human/dev, GPU, IDSSE TUNE only): python -m possession.tcn_v3 [seeds ...]
"""
import json
from pathlib import Path

import numpy as np
import torch

from . import tcn_lib as T

HERE = Path(__file__).parent
FUSION = 0.5


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v3.json").read_text())


def load_nets(seeds=None, params=None):
    params = params or load_params()
    nets = []
    for s in (seeds if seeds is not None else params["seeds"]):
        ck = torch.load(HERE / "models" / f"tcn_v3_s{s}.pt", map_location="cpu", weights_only=True)
        net = T.TCN(**json.loads(ck["config"])["net"])
        net.load_state_dict(ck["state_dict"])
        nets.append({"net": net.float().eval(), "mu": ck["mu"].numpy(), "sd": ck["sd"].numpy()})
    return nets


def logp(half, nets):
    """Ensemble (mean) per-frame log-probs (T,3) at 25 fps; CPU float32, deterministic."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    n = len(half["ball"])
    X = T.frame_features(half)
    return np.mean([T.predict_logp(m, X, n) for m in nets], axis=0)


def predict(half, params, models, nets):
    from . import pipeline as pl
    e = FUSION * logp(half, nets) + (1 - FUSION) * pl.evidence(half, params, models)
    return pl.decode(e, params)


def freeze(seeds=(0, 1, 2), steps=1500):
    import joblib
    from . import m2_lib as L
    from .data import TUNE, idsse
    data = L.data2()
    feats = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    F = [f for m in TUNE for f in feats[m]]
    Y = [T.labels5(h) for m in TUNE for h in data[m]]
    for s in seeds:
        mod = T.train(F, Y, seed=s, steps=steps)
        cfg = {"net": {"nin": 26, "c": 64, "p": 0.2}, "steps": steps, "seed": s, "crop": 300, "bs": 64, "dil": list(T.DIL),
               "trained_on": list(TUNE), "profile": "matched2", "torch": torch.__version__}
        torch.save({"state_dict": mod["net"].state_dict(), "mu": torch.tensor(mod["mu"]), "sd": torch.tensor(mod["sd"]),
                    "config": json.dumps(cfg)}, HERE / "models" / f"tcn_v3_s{s}.pt")
        print("saved seed", s, flush=True)


def write_params(seeds=(0, 1, 2), steps=1500):
    p = {**load_params(HERE / "params_v2_1.json"), "seeds": list(seeds), "fusion": FUSION, "steps": steps,
         "selection": "TCN seed-ensemble (mean log-probs) fused 50/50 with v2.1 evidence, Viterbi with params_v2_1 costs"}
    (HERE / "params_v3.json").write_text(json.dumps(p, indent=1))




if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["params"]:
        write_params()
    else:   # one process per seed may run concurrently: python -m possession.tcn_v3 <seed>
        freeze(tuple(int(a) for a in sys.argv[1:]) or (0, 1, 2))
