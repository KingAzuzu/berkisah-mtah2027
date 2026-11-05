"""v4 = TCN4 (boundary head, EMA, domain randomisation, mirror TTA) seed ensemble + v2.1 evidence fused 50/50, then Viterbi
with the switch cost scaled by (1 - HOOK * ensemble boundary prob). SkillCorner NOT used.

Inference: python -m possession.bar_ath --method v4 (CPU, float32, 1 thread, deterministic algorithms).
Freeze (dev, GPU, IDSSE TUNE only): python -m possession.tcn_v4 CONFIG <seed> ...  then  python -m possession.tcn_v4 params CONFIG
"""
import json
from pathlib import Path

import numpy as np
import torch

from . import tcn_lib as T, v4_lib as V

HERE = Path(__file__).parent
FUSION = 0.5


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v4.json").read_text())


def load_nets(seeds=None, params=None):
    params = params or load_params()
    nets = []
    for s in (seeds if seeds is not None else params["seeds"]):
        ck = torch.load(HERE / "models" / f"tcn_v4_s{s}.pt", map_location="cpu", weights_only=True)
        net = V.TCN4(**json.loads(ck["config"])["net"])
        net.load_state_dict(ck["state_dict"])
        nets.append({"net": net.float().eval(), "mu": ck["mu"].numpy(), "sd": ck["sd"].numpy()})
    return nets


def predict(half, params, models, nets):
    from . import pipeline as pl
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    n = len(half["ball"])
    X = T.frame_features(half)
    outs = [V.predict4(m, X) for m in nets]
    lp = V.upsample(np.mean([o[0] for o in outs], 0), n)
    b = V.upsample(np.mean([o[1] for o in outs], 0)[:, None], n)[:, 0]
    e = FUSION * lp + (1 - FUSION) * pl.evidence(half, params, models)
    p = pl.base_params(params)
    t = p.get("TEMP", 1.0)
    return V.viterbi_scaled(e / t, p["SWITCH_TEAM"] / t, p["SWITCH_DEAD"] / t, 1 - params.get("hook", V.HOOK) * b)


def freeze(seeds, cfg):
    import joblib
    from . import m2_lib as L
    from .data import TUNE, idsse
    use_met = {"cA": 1, "cB": 0}[cfg]
    data = L.data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    ext = idsse.DATA / "external"
    met_f, met_l = joblib.load(ext / "tcn_feats_ext_matched2.pkl") if use_met else ([], [])
    for s in seeds:
        items = [{"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))], "y": T.labels5(data[k][i])}
                 for k in TUNE for i in range(2)]
        items += [{"src": "met", "feats": [ff], "y": ll} for ff, ll in zip(met_f, met_l)]
        steps = 1500
        mod = V.train4(items, seed=s, steps=steps)
        c = {"net": {"nin": 26, "c": 64, "p": 0.2}, "steps": steps, "seed": s, "crop": 300, "bs": 64, "dil": list(T.DIL),
             "trained_on": list(TUNE) + (["metrica1-3"] if use_met else []), "profile": "matched2+DR", "torch": torch.__version__,
             "bd_lambda": V.BD_LAMBDA, "ema": V.EMA, "config": cfg}
        torch.save({"state_dict": mod["net"].state_dict(), "mu": torch.tensor(mod["mu"]), "sd": torch.tensor(mod["sd"]),
                    "config": json.dumps(c)}, HERE / "models" / f"tcn_v4_s{s}.pt")
        print("saved seed", s, flush=True)


def write_params(cfg, seeds=(10, 11, 12, 13, 14), steps=1500):
    p = {**load_params(HERE / "params_v3.json"), "seeds": list(seeds), "fusion": FUSION, "steps": steps, "hook": V.HOOK,
         "config": cfg, "sources": "IDSSE TUNE (matched2 + 6 DR variants)" + (" + Metrica 1-3" if cfg == "cA" else "") + ", mirror aug + TTA, no SkillCorner",
         "selection": "TCN4 seed-ensemble fused 50/50 with v2.1 evidence, Viterbi with params_v2_1 costs scaled by (1 - 0.6 x boundary prob)"}
    (HERE / "params_v4.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "params":
        write_params(sys.argv[2])
    else:   # python -m possession.tcn_v4 CONFIG seed [seed ...]
        freeze(tuple(int(a) for a in sys.argv[2:]), sys.argv[1])
