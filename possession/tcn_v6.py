"""v6 = v4 architecture and decoder (TCN4 boundary head, seed ensemble + v2.1 evidence 50/50, hook-0.6 Viterbi), with the TCN4
nets trained with the loss upweighted x3 within +-1 s of a true H<->A change next to a spell shorter than 3 s (seeds 20-24).
Inference: python -m possession.bar_ath --method v6 (CPU, float32, 1 thread, deterministic).
Freeze (dev, GPU, IDSSE TUNE only): python -m possession.tcn_v6 freeze <seed> [...]  then  python -m possession.tcn_v6 params
"""
import json
from pathlib import Path

import torch

from . import tcn_lib as T, tcn_v4, v4_lib as V

HERE = Path(__file__).parent
SHORT_W = 3.0
SEEDS = (20, 21, 22, 23, 24)


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v6.json").read_text())


def load_nets(seeds=None, params=None):
    params = params or load_params()
    nets = []
    for s in (seeds if seeds is not None else params["seeds"]):
        ck = torch.load(HERE / "models" / f"tcn_v6_s{s}.pt", map_location="cpu", weights_only=True)
        net = V.TCN4(**json.loads(ck["config"])["net"])
        net.load_state_dict(ck["state_dict"])
        nets.append({"net": net.float().eval(), "mu": ck["mu"].numpy(), "sd": ck["sd"].numpy()})
    return nets


predict = tcn_v4.predict   # identical fusion + decoder


def freeze(seeds):
    import joblib
    from . import m2_lib as L
    from .data import TUNE, idsse
    data = L.data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    for s in seeds:
        items = [{"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))], "y": T.labels5(data[k][i])}
                 for k in TUNE for i in range(2)]
        mod = V.train4(items, seed=s, steps=1500, short_w=SHORT_W)
        c = {"net": {"nin": 26, "c": 64, "p": 0.2}, "steps": 1500, "seed": s, "crop": 300, "bs": 64, "dil": list(T.DIL),
             "trained_on": list(TUNE), "profile": "matched2+DR", "torch": torch.__version__, "bd_lambda": V.BD_LAMBDA,
             "ema": V.EMA, "config": "cB", "short_w": SHORT_W}
        torch.save({"state_dict": mod["net"].state_dict(), "mu": torch.tensor(mod["mu"]), "sd": torch.tensor(mod["sd"]),
                    "config": json.dumps(c)}, HERE / "models" / f"tcn_v6_s{s}.pt")
        print("saved seed", s, flush=True)


def write_params():
    p = {**tcn_v4.load_params(), "seeds": list(SEEDS), "short_w": SHORT_W, "config": "cB+short3x",
         "sources": "IDSSE TUNE (matched2 + 6 DR variants), mirror aug + TTA, no SkillCorner, no Metrica",
         "selection": "v4 recipe + loss x3 within +-1 s of H<->A change adjacent to a spell < 3 s; v4 decoder (E1 retune failed gate)"}
    (HERE / "params_v6.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "params":
        write_params()
    else:
        freeze(tuple(int(a) for a in sys.argv[2:]))
