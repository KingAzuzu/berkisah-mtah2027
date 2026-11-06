"""v8 = v6 recipe (TCN4 boundary head, short-spell loss x3, seeds 20-24, v4 decoder) with nets trained on touch-merged labels:
an alive spell < 2 s directly between two spells of the other team is relabelled as that team (v8_convention.merge, K 2), following
CHALLENGE.md (an uncontrolled touch / deflection is not a change of possession). Use only if the organiser confirms that convention.
Inference: python -m possession.bar_ath --method v8 (CPU, float32, 1 thread, deterministic).
Freeze (dev, GPU, IDSSE TUNE only): python -m possession.tcn_v8 freeze <seed> [...]  then  python -m possession.tcn_v8 params
"""
import json
from pathlib import Path

import torch

from . import tcn_lib as T, tcn_v4, v4_lib as V

HERE = Path(__file__).parent
SHORT_W = 3.0
SEEDS = (20, 21, 22, 23, 24)


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v8.json").read_text())


def load_nets(seeds=None, params=None):
    params = params or load_params()
    nets = []
    for s in (seeds if seeds is not None else params["seeds"]):
        ck = torch.load(HERE / "models" / f"tcn_v8_s{s}.pt", map_location="cpu", weights_only=True)
        net = V.TCN4(**json.loads(ck["config"])["net"])
        net.load_state_dict(ck["state_dict"])
        nets.append({"net": net.float().eval(), "mu": ck["mu"].numpy(), "sd": ck["sd"].numpy()})
    return nets


predict = tcn_v4.predict   # identical fusion + decoder


def freeze(seeds):
    import joblib
    from . import m2_lib as L
    from .data import TUNE, idsse
    from .v8_convention import merge
    data = L.data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    for s in seeds:
        items = [{"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))], "y": T.labels5(dict(data[k][i], truth=merge(data[k][i]["truth"], 2)))}
                 for k in TUNE for i in range(2)]
        mod = V.train4(items, seed=s, steps=1500, short_w=SHORT_W)
        c = {"net": {"nin": 26, "c": 64, "p": 0.2}, "steps": 1500, "seed": s, "crop": 300, "bs": 64, "dil": list(T.DIL),
             "trained_on": list(TUNE), "profile": "matched2+DR", "torch": torch.__version__, "bd_lambda": V.BD_LAMBDA,
             "ema": V.EMA, "config": "cB", "short_w": SHORT_W, "labels": "merged K2"}
        torch.save({"state_dict": mod["net"].state_dict(), "mu": torch.tensor(mod["mu"]), "sd": torch.tensor(mod["sd"]),
                    "config": json.dumps(c)}, HERE / "models" / f"tcn_v8_s{s}.pt")
        print("saved seed", s, flush=True)


def write_params():
    p = {**tcn_v4.load_params(), "seeds": list(SEEDS), "short_w": SHORT_W, "config": "cB+short3x+mergedK2", "labels": "touch-merged K 2 s",
         "sources": "IDSSE TUNE (matched2 + 6 DR variants), mirror aug + TTA, no SkillCorner, no Metrica",
         "selection": "v6 recipe on touch-merged labels; LOMO vs merged truth 0.7563 (5/5 vs v6+merge 0.7310); vs DFL truth 0.6517"}
    (HERE / "params_v8.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "params":
        write_params()
    else:
        freeze(tuple(int(a) for a in sys.argv[2:]))
