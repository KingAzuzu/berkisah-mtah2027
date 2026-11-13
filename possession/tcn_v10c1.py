"""v10c1 = v10 (hedge, v6 + v8 recipe, equal weight, v4 decoder) with the 10 nets retrained on all 7 IDSSE matches
(TUNE + LOCKED). Recipe, seeds, steps and decoder unchanged; only the training set grows. Allowed after the one-time locked check
(logs/locked_v10.json, 13 Nov 2026): no parameter is chosen from the locked matches. The dead-ball model (dead_v2_1) is unchanged.
No held-out score exists for v10c1 by construction; it is kept only if the sanity check passes (see HANDOVER).
Freeze (dev, GPU): python -m possession.tcn_v10c1 freeze  then  python -m possession.tcn_v10c1 params
Inference: python -m possession.bar_ath --method v10c1
"""
import json
from pathlib import Path

import torch

from . import tcn_lib as T, tcn_v4, tcn_v10, v4_lib as V

HERE = Path(__file__).parent
SEEDS = (20, 21, 22, 23, 24)
SHORT_W = 3.0
predict = tcn_v4.predict


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v10c1.json").read_text())


def _load(name, s):
    ck = torch.load(HERE / "models" / f"tcn_{name}_s{s}.pt", map_location="cpu", weights_only=True)
    net = V.TCN4(**json.loads(ck["config"])["net"])
    net.load_state_dict(ck["state_dict"])
    return {"net": net.float().eval(), "mu": ck["mu"].numpy(), "sd": ck["sd"].numpy()}


def load_nets(params=None):
    return [_load("v6c1", s) for s in SEEDS] + [_load("v8c1", s) for s in SEEDS]


def features():
    """-> {match: (canonical matched2 feats per half, [6 DR variants of feats per half], halves)} for all 7 matches."""
    import joblib
    from . import degrade_matched2 as DM2, m2_lib as L, v4_prep as P
    from .data import LOCKED, TUNE, idsse, load_idsse
    cache = idsse.DATA / "cache" / "tcn_feats_locked.pkl"
    if not cache.exists():
        out = {}
        for m in LOCKED:
            hv = load_idsse(m)
            d = DM2.degrade_match(hv, m, 0)
            out[m] = ([T.frame_features(h) for h in d], [[T.frame_features(h) for h in P.degrade_variant(hv, m, k)] for k in range(P.K)], d)
            print("features", m, flush=True)
        joblib.dump(out, cache)
    out = joblib.load(cache)
    data = L.data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    for m in TUNE:
        out[m] = (canon[m], dr[m], data[m])
    return out


def freeze():
    from .v8_convention import merge
    F = features()
    for name, lab in (("v6c1", lambda h: T.labels5(h)), ("v8c1", lambda h: T.labels5(dict(h, truth=merge(h["truth"], 2))))):
        for s in SEEDS:
            dst = HERE / "models" / f"tcn_{name}_s{s}.pt"
            if dst.exists():
                continue
            items = [{"src": "idsse", "feats": [c[i]] + [dr[v][i] for v in range(len(dr))], "y": lab(d[i])}
                     for c, dr, d in F.values() for i in range(2)]
            mod = V.train4(items, seed=s, steps=1500, short_w=SHORT_W)
            c = {"net": {"nin": 26, "c": 64, "p": 0.2}, "steps": 1500, "seed": s, "crop": 300, "bs": 64, "dil": list(T.DIL),
                 "trained_on": sorted(F), "profile": "matched2+DR", "torch": torch.__version__, "bd_lambda": V.BD_LAMBDA,
                 "ema": V.EMA, "config": "cB", "short_w": SHORT_W, "labels": "DFL" if name == "v6c1" else "merged K2"}
            torch.save({"state_dict": mod["net"].state_dict(), "mu": torch.tensor(mod["mu"]), "sd": torch.tensor(mod["sd"]),
                        "config": json.dumps(c)}, dst)
            print("saved", name, s, flush=True)


def write_params():
    p = {**tcn_v10.load_params(), "nets": "v6c1 s20-24 + v8c1 s20-24 (equal weight), trained on 7 IDSSE matches",
         "config": "hedge v6+v8, 7-match retrain"}
    (HERE / "params_v10c1.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    import sys
    write_params() if sys.argv[1] == "params" else freeze()
