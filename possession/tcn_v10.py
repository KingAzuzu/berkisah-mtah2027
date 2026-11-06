"""v10 = hedge between label conventions: the 5 v6 nets (DFL-style labels) and the 5 v8 nets (touch-merged labels) in one ensemble
(mean log-probs and boundary probs over all 10 nets = 0.5/0.5 average of the two ensembles), v4/v6 decoder unchanged.
LOMO (IDSSE TUNE): vs DFL truth 0.6686, vs merged truth 0.7449; worst-case regret 0.0174 (v6 0.0465, v8 0.0343). results/v10_hedge_mbr_seeds.md.
Chosen after seeing both truths: a decision aid for when the convention is unknown, not a gated improvement. No training.
Inference: python -m possession.bar_ath --method v10"""
import json
from pathlib import Path

from . import tcn_v4, tcn_v6, tcn_v8

HERE = Path(__file__).parent
predict = tcn_v4.predict


def load_params(path=None):
    return json.loads(Path(path or HERE / "params_v10.json").read_text())


def load_nets(params=None):
    return tcn_v6.load_nets(seeds=(20, 21, 22, 23, 24)) + tcn_v8.load_nets(seeds=(20, 21, 22, 23, 24))


def write_params():
    p = {**tcn_v6.load_params(), "nets": "v6 s20-24 + v8 s20-24 (equal weight)", "config": "hedge v6+v8",
         "selection": "minimax regret over DFL and merged truth (0.0174); not a gated improvement"}
    p.pop("short_w", None)
    (HERE / "params_v10.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    write_params()
