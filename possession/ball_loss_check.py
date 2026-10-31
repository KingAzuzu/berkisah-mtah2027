"""Does the real tracker lose the ball mostly while it is dead? v2.1 assumes so (degrade_matched._hide).

Label-free check: share of frames with the ball missing, split by v2's own D / alive prediction (v2's
D comes from the players-only model, so it does not read ball visibility). Compare the simulator with
the real tracking:

    python -m possession.ball_loss_check --idsse            # simulator, here
    python -m possession.ball_loss_check --bar-ath CSV      # human only: CSV = the v2 submission

Prints aggregates only.
"""
import argparse

import numpy as np

from . import pipeline
from .data import TUNE, load_idsse, pm


def split(seen, labels):
    dead = labels == pm.D
    return float(np.mean(~seen[dead])), float(np.mean(~seen[~dead])), float(np.mean(dead))


def show(name, miss_d, miss_alive, d_share):
    print(f"{name:<28} P(missing | D) {miss_d:.3f}   P(missing | alive) {miss_alive:.3f}   D share {d_share:.3f}")


def idsse():
    from .degrade_matched import degrade_match
    p, models = pipeline.load_params(), pipeline.load_models()
    pred_s, pred_l, tr_l = [], [], []
    for mid in TUNE:
        for half in degrade_match(load_idsse(mid), mid, 0):
            seen = ~np.isnan(half["ball"][:, 0])
            graded = half["truth"] != pm.X
            pred_s.append(seen[graded]); pred_l.append(pipeline.predict(half, p, models)[graded])
            tr_l.append(half["truth"][graded])
    seen, pred, truth = map(np.concatenate, (pred_s, pred_l, tr_l))
    show("IDSSE matched, truth", *split(seen, truth))
    show("IDSSE matched, v2 pred", *split(seen, pred))


def bar_ath(csv):
    from .bar_ath import load_atd, mtah_guard
    from games import GAME, half_video_blocks
    mtah_guard.refuse_under_claude()
    ball, _, _ = load_atd()
    letters = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=1, dtype=str)
    labels = np.array([pm.LABELS.index(c) for c in letters])
    blocks = half_video_blocks(GAME)
    seen = np.concatenate([~np.isnan(ball[a:b + 1, 0]) for _, a, b in blocks])
    lab = np.concatenate([labels[a:b + 1] for _, a, b in blocks])
    show("real tracking, v2 pred", *split(seen, lab))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--idsse", action="store_true")
    g.add_argument("--bar-ath", metavar="CSV")
    a = ap.parse_args()
    idsse() if a.idsse else bar_ath(a.bar_ath)
