"""A more realistic degrade, wrapping (not editing) possession.degrade. Applied on top of degraded halves
(run_idsse.load output); each perturbation is separate and seeded.

    ball_noise(half, mid, i, sigma=0.3)        Gaussian noise on the observed ball x, y (metres)
    dead_missing(half, mid, i, rate=0.7)       extra ball loss during dead stretches (truth == D) until the
                                               missing rate over dead frames is `rate` (gap chain, mean 2 s)
    wrong_team(half, mid, i, frac=0.02)        a fraction of tracklets moved to a free slot of the other team

Assumptions (not measured on the real tracker): noise is white and independent of speed; the extra dead-ball loss
uses the IDSSE truth to know which frames are dead (a simulation device only); a "tracklet" is a maximal run of
frames where one slot is occupied (back-to-back tracklets in one slot count as one run, so the count of runs is a
lower bound on tracklets) and a moved run keeps its positions, only the slot (hence team) changes.
"""
import zlib

import numpy as np

from . import degrade as D
from .data import pm


def _rng(mid, half, tag):
    return np.random.default_rng([zlib.crc32(mid.encode()), half, 2025, tag])


def ball_noise(half, mid, i, sigma=0.3):
    rng = _rng(mid, i, 1)
    ball = half["ball"].copy()
    ball += (rng.standard_normal(ball.shape) * sigma).astype(np.float32) * (~np.isnan(ball))
    return {**half, "ball": ball}


def dead_missing(half, mid, i, rate=0.7):
    rng = _rng(mid, i, 2)
    ball = half["ball"].copy()
    dead = half["truth"] == pm.D
    seen = ~np.isnan(ball[:, 0])
    miss = 1 - (seen & dead).sum() / max(1, dead.sum())       # current missing rate over dead frames
    q = (rate - miss) / (1 - miss) if miss < rate else 0.0     # extra hide probability among seen dead frames
    if q > 0:
        mean_off = D.BALL_GAP_S * D.FPS
        mean_on = mean_off * (1 - q) / q
        hide = ~D._alternating_runs(len(ball), mean_on, mean_off, rng)
        ball[hide & dead] = np.nan
    return {**half, "ball": ball}


def wrong_team(half, mid, i, frac=0.02):
    rng = _rng(mid, i, 3)
    xy = half["xy"].copy()
    team = half["team"]
    present = ~np.isnan(xy[:, :, 0])
    T, S = present.shape
    segs = []
    for s in range(S):
        p = np.r_[False, present[:, s], False].astype(np.int8)
        e = np.diff(p)
        segs += [(a, b, s) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    pick = rng.random(len(segs)) < frac
    moved = 0
    for (a, b, s), take in zip(segs, pick):
        if not take:
            continue
        other = np.flatnonzero(team != team[s])
        free = [o for o in other if not present[a:b, o].any()]
        if not free:
            continue
        o = free[rng.integers(len(free))]
        xy[a:b, o] = xy[a:b, s]
        xy[a:b, s] = np.nan
        present[a:b, o] = True
        present[a:b, s] = False
        moved += 1
    return {**half, "xy": xy}


PERTURB = {"ball_noise0.3": ball_noise, "dead_missing0.7": dead_missing, "wrong_team0.02": wrong_team}


def apply(name, halves, mid):
    return [PERTURB[name](h, mid, i) for i, h in enumerate(halves)]
