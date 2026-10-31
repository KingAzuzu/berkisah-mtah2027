"""'Matched' degrade profile: IDSSE made to reproduce the aggregates measured on the real ATD tracking
(possession.atd_stats, pasted by the human as aggregates only; no competition data is read here).

Targets (graded frames, 25 fps): ball seen 0.593; ball-seen runs median 13.8 s, mean 25.3; ball-missing runs
median 10.7 s, mean 17.2 (p10 0.47 s, p99 105 s); 20.5 objects/frame; slot tracklets mean 107 s, median 35 s;
ball jitter ~1-2 cm; 2.7% of seen ball more than 1 m outside the pitch.

Ball: alternating seen / missing runs, lognormal (missing = mixture with a short-gap component), candidates drawn
until the seen share is close to 0.593; among the candidates the one whose missing frames are most crowded
(players within 3 m of the true ball) wins -- run lengths are untouched by this choice.
Players: each IDSSE player alternates tracklets (lognormal) and gaps (lognormal); tracklets go to free slots of
their team as in possession.degrade. Ball: Gaussian jitter BALL_JITTER_M.
The old assumed profile (possession.degrade) is untouched.
"""
import zlib

import numpy as np

from . import degrade as D
from .data import HALF_LEN, HALF_WID

FPS = D.FPS
BALL_SEEN = 0.593
BASE_SEEN = 0.80          # seen share of the renewal process before dead / outside hiding
P_DEAD, P_OUT = 0.5, 0.6  # chance a whole dead-ball (truth D) / outside-pitch episode is hidden
SEEN_MED_S, SEEN_SIG = 13.8, 1.15
MISS_MED_S, MISS_SIG = 9.0, 0.75
MISS_SHORT_P, MISS_SHORT_MED_S, MISS_SHORT_SIG = 0.12, 0.4, 0.6
SEEN_TOL = 0.04
N_CAND = 24
TRK_MED_S, TRK_SIG = 35.0, 1.75
GAP_MED_S, GAP_SIG = 3.5, 1.4
BALL_JITTER_M = 0.01
OUT_FRAC = 0.0              # extra outside-pitch ball, added only if IDSSE does not already give ~2.7% (see md)


def _rng(mid, half, seed):
    return np.random.default_rng([zlib.crc32(mid.encode()), half, seed, 4242])


def _ln(rng, med_s, sig):
    return np.maximum(1, np.round(med_s * FPS * np.exp(sig * rng.standard_normal())).astype(np.int64))


def _miss_len(rng):
    if rng.random() < MISS_SHORT_P:
        return _ln(rng, MISS_SHORT_MED_S, MISS_SHORT_SIG)
    return _ln(rng, MISS_MED_S, MISS_SIG)


def _ball_candidate(T, rng):
    seen = np.empty(T, bool)
    t, on = 0, rng.random() < BASE_SEEN
    while t < T:
        n = _ln(rng, SEEN_MED_S, SEEN_SIG) if on else _miss_len(rng)
        seen[t:t + n] = on
        t += n
        on = not on
    return seen


def _tracks(on_pitch, rng):
    """Alternate tracklets and gaps over one player's on-pitch frames; yields (a, b)."""
    T = len(on_pitch)
    out, t = [], 0
    present = rng.random() < 0.9
    while t < T:
        n = _ln(rng, TRK_MED_S, TRK_SIG) if present else _ln(rng, GAP_MED_S, GAP_SIG)
        if present:
            seg = on_pitch[t:t + n]
            e = np.diff(np.r_[False, seg, False].astype(np.int8))
            out += [(t + a, t + b) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
        t += n
        present = not present
    return out


def _hide(half, rng):
    """Dead-ball and outside-pitch episodes the tracker loses (episode level). Uses IDSSE truth: simulation only."""
    ball, dead = half["ball"], half["truth"] == 2
    out = (np.abs(ball[:, 0]) > HALF_LEN + 1) | (np.abs(ball[:, 1]) > HALF_WID + 1)
    hide = np.zeros(len(ball), bool)
    for mask, p in ((dead, P_DEAD), (out, P_OUT)):
        e = np.diff(np.r_[False, mask, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if rng.random() < p:
                hide[a:b] = True
    return hide


def _pick_mask(half, crowd, rng):
    T = len(crowd)
    hide = _hide(half, rng)
    cands, tries = [], 0
    while len(cands) < N_CAND and tries < N_CAND * 200:
        tries += 1
        m = _ball_candidate(T, rng) & ~hide
        if abs(m.mean() - BALL_SEEN) <= SEEN_TOL:
            cands.append(m)
    if not cands:
        return _ball_candidate(T, rng) & ~hide
    return max(cands, key=lambda m: crowd[~m].mean())


def degrade(half, mid, half_index, seed=0):
    rng = _rng(mid, half_index, seed)
    T = len(half["ball"])
    xy, team = half["xy"], half["team"]
    ball = half["ball"].copy()
    ball[~_pick_mask(half, D.crowding(half["ball"], xy), rng)] = np.nan
    ball += (rng.standard_normal(ball.shape) * BALL_JITTER_M).astype(np.float32)

    slots = D.N_SLOTS // 2
    out_xy = np.full((T, D.N_SLOTS, 2), np.nan, np.float32)
    out_team = np.repeat(np.array([0, 1], np.int8), slots)
    for k in (0, 1):
        pieces = []
        for j in np.flatnonzero(team == k):
            pieces += [(a, b, j) for a, b in _tracks(~np.isnan(xy[:, j, 0]), rng)]
        pieces.sort()
        free_at = np.zeros(slots, np.int64)
        for a, b, j in pieces:
            idle = np.flatnonzero(free_at <= a)
            if not len(idle):
                continue
            s = idle[rng.integers(len(idle))]
            out_xy[a:b, k * slots + s] = xy[a:b, j]
            free_at[s] = b
    return {"ball": ball, "xy": out_xy, "team": out_team, "truth": half["truth"]}


def degrade_match(halves, mid, seed=0):
    return [degrade(h, mid, i, seed) for i, h in enumerate(halves)]
