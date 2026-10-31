"""Profile 'matched2': like degrade_matched but ball-missing conditional on truth dead / alive matches the real
tracking (P(missing|dead)=0.70, P(missing|alive)=0.10, overall seen ~0.593). degrade_matched is untouched.
Differences: renewal process mostly seen (BASE_SEEN up), dead-ball episodes hidden w.p. P_DEAD, outside-pitch
episodes only when alive-and-outside are hidden w.p. P_OUT. Players and jitter as in degrade_matched."""
import numpy as np

from . import degrade as D, degrade_matched as M
from .data import HALF_LEN, HALF_WID

BASE_SEEN = 0.93
P_DEAD, P_OUT = 0.63, 0.3
SEEN_TOL = 0.04
N_CAND = 24
M_SEEN_MED_S, M_SEEN_SIG = 32.0, 1.15
M_MISS_MED_S, M_MISS_SIG = 4.0, 0.75


def _cand(T, rng):
    seen = np.empty(T, bool)
    t, on = 0, rng.random() < BASE_SEEN
    while t < T:
        n = M._ln(rng, M_SEEN_MED_S, M_SEEN_SIG) if on else M._ln(rng, M_MISS_MED_S, M_MISS_SIG)
        seen[t:t + n] = on
        t += n
        on = not on
    return seen


def _hide(half, rng):
    ball, dead = half["ball"], half["truth"] == 2
    out = (np.abs(ball[:, 0]) > HALF_LEN + 1) | (np.abs(ball[:, 1]) > HALF_WID + 1)
    hide = np.zeros(len(ball), bool)
    for mask, p in ((dead, P_DEAD), (out & ~dead, P_OUT)):
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
        m = _cand(T, rng) & ~hide
        if abs(m.mean() - M.BALL_SEEN) <= SEEN_TOL:
            cands.append(m)
    if not cands:
        return _cand(T, rng) & ~hide
    return max(cands, key=lambda m: crowd[~m].mean())


def degrade(half, mid, half_index, seed=0):
    old = M._pick_mask
    M._pick_mask = _pick_mask
    try:
        return M.degrade(half, mid, half_index, seed)
    finally:
        M._pick_mask = old


def degrade_match(halves, mid, seed=0):
    return [degrade(h, mid, i, seed) for i, h in enumerate(halves)]
