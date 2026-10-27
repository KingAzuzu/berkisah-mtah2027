"""Make clean IDSSE tracking look like Metrica's automatic tracking (ATD) of BAR-ATH.

Targets taken from DATA.md, and only from there:
    ball observed on 59.3% of graded frames                 -> BALL_SEEN
    20.5 tracked objects per graded frame, for 22 players   -> PLAYER_SEEN = 20.5 / 22
    46 track slots, slots appear, vanish and swap identity  -> N_SLOTS, tracklets reassigned to slots
    the ball goes missing disproportionately where it is crowded

Everything else here is an assumption, named in CAPS so it can be varied: how long a ball or player
gap lasts, how strongly crowding drives ball loss, how long a tracklet keeps its slot. The ATD does
not tell us, and we may not measure it on BAR-ATH.

Not modelled: a slot assigned to the wrong team, duplicated players, position noise. DATA.md says
nothing about how often they happen.
"""
import zlib

import numpy as np

BALL_SEEN = 0.593
PLAYER_SEEN = 20.5 / 22
N_SLOTS = 46
BALL_GAP_S = 2.0       # ASSUMPTION mean length of a ball-missing stretch
CROWD_RADIUS = 3.0     # metres
CROWD_GAIN = 0.5       # ASSUMPTION each extra player within CROWD_RADIUS adds this to the loss hazard
PLAYER_GAP_S = 1.0     # ASSUMPTION mean length of a player-missing stretch
TRACKLET_S = 20.0      # ASSUMPTION mean time a tracklet keeps its slot before the tracker re-acquires it
FPS = 25


def _rng(mid, half, seed):
    return np.random.default_rng([zlib.crc32(mid.encode()), half, seed])


def crowding(ball, xy):
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    return np.sum(d < CROWD_RADIUS, axis=1)


def _ball_mask(crowd, rng):
    """Two-state chain. Leave seen with hazard p0 * (1 + gain * crowd); leave missing at 1 / gap.
    p0 is solved by bisection so the seen fraction hits BALL_SEEN on this half."""
    q = 1.0 / (BALL_GAP_S * FPS)
    w = 1.0 + CROWD_GAIN * crowd
    u = rng.random(len(crowd))

    def run(p0):
        seen = np.empty(len(crowd), bool)
        s = True
        for t in range(len(crowd)):
            s = (u[t] >= p0 * w[t]) if s else (u[t] < q)
            seen[t] = s
        return seen

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if run(mid).mean() > BALL_SEEN:
            lo = mid
        else:
            hi = mid
    return run((lo + hi) / 2)


def _alternating_runs(T, mean_on, mean_off, rng):
    """Boolean mask of alternating geometric on/off runs, starting in the stationary state."""
    out = np.empty(T, bool)
    t, on = 0, rng.random() < mean_on / (mean_on + mean_off)
    while t < T:
        n = rng.geometric(1.0 / (mean_on if on else mean_off))
        out[t:t + n] = on
        t += n
        on = not on
    return out


def _tracklets(present, rng):
    """Split one player's presence into (start, stop) tracklets: at every gap, and at random within runs."""
    p = np.r_[False, present, False].astype(np.int8)
    edges = np.diff(p)
    out = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        while b - a > 0:
            n = rng.geometric(1.0 / (TRACKLET_S * FPS))
            out.append((a, min(b, a + n)))
            a += n
    return out


def degrade(half, mid, half_index, seed=0):
    """Degraded copy of a clean half from data.load_idsse(); truth is passed through untouched."""
    rng = _rng(mid, half_index, seed)
    T = len(half["ball"])
    xy, team = half["xy"], half["team"]

    ball = half["ball"].copy()
    ball[~_ball_mask(crowding(half["ball"], xy), rng)] = np.nan

    mean_off = PLAYER_GAP_S * FPS
    mean_on = mean_off * PLAYER_SEEN / (1 - PLAYER_SEEN)
    slots_per_team = N_SLOTS // 2
    out_xy = np.full((T, N_SLOTS, 2), np.nan, np.float32)
    out_team = np.repeat(np.array([0, 1], np.int8), slots_per_team)
    for k in (0, 1):
        pieces = []
        for j in np.flatnonzero(team == k):
            on_pitch = ~np.isnan(xy[:, j, 0])
            present = on_pitch & _alternating_runs(T, mean_on, mean_off, rng)
            pieces += [(a, b, j) for a, b in _tracklets(present, rng)]
        pieces.sort()
        free_at = np.zeros(slots_per_team, np.int64)  # frame from which each slot of this team is free
        for a, b, j in pieces:
            idle = np.flatnonzero(free_at <= a)
            if not len(idle):
                continue  # every slot of this team busy: the tracker drops this tracklet
            s = idle[rng.integers(len(idle))]
            out_xy[a:b, k * slots_per_team + s] = xy[a:b, j]
            free_at[s] = b
    return {"ball": ball, "xy": out_xy, "team": out_team, "truth": half["truth"]}


def degrade_match(halves, mid, seed=0):
    return [degrade(h, mid, i, seed) for i, h in enumerate(halves)]
