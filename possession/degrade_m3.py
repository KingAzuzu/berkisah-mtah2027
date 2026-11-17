"""Profile 'matched3': IDSSE made to look like the provided ATD, with the dead ball fixed.

Why (HANDOVER 17 Nov, five public competitor repos found it independently, and our own public-window check showed the symptom):
DFL freezes the ball at exactly zero speed while play is dead. In the provided tracking the dead ball is missing about 65-70% of the
time and, when seen, is being carried / rolled to the restart spot (several m/s). matched2 hid whole stoppages w.p. 0.63 and left
the other 37% as a frozen, visible ball, so every model learned "visible still ball = dead, moving ball = alive", and real
stoppages with a carried ball read as alive (v10 public D recall 0.33).

What changes vs matched2 (players / tracklets are generated the same way, constants in CANON):
  1. Stoppage ball path. With prob move_p the frozen DFL ball of a stoppage is replaced by: a decelerating roll from where play
     stopped, a carry by the nearest player, a pass / toss to the restart spot, and a short rest there (placed ball).
     The first stoppage of a half (before kick-off) and a stoppage running to the end of the half keep the DFL ball.
  2. Ball smoothing: Gaussian, ball_sigma frames (ATD ball is smooth: live acceleration p50/p90/p99 6.6/18.7/33.8 m/s^2 per a
     public competitor's measurement; raw DFL 19.8/97.6/604).
  3. Visibility split by state. Dead: a share of stoppages hidden end to end, the rest alternate seen / missing runs; edges are
     jittered around the label change so the hide is not a perfect label cue. Alive: a seen / missing renewal process tuned to
     P(missing | alive) = live_miss, the most crowded of N_CAND candidates kept (as matched2). Outside-pitch alive episodes are
     hidden w.p. out_hide. Targets from our own label-free check (ball_loss_check, v2 D proxy): P(missing|D) 0.702,
     P(missing|alive) 0.101.
  4. Optional: short missing runs filled by linear interpolation (fill_gap_s, the tracker's own gap filling), wrong-team
     tracklets (swap_q).
variant(k) draws the constants from RANGES for domain randomisation; k = 0 is CANON.
Simulation only: uses IDSSE truth to know where stoppages are. Nothing here reads competition data."""
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import degrade as D
from .data import FPS, HALF_LEN, HALF_WID

CANON = {
    "ball_sigma": 6.0, "move_p": 0.85, "roll_s": (0.4, 1.5), "rest_s": (0.4, 2.5), "pass_v": 14.0, "carry_off_m": 0.6,
    "hop_s": 4.0,
    "dead_whole_hide": 0.25, "dead_seen_s": 6.0, "dead_miss_s": 8.0, "edge_jit_s": 1.0,
    "live_miss": 0.10, "live_gap_med_s": 8.0, "live_gap_sig": 0.9, "live_short_p": 0.3, "live_short_med_s": 0.3,
    "live_seen_sig": 1.15, "out_hide": 0.3,
    "fill_gap_s": 0.0, "jitter_m": 0.001, "swap_q": 0.0, "trk_med_s": 35.0, "trk_sig": 1.75, "gap_med_s": 3.5, "gap_sig": 1.4,
}
RANGES = {  # domain randomisation, uniform
    "ball_sigma": (3.0, 9.0), "move_p": (0.6, 1.0), "pass_v": (8.0, 20.0), "hop_s": (2.0, 10.0), "dead_whole_hide": (0.15, 0.6),
    "dead_seen_s": (3.0, 10.0), "dead_miss_s": (4.0, 14.0), "edge_jit_s": (0.3, 2.0), "live_miss": (0.05, 0.13),
    "live_gap_med_s": (3.0, 10.0), "live_short_p": (0.1, 0.5), "out_hide": (0.1, 0.6), "fill_gap_s": (0.0, 3.0),
    "jitter_m": (0.0005, 0.004), "swap_q": (0.0, 0.04), "trk_med_s": (20.0, 60.0), "gap_med_s": (2.0, 6.0),
}
N_CAND, LIVE_TOL = 16, 0.03
N_SLOTS = D.N_SLOTS


def variant(k):
    if k == 0:
        return dict(CANON)
    rng = np.random.default_rng([4141, k])
    return {**CANON, **{n: float(rng.uniform(lo, hi)) for n, (lo, hi) in sorted(RANGES.items())}}


def _rng(mid, half, seed):
    return np.random.default_rng([zlib.crc32(mid.encode()), half, seed, 4343])


def _ln(rng, med_s, sig):
    return max(1, int(round(med_s * FPS * np.exp(sig * rng.standard_normal()))))


def runs(mask):
    e = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


# --- 1. stoppage ball path ---------------------------------------------------------------------------

def _track(xy, j, t0, n, fallback):
    car = xy[t0:t0 + n, j].astype(float)
    ok = np.isfinite(car[:, 0])
    if ok.all():
        return car
    if ok.sum() < 2:
        return np.repeat(fallback[None], n, 0)
    idx = np.arange(n)
    return np.stack([np.interp(idx, idx[ok], car[ok, 0]), np.interp(idx, idx[ok], car[ok, 1])], 1)


def _carry(ball0, p1, xy, t0, n, rng, cfg):
    """n frames from ball0 to p1: carried by a nearby player, handed / passed on to other nearby players every ~hop_s,
    then passed to the restart spot. Pass duration = distance / pass_v (at least 0.3 s)."""
    out = np.empty((n, 2))
    if not np.isfinite(xy[t0, :, 0]).any():
        out[:] = ball0 + (p1 - ball0) * np.linspace(0, 1, n)[:, None]
        return out
    pos, t = ball0.astype(float), 0
    pdur = lambda a, b: max(int(0.3 * FPS), int(np.ceil(np.linalg.norm(b - a) / cfg["pass_v"] * FPS)))
    while True:
        rem = n - t
        if rem <= pdur(pos, p1) + int(0.5 * FPS):                       # last pass to the restart spot
            out[t:] = pos + (p1 - pos) * np.linspace(0, 1, rem + 1)[1:, None]
            return out
        d = np.linalg.norm(xy[t0 + t] - pos, axis=1)
        cand = np.flatnonzero(np.isfinite(d))
        cand = cand[np.argsort(d[cand])][:3]
        j = int(cand[rng.integers(len(cand))])
        hp = pdur(pos, xy[t0 + t, j]) if d[j] > 2 else 0                # pass to the new carrier first
        hold = max(int(0.5 * FPS), int(rng.exponential(cfg["hop_s"]) * FPS))
        seg = min(rem - pdur(pos, p1) - int(0.5 * FPS), hp + hold)
        if seg <= 0:
            out[t:] = pos + (p1 - pos) * np.linspace(0, 1, rem + 1)[1:, None]
            return out
        off = rng.standard_normal(2)
        off = off / (np.linalg.norm(off) + 1e-9) * cfg["carry_off_m"]
        car = _track(xy, j, t0 + t, seg, pos) + off
        if hp:
            k = min(hp, seg)
            out[t:t + k] = pos + (car[k - 1] - pos) * np.linspace(0, 1, k + 1)[1:, None]
            out[t + k:t + seg] = car[k:]
        else:
            w = np.clip(np.arange(seg) / 8.0, 0, 1)[:, None]
            out[t:t + seg] = (1 - w) * pos + w * car
        pos, t = out[t + seg - 1].copy(), t + seg


def dead_path(ball, xy, a, b, rng, cfg):
    """(b - a, 2) moving ball for the stoppage [a, b): roll, carry, pass, rest at the restart spot ball[b]."""
    L = b - a
    p0, p1 = ball[a].astype(float), ball[b].astype(float)
    v0 = (ball[a - 1] - ball[max(0, a - 6)]).astype(float) * FPS / 5 if a >= 6 else np.zeros(2)
    sp = np.linalg.norm(v0)
    if sp > 25:
        v0 *= 25 / sp
    nr = min(int(rng.uniform(*cfg["roll_s"]) * FPS), L // 3)
    nrest = min(int(rng.uniform(*cfg["rest_s"]) * FPS), L // 3)
    nc = L - nr - nrest
    path = np.empty((L, 2))
    tau = 0.6 * FPS
    t = np.arange(1, nr + 1)
    path[:nr] = p0 + (v0 / FPS) * (tau * (1 - np.exp(-t / tau)))[:, None]
    pr = path[nr - 1] if nr else p0
    if nc > 0:
        path[nr:nr + nc] = _carry(pr, p1, xy, a + nr, nc, rng, cfg)
    path[nr + nc:] = p1
    return path


def move_dead_ball(ball, xy, dead, rng, cfg):
    out = ball.astype(float).copy()
    T = len(ball)
    for a, b in runs(dead):
        if a == 0 or b >= T or b - a < 3 or rng.random() >= cfg["move_p"]:
            continue
        out[a:b] = dead_path(ball, xy, a, b, rng, cfg)
    return out


# --- 3. visibility -----------------------------------------------------------------------------------

def _jit(rng, cfg):
    return int(round(rng.normal(0, cfg["edge_jit_s"]) * FPS))


def dead_hide(dead, rng, cfg):
    T = len(dead)
    hide = np.zeros(T, bool)
    for a, b in runs(dead):
        a2, b2 = max(0, a + _jit(rng, cfg)), min(T, b + _jit(rng, cfg))
        if b2 <= a2:
            continue
        if rng.random() < cfg["dead_whole_hide"]:
            hide[a2:b2] = True
            continue
        ms, mm = cfg["dead_seen_s"], cfg["dead_miss_s"]
        t, on = a2, rng.random() < mm / (ms + mm)          # on = hidden
        while t < b2:
            n = max(1, int(rng.exponential(mm if on else ms) * FPS))
            hide[t:min(b2, t + n)] = on
            t += n
            on = not on
    return hide


def _gap(rng, cfg):
    if rng.random() < cfg["live_short_p"]:
        return _ln(rng, cfg["live_short_med_s"], 0.6)
    return _ln(rng, cfg["live_gap_med_s"], cfg["live_gap_sig"])


def _renewal(T, rng, cfg, seen_med_s):
    seen = np.empty(T, bool)
    t, on = 0, rng.random() < 0.9
    while t < T:
        n = _ln(rng, seen_med_s, cfg["live_seen_sig"]) if on else _gap(rng, cfg)
        seen[t:t + n] = on
        t += n
        on = not on
    return seen


def live_mask(alive, crowd, rng, cfg):
    """Seen mask for the alive frames: renewal process with P(missing | alive) ~ live_miss, most crowded candidate."""
    p = cfg["live_miss"]
    q = cfg["live_short_p"]
    gap_mean = (q * cfg["live_short_med_s"] * np.exp(0.18)
                + (1 - q) * cfg["live_gap_med_s"] * np.exp(cfg["live_gap_sig"] ** 2 / 2))
    seen_med = gap_mean * (1 - p) / p / np.exp(cfg["live_seen_sig"] ** 2 / 2)
    T = len(alive)
    cands = []
    for _ in range(N_CAND * 20):
        m = _renewal(T, rng, cfg, seen_med)
        if abs((~m[alive]).mean() - p) <= LIVE_TOL:
            cands.append(m)
            if len(cands) == N_CAND:
                break
    if not cands:
        return _renewal(T, rng, cfg, seen_med)
    return max(cands, key=lambda m: crowd[~m & alive].mean() if (~m & alive).any() else 0)


def fill_short(ball, seen, n_max):
    """Missing runs of at most n_max frames with seen frames on both sides become seen, linearly interpolated."""
    if n_max < 1:
        return seen
    seen = seen.copy()
    for a, b in runs(~seen):
        if a > 0 and b < len(seen) and b - a <= n_max:
            w = np.linspace(0, 1, b - a + 2)[1:-1, None]
            ball[a:b] = (1 - w) * ball[a - 1] + w * ball[b]
            seen[a:b] = True
    return seen


# --- players (as degrade_matched, constants from cfg) ------------------------------------------------

def _tracks(on_pitch, rng, cfg):
    T = len(on_pitch)
    out, t = [], 0
    present = rng.random() < 0.9
    while t < T:
        n = _ln(rng, cfg["trk_med_s"], cfg["trk_sig"]) if present else _ln(rng, cfg["gap_med_s"], cfg["gap_sig"])
        if present:
            out += [(t + a, t + b) for a, b in runs(on_pitch[t:t + n])]
        t += n
        present = not present
    return out


def players(xy, team, rng, cfg):
    T = len(xy)
    slots = N_SLOTS // 2
    out_xy = np.full((T, N_SLOTS, 2), np.nan, np.float32)
    out_team = np.repeat(np.array([0, 1], np.int8), slots)
    pieces = {0: [], 1: []}
    for k in (0, 1):
        for j in np.flatnonzero(team == k):
            for a, b in _tracks(~np.isnan(xy[:, j, 0]), rng, cfg):
                kk = 1 - k if rng.random() < cfg["swap_q"] else k      # tracklet labelled with the wrong team
                pieces[kk].append((a, b, j))
    for k in (0, 1):
        free_at = np.zeros(slots, np.int64)
        for a, b, j in sorted(pieces[k]):
            idle = np.flatnonzero(free_at <= a)
            if not len(idle):
                continue
            s = idle[rng.integers(len(idle))]
            out_xy[a:b, k * slots + s] = xy[a:b, j]
            free_at[s] = b
    return out_xy, out_team


# --- entry points ------------------------------------------------------------------------------------

def degrade(half, mid, half_index, seed=0, cfg=None):
    cfg = cfg or CANON
    rng = _rng(mid, half_index, seed)
    xy, team, truth = half["xy"], half["team"], half["truth"]
    dead = truth == 2
    true_ball = move_dead_ball(half["ball"], xy, dead, rng, cfg)
    if cfg["ball_sigma"] > 0:
        true_ball = gaussian_filter1d(true_ball, cfg["ball_sigma"], axis=0, mode="nearest")
    out = (np.abs(true_ball[:, 0]) > HALF_LEN + 1) | (np.abs(true_ball[:, 1]) > HALF_WID + 1)
    hide = dead_hide(dead, rng, cfg)
    for a, b in runs(out & ~dead):
        if rng.random() < cfg["out_hide"]:
            hide[a:b] = True
    crowd = D.crowding(true_ball.astype(np.float32), xy)
    seen = live_mask(~dead, crowd, rng, cfg) & ~hide
    ball = true_ball.copy()
    seen = fill_short(ball, seen, int(cfg["fill_gap_s"] * FPS))
    ball[~seen] = np.nan
    ball += rng.standard_normal(ball.shape) * cfg["jitter_m"]
    oxy, oteam = players(xy, team, rng, cfg)
    return {"ball": ball.astype(np.float32), "xy": oxy, "team": oteam, "truth": truth}


def degrade_match(halves, mid, seed=0, cfg=None):
    return [degrade(h, mid, i, seed, cfg) for i, h in enumerate(halves)]
