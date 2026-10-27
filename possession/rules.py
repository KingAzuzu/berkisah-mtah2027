"""WS2: possession rules from the literature, as team evidence for safety.viterbi.

A team *gains* the ball only at a validated frame (Link & Hoernig 2017; Vidal-Codina 2022; databallpy):
  ball observed, nearest player within R, not a duel, and the ball shows a kick signature nearby
  (velocity-vector change >= BV m/s, or direction change >= BA deg) or is nearly still (< VSLOW m/s),
  or the signature cannot be computed (ball not seen around that frame; accepted when ALLOW_UNK).
Between validated frames (flight, ball gaps, duels, a ball merely passing a player) the last validated
team is held: strength W_HOLD when the validated team on both sides of the stretch is the same,
W_DIFF when it changes (the receiver's arrival then decides). The D column is exactly safety.emissions'.
Team columns carry the same dead-ball damping as safety.emissions.

    evidence(half, params, cache) -> (T, 3) log-scores for H, A, D        predict() -> codes
"""
import numpy as np

from . import safety
from .data import FPS, pm

PARAMS = {"R": 2.0, "BV": 3.0, "BA": 25.0, "K": 5, "KW": 10, "VSLOW": 1.5, "ALLOW_UNK": 1, "GAP": 25,
          "DUEL_R": 1.0, "W_VAL": 3.0, "W_HOLD": 0.5, "W_DIFF": 0.1}


def _fill_ball(ball, gap):
    """Linear interpolation over ball gaps of at most `gap` frames; NaN elsewhere."""
    T = len(ball)
    seen = ~np.isnan(ball[:, 0])
    out = np.full_like(ball, np.nan)
    if seen.sum() < 2:
        out[seen] = ball[seen]
        return out
    idx = np.arange(T)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, 10 * T)[::-1])[::-1]
    ok = (prev >= 0) & (nxt < T) & (nxt - prev <= gap + 1)
    for a in (0, 1):
        out[:, a] = np.where(ok, np.interp(idx, idx[seen], ball[seen, a]), np.nan)
    return out


def _window_any(x, w):
    """x (bool) dilated: any True within +-w frames."""
    c = np.r_[0, np.cumsum(x)]
    i = np.arange(len(x))
    return (c[np.minimum(len(x), i + w + 1)] - c[np.maximum(0, i - w)]) > 0


def kick_signature(ball, p):
    """-> (kick_near, known_near, slow): per-frame booleans from the gap-filled ball track."""
    b = _fill_ball(ball, int(p["GAP"]))
    k, T = int(p["K"]), len(b)
    pad = np.full((k, 2), np.nan)
    before = np.vstack([pad, b[:-k]]) if T > k else np.full_like(b, np.nan)
    after = np.vstack([b[k:], pad]) if T > k else np.full_like(b, np.nan)
    vb, va = (b - before) * FPS / k, (after - b) * FPS / k
    sb, sa = np.linalg.norm(vb, axis=1), np.linalg.norm(va, axis=1)
    with np.errstate(all="ignore"):
        dv = np.linalg.norm(va - vb, axis=1)
        cosang = np.sum(va * vb, axis=1) / (sa * sb)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        known = np.isfinite(dv)
        kick = known & ((dv >= p["BV"]) | ((ang >= p["BA"]) & (sa > 1.0) & (sb > 1.0)))
        slow = known & (np.fmin(sa, sb) < p["VSLOW"])
    return _window_any(kick, int(p["KW"])), _window_any(known, int(p["KW"])), slow


def _team_damp(f, p):
    """The dead-ball damping safety.emissions applies to both team columns."""
    s = np.nan_to_num(f["speed"], nan=p["V0"])
    dead = np.clip((p["V0"] - s) / p["VS"], -20, 20) + np.where(f["out"], p["OUT"], 0.0)
    return np.log(2.0) - np.logaddexp(0, dead)


def evidence(half, params=None, cache=None):
    p = {**safety.PARAMS, **PARAMS, **(params or {})}
    cache = cache if cache is not None else {}
    if "f" not in cache:
        cache["f"] = safety.features(half)
    f = cache["f"]
    ks = tuple(p[k] for k in ("GAP", "K", "KW", "BV", "BA", "VSLOW"))
    if cache.get("ks_key") != ks:
        cache["ks_key"], cache["ks"] = ks, kick_signature(half["ball"], p)
    kick_near, known_near, slow = cache["ks"]

    e = safety.emissions(f, p)  # D column and the dead damping come from here, untouched
    T = len(e)
    dh = np.nan_to_num(f["d_home"], nan=np.inf)
    da = np.nan_to_num(f["d_away"], nan=np.inf)
    dmin = np.minimum(dh, da)
    seen = ~np.isnan(half["ball"][:, 0])
    duel = (dh < p["DUEL_R"]) & (da < p["DUEL_R"])
    sig = kick_near | slow | (~known_near if p["ALLOW_UNK"] else False)
    val = seen & (dmin < p["R"]) & ~duel & sig
    team = np.where(da < dh, pm.A, pm.H)

    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(val, idx, -1))
    nxt = np.minimum.accumulate(np.where(val, idx, 10 * T)[::-1])[::-1]
    has_l, has_n = last >= 0, nxt < T
    lt = team[np.clip(last, 0, T - 1)]
    nt = team[np.clip(nxt, 0, T - 1)]
    held = np.where(has_l, lt, nt)                     # edges: whichever side exists
    strength = np.where(has_l & has_n & (lt != nt), p["W_DIFF"], p["W_HOLD"])
    strength = np.where(has_l | has_n, strength, 0.0)

    ev = np.zeros((T, 2))
    other = 1 - held
    hold = ~val
    ev[idx[hold], other[hold]] = -strength[hold]
    ev[idx[val], 1 - team[val]] = -p["W_VAL"]
    e[:, :pm.D] = ev + _team_damp(f, p)[:, None]
    return e


def predict(half, params=None, cache=None):
    p = {**safety.PARAMS, **PARAMS, **(params or {})}
    return safety.viterbi(evidence(half, p, cache), p["SWITCH_TEAM"], p["SWITCH_DEAD"])
