"""Set-piece restart evidence (HANDOVER idea 4 and the Smart Tagging restarts).

A restart ends a dead-ball stretch and starts a possession by a known team. Three pieces:

  throw_team(half, frame, ...)   who throws a throw-in: the player standing at or beyond the touchline
                                 (nearest the ball when it is seen), from tracking only. Smart Tagging
                                 gives no team for throw-ins, so this fills it.
  simulate_tags(...)             imperfect automatic tags from DFL events (IDSSE only). Every parameter of
                                 the imperfection is an explicit argument; see SIM_ASSUMPTIONS.
  adjust(e, tags, p)             evidence adjustment: in a window before the tag the D score is raised, from
                                 the tag on the tagged team's score is raised for a short window.

Evidence arrays follow the PLAN_refine contract: (T, 3) log-scores for H, A, D.
"""
import datetime as dt
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from .data import FPS, HALF_WID, idsse, pm

RULE = {"W": 25, "MARGIN": 1.0}   # frames searched either side of the tag, metres inside the line still counted
SIM_ASSUMPTIONS = {
    "tag_source": "DFL EventTime of ThrowIn, CornerKick, GoalKick, FreeKick, KickOff",
    "timing": "Gaussian jitter on the tag frame, sigma in seconds (variants 0.5 and 1.5)",
    "dropped": "each tag dropped independently with probability DROP (0.15)",
    "team": "corner, goal kick, free kick, kick-off keep the true team; throw-in team removed then "
            "filled by throw_team() on the jittered frame (None when no candidate)",
    "not_modelled": "false-positive tags, wrong team on non-throw-in tags, interval (not instant) tags",
}
TYPES = ("ThrowIn", "CornerKick", "GoalKick", "FreeKick", "KickOff")


def fingerprint(half):
    return zlib.crc32(half["ball"].tobytes()) ^ zlib.crc32(half["xy"][:, :, 0].tobytes()) ^ len(half["ball"])


# ---------------------------------------------------------------- DFL events (IDSSE)

def load_restart_events(mid):
    """[{type, team (club id), time (UTC datetime)}] from the DFL event XML (IDSSE only)."""
    out = []
    for ev in ET.parse(idsse._path("DFL_03_02_events_raw", mid)).getroot().iter("Event"):
        t = dt.datetime.fromisoformat(ev.get("EventTime")).astimezone(dt.timezone.utc)
        for ch in ev:
            if ch.tag in TYPES:
                team = ch.get("Team")
                if team is None:
                    play = ch.find("Play")
                    team = play.get("Team") if play is not None else None
                out.append({"type": ch.tag, "team": team, "time": t})
    return out


def events_by_half(mid, m):
    """Events as [{type, team 0/1/None, half, frame}] with frames within the half (idsse.to_frame)."""
    out = []
    for e in load_restart_events(mid):
        hf = idsse.to_frame(m, e["time"])
        if hf is None:
            continue
        team = 0 if e["team"] == m["home"] else 1 if e["team"] == m["away"] else None
        out.append({"type": e["type"], "team": team, "half": hf[0], "frame": hf[1]})
    return out


# ---------------------------------------------------------------- throw-in team rule

def throw_team(half, frame, W=RULE["W"], margin=RULE["MARGIN"]):
    """Team (0 home, 1 away) of the player at/over the touchline near `frame`, or None.

    Candidates: slots whose |y| comes within `margin` m of the touchline at some frame in
    [frame-W, frame+W]. Each is taken at its most-outside frame. If the ball is seen in the window,
    the candidate nearest the ball (at the ball frame nearest in time) wins; otherwise the most
    outside one."""
    xy, ball, team = half["xy"], half["ball"], half["team"]
    T = len(xy)
    lo, hi = max(0, frame - W), min(T, frame + W + 1)
    if hi <= lo:
        return None
    seg = xy[lo:hi]
    out = np.abs(seg[:, :, 1]) - HALF_WID
    ok = np.isfinite(out)
    out = np.where(ok, out, -np.inf)
    best = out.max(axis=0)                       # (S,) most-outside value per slot
    cand = np.where(best >= -margin)[0]
    if len(cand) == 0:
        return None
    if len(cand) == 1:
        return int(team[cand[0]])
    b = ball[lo:hi]
    seen = np.where(np.isfinite(b[:, 0]))[0]
    if len(seen) == 0:
        return int(team[cand[np.argmax(best[cand])]])
    d = []
    for s in cand:
        k = int(np.argmax(out[:, s]))
        j = seen[np.argmin(np.abs(seen - k))]
        d.append(np.linalg.norm(seg[k, s] - b[j]))
    return int(team[cand[int(np.argmin(d))]])


# ---------------------------------------------------------------- simulated automatic tags

def simulate_tags(events, half_idx, half, mid, sigma_s, drop=0.15, seed=0, rule_w=None):
    """Imperfect tags for one half. events: events_by_half output. Returns [{type, frame, team|None, true_frame}].
    Throw-in team is always filled with throw_team() on the jittered frame (window rule_w, default
    max(RULE W, 2*sigma frames))."""
    rng = np.random.default_rng([zlib.crc32(mid.encode()), half_idx, seed, int(round(sigma_s * 100))])
    T = len(half["ball"])
    w = rule_w if rule_w is not None else max(RULE["W"], int(round(2 * sigma_s * FPS)))
    tags = []
    for e in sorted((e for e in events if e["half"] == half_idx), key=lambda e: e["frame"]):
        u, jit = rng.random(), rng.normal(0.0, sigma_s * FPS)   # draw both always: streams stay aligned
        if u < drop:
            continue
        f = int(np.clip(round(e["frame"] + jit), 0, T - 1))
        team = e["team"]
        if e["type"] == "ThrowIn":
            team = throw_team(half, f, W=w)
        tags.append({"type": e["type"], "frame": f, "team": team, "true_frame": e["frame"]})
    return tags


# ---------------------------------------------------------------- evidence adjustment

GRID_DEFAULT = {"PRE": 125, "W_D": 2.0, "POST": 50, "W_T": 1.0}


def adjust(e, tags, p):
    """Copy of evidence e with: D += W_D on [t-PRE, t), tagged team += W_T on [t, t+POST)."""
    e = e.copy()
    T = len(e)
    pre, post, wd, wt = int(p["PRE"]), int(p["POST"]), p["W_D"], p["W_T"]
    for g in tags:
        t = g["frame"]
        if wd:
            e[max(0, t - pre):t, pm.D] += wd
        if wt and g["team"] is not None:
            e[t:min(T, t + post), g["team"]] += wt
    return e
