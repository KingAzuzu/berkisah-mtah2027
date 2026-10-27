"""Loader and shared helpers for the 7 open IDSSE matches (DFL XML, 25 fps).

Per match we build, for each half: ball xy / status / possession per frame, and for each team a
(T, P, 2) position array (NaN when a player is not on the pitch). Coordinates are metres, pitch
centred on (0, 0), 105 x 68. Parsed matches are cached as pickles under data/cache.
"""
import datetime as dt
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FPS = 25
HALF_LEN, HALF_WID = 52.5, 34.0
MATCHES = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
           "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
           "J03WR9": "DFL-COM-000002"}


def _path(kind, mid):
    return DATA / f"{kind}_{MATCHES[mid]}_DFL-MAT-{mid}.xml"


def _parse_meta(mid):
    root = ET.parse(_path("DFL_02_01_matchinformation", mid)).getroot()
    g = root.find(".//General").attrib
    gk, teams = set(), {}
    for t in root.iter("Team"):
        teams[t.get("Role")] = t.get("TeamId")
        gk |= {p.get("PersonId") for p in t.iter("Player") if p.get("PlayingPosition") == "TW"}
    return {"home": g["HomeTeamId"], "away": g["GuestTeamId"], "title": g["MatchTitle"]}, gk


def _parse_tracking(mid):
    """Stream the 400 MB file line by line; split('"') puts N,T,X,Y at 1,3,5,7 (ball flags last)."""
    sets, cur = {}, None
    hdr = re.compile(r'GameSection="(\w+)".*TeamId="([^"]+)" PersonId="([^"]+)"')
    with open(_path("DFL_04_03_positions_raw_observed", mid), encoding="utf-8") as f:
        for line in f:
            if line.startswith("<Frame "):
                if cur is not None:
                    p = line.split('"')
                    cur.append((int(p[1]), p[3], p[5], p[7], p[-2] if "BallStatus" in line else "", p[-4] if "BallStatus" in line else ""))
            elif line.startswith("<FrameSet"):
                sec, team, pid = hdr.search(line).groups()
                cur = [] if team != "referee" else None
                if cur is not None:
                    sets[(sec, team, pid)] = cur
    return sets


def load_match(mid):
    cache = DATA / "cache" / f"{mid}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    meta, gk_ids = _parse_meta(mid)
    sets = _parse_tracking(mid)
    halves = []
    for sec in ("firstHalf", "secondHalf"):
        ball = sets[(sec, "BALL", next(k[2] for k in sets if k[0] == sec and k[1] == "BALL"))]
        n0, n1 = ball[0][0], ball[-1][0]
        T = n1 - n0 + 1
        assert [r[0] for r in ball] == list(range(n0, n1 + 1)), "ball frames not contiguous"
        h = {"name": sec, "n0": n0, "T": T,
             "t0": dt.datetime.fromisoformat(ball[0][1]),
             "t_last": dt.datetime.fromisoformat(ball[-1][1]),
             "ball": np.array([[float(r[2]), float(r[3])] for r in ball], dtype=np.float32),
             "status": np.array([int(r[4]) for r in ball], dtype=np.int8),
             "poss": np.array([int(r[5]) for r in ball], dtype=np.int8), "teams": {}}
        for tid in (meta["home"], meta["away"]):
            pids = sorted(k[2] for k in sets if k[0] == sec and k[1] == tid)
            xy = np.full((T, len(pids), 2), np.nan, np.float32)
            for j, pid in enumerate(pids):
                for r in sets[(sec, tid, pid)]:
                    i = r[0] - n0
                    if 0 <= i < T and r[2] and r[3]:
                        xy[i, j] = (float(r[2]), float(r[3]))
            is_gk = np.array([p in gk_ids for p in pids])
            h["teams"][tid] = {"ids": pids, "gk": is_gk, "xy": xy}
        # Attack direction from the goalkeeper's mean x: GK on the left => team attacks +x.
        for tid, t in h["teams"].items():
            gx = np.nanmean(t["xy"][:, t["gk"], 0])
            t["sign"] = 1 if gx < 0 else -1
        s = [t["sign"] for t in h["teams"].values()]
        assert s[0] == -s[1], f"{mid} {sec}: inconsistent attack direction {s}"
        halves.append(h)
    m = {"id": mid, **meta, "halves": halves}
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(m, protocol=4))
    return m


def load_events(mid):
    """Corners (taker team, EventTime) and shots (team, CalculatedTimestamp else EventTime), as UTC datetimes."""
    out = {"corners": [], "shots": []}
    for ev in ET.parse(_path("DFL_03_02_events_raw", mid)).getroot().iter("Event"):
        def when(attr="EventTime"):
            return dt.datetime.fromisoformat(ev.get(attr)).astimezone(dt.timezone.utc)
        for ch in ev:
            if ch.tag == "CornerKick":
                out["corners"].append({"team": ch.get("Team"), "time": when(), "side": ch.get("Side")})
            elif ch.tag == "ShotAtGoal":
                out["shots"].append({"team": ch.get("Team"), "time": when("CalculatedTimestamp" if ev.get("CalculatedTimestamp") else "EventTime")})
    return out


def to_frame(m, time):
    """UTC datetime -> (half index, frame index within half), or None. Tracking T stamps are UTC, 25 fps."""
    for hi, h in enumerate(m["halves"]):
        i = round((time - h["t0"].astimezone(dt.timezone.utc)).total_seconds() * FPS)
        if 0 <= i < h["T"]:
            return hi, i
    return None


def other_team(m, tid):
    return m["away"] if tid == m["home"] else m["home"]


def team_xy(h, tid, flip_to):
    """Positions of `tid` in the frame where team `flip_to` attacks +x (one 180-degree rotation per half)."""
    s = h["teams"][flip_to]["sign"]
    return h["teams"][tid]["xy"] * s


def ball_xy(h, flip_to):
    return h["ball"] * h["teams"][flip_to]["sign"]
