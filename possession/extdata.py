"""External public tracking data (Metrica Sports sample games 1-3) in the IDSSE half-dict format, with truth derived
from the event stream. Other games only: no BAR-ATH data. Data lives in idsse-tests/data/external (gitignored).
half = {ball (T,2) m centred, xy (T,S,2), team (S,) 0/1, truth (T,) int8 H0/A1/D2}; 25 fps; y up."""
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .data import HALF_LEN, HALF_WID, idsse
import possession_metrics as pm  # noqa: E402

EXT = Path(idsse.DATA) / "external" / "metrica"
GAMES = ("MET1", "MET2", "MET3")
CONTROLLED = {"PASS", "SHOT", "SET PIECE", "RECOVERY", "CARRY"}


def _m(x, y):
    return np.stack([(x - 0.5) * 2 * HALF_LEN, -(y - 0.5) * 2 * HALF_WID], -1).astype(np.float32)


def _csv_track(g):
    cols = {}
    for side in ("Home", "Away"):
        df = pd.read_csv(EXT / f"Sample_Game_{g}_RawTrackingData_{side}_Team.csv", header=None, skiprows=3)
        cols[side] = df.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    h, a = cols["Home"], cols["Away"]
    period, frame = h[:, 0].astype(int), h[:, 1].astype(int)
    pos = {"Home": h[:, 3:-2], "Away": a[:, 3:-2]}
    ball = h[:, -2:]
    xyz = [np.stack([pos[s][:, 0::2], pos[s][:, 1::2]], -1) for s in ("Home", "Away")]
    xy = np.concatenate(xyz, 1)
    team = np.r_[np.zeros(xyz[0].shape[1]), np.ones(xyz[1].shape[1])].astype(np.int8)
    return period, frame, ball, xy, team


def _csv_events(g):
    e = pd.read_csv(EXT / f"Sample_Game_{g}_RawEventsData.csv")
    return [(0 if r.Team == "Home" else 1, r.Type, str(r.Subtype) if isinstance(r.Subtype, str) else "",
             int(r["Start Frame"]), int(r["End Frame"]), int(r.Period)) for _, r in e.iterrows()]


def _epts_track():
    s = (EXT / "Sample_Game_3_metadata.xml").read_text()
    team_of = dict(re.findall(r'<Player id="(\w+)" teamId="(\w+)"', s))
    ch = re.findall(r'<PlayerChannel channelId="x" id="(player\d+)_x" playerId="(\w+)"', s)
    pid = {c: p for c, p in ch}
    specs = []
    for m in re.finditer(r'<DataFormatSpecification endFrame="(\d+)" separator=":" startFrame="(\d+)">(.*?)</DataFormatSpecification>', s, re.S):
        specs.append((int(m.group(2)), int(m.group(1)),
                      [pid[c] for c in re.findall(r'playerChannelId="(player\d+)_x"', m.group(3))]))
    plist = sorted({p for _, _, ps in specs for p in ps}, key=lambda p: (team_of[p], p))
    slot = {p: i for i, p in enumerate(plist)}
    team = np.array([0 if team_of[p].endswith("A") else 1 for p in plist], np.int8)
    N = sum(len(ps) for _, _, ps in specs[:1])
    lines = (EXT / "Sample_Game_3_tracking.txt").read_text().splitlines()
    T = len(lines)
    xy = np.full((T, len(plist), 2), np.nan)
    ball = np.full((T, 2), np.nan)
    frame = np.arange(1, T + 1)
    for line in lines:
        f, rest = line.split(":", 1)
        f = int(f)
        pl_, b = rest.split(":")
        sp = next(ps for a, z, ps in specs if a <= f <= z)
        vals = [v.split(",") for v in pl_.split(";")]
        for p, v in zip(sp, vals):
            xy[f - 1, slot[p]] = [float(v[0]), float(v[1])]
        ball[f - 1] = [float(v) for v in b.split(",")]
    meta = re.findall(r'<ProviderParameter>\s*<Name>(first_half_end|second_half_start|second_half_end)</Name>\s*<Value>(\d+)</Value>', s)
    md = {k: int(v) for k, v in meta}
    period = np.where(frame <= md["first_half_end"], 1, 2)
    return period, frame, ball, xy, team, N


def _json_events():
    ev = json.load(open(EXT / "Sample_Game_3_events.json"))["data"]
    out = []
    for e in ev:
        sub = (e["subtypes"] or {}).get("name", "") if isinstance(e["subtypes"], dict) else ""
        out.append((0 if e["team"]["id"].endswith("A") else 1, e["type"]["name"], sub or "",
                    e["start"]["frame"], e["end"]["frame"], e["period"]))
    return out


def derive_truth(events, f0, f1, period):
    """Per-frame H0/A1/D2 for frames f0..f1 (inclusive, global frame ids) of one period from events.
    D: before the first restart, from BALL OUT / goal / foul / offside / hand ball until the next controlled event.
    H/A: team of the last controlled event (PASS/SHOT/SET PIECE/RECOVERY/CARRY); BALL LOST, CHALLENGE, aerial duels
    never change it, only a controlled event by the other team does (DFL: loose/contested stays with last team)."""
    T = f1 - f0 + 1
    seq = []
    for team, typ, sub, s_, en, per in events:
        if per != period:
            continue
        goal = typ == "SHOT" and "GOAL" in sub and "OFF" not in sub
        dead = (typ in ("BALL OUT", "FAULT RECEIVED", "CARD") or
                (typ == "BALL LOST" and any(k in sub for k in ("OFFSIDE", "HAND BALL", "END HALF"))) or
                (typ == "CHALLENGE" and "FAULT" in sub))
        if typ in CONTROLLED:
            seq.append((s_, 1, "ctrl", team))
        if goal:
            seq.append((max(en, s_), 2, "dead", team))
        elif dead:
            seq.append((s_, 0, "dead", team))
    seq.sort()
    truth = np.full(T, pm.D, np.int8)
    state, poss, last = "dead", None, 0
    for fr, _, kind, team in seq:
        i = int(np.clip(fr - f0, 0, T))
        if state == "alive":
            truth[last:i] = poss
        last = i
        if kind == "dead":
            state = "dead"
        else:
            state, poss = "alive", (pm.H if team == 0 else pm.A)
    if state == "alive":
        truth[last:T] = poss
    return truth


NAN_DEAD_S = 2.0


def _long_runs(mask, n):
    out = np.zeros(len(mask), bool)
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a >= n:
            out[a:z] = True
    return out


def load_game(gid):
    if gid == "MET3":
        period, frame, ball, xy, team, _ = _epts_track()
        events = _json_events()
    else:
        g = int(gid[-1])
        period, frame, ball, xy, team = _csv_track(g)
        events = _csv_events(g)
    halves = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        f0, f1 = frame[idx[0]], frame[idx[-1]]
        b = _m(ball[idx, 0], ball[idx, 1])
        gone = _long_runs(np.isnan(ball[idx, 0]), NAN_DEAD_S * 25)   # tracking ball status: ball absent >= 2 s = dead
        for j in range(2):   # fill short ball gaps (clean ground-truth-like tracking)
            ok = ~np.isnan(b[:, j])
            b[:, j] = np.interp(np.arange(len(b)), np.flatnonzero(ok), b[ok, j])
        halves.append({"ball": b, "xy": _m(xy[idx, :, 0], xy[idx, :, 1]), "team": team,
                       "truth": np.where(gone, pm.D, derive_truth(events, f0, f1, p)).astype(np.int8), "nball_nan": float(np.isnan(ball[idx, 0]).mean())})
    return halves


def spell_stats(halves):
    """Mean spell length (s) of H/A spells and D share over all frames."""
    t = np.concatenate([h["truth"] for h in halves])
    d = float((t == pm.D).mean())
    lens = []
    for h in halves:
        x = h["truth"]
        b = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
        lens += [(b[k + 1] - b[k]) / 25 for k in range(len(b) - 1) if x[b[k]] != pm.D]
    return {"dead_share": d, "mean_spell_s": float(np.mean(lens)), "n_spells": len(lens),
            "n_changes": int(sum(1 for h in halves for a, c in zip(h["truth"][:-1], h["truth"][1:]) if a != c and a != 2 and c != 2))}


if __name__ == "__main__":
    import joblib
    from .data import load_idsse, TUNE
    out = {}
    for g in GAMES:
        hv = load_game(g)
        print(g, [len(h["ball"]) for h in hv], "ball nan", [round(h["nball_nan"], 4) for h in hv], spell_stats(hv), flush=True)
        out[g] = hv
    joblib.dump(out, EXT.parent / "metrica_halves.pkl")
    for m in TUNE:
        print(m, spell_stats(load_idsse(m)), flush=True)
