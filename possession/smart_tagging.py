"""Guarded loader for the competition's Smart Tagging set pieces (restarts). Standard library only.

    from possession import smart_tagging
    rs = smart_tagging.load_restarts()            # run in your own terminal, never under Claude Code
    tags = smart_tagging.to_tags(rs)              # input format of restarts.adjust (team 0 home / 1 away / None)

Format (read from starter/load_smart_tagging.py): SportsCode XML, one <instance> per phase with
<code>, <start>, <end> in seconds from the first frame of the tactical video (video frame =
round(seconds * 25)) and <label><group>/<text> pairs. Codes are upper case ("SET PIECES"), groups are
Title Case ("Type", "Team"), values Title Case ("Throw-in"); all matched case-insensitively.
Team is "N/A" for every throw-in -> None. Barcelona -> H, Atletic Club -> A.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mtah_guard  # noqa: E402

FPS = 25
SET_PIECE_CODE = "set pieces"
TEAM = {"barcelona": "H", "atletic club": "A"}
TYPE = {"throw-in": "ThrowIn", "corner kick": "CornerKick", "goal kick": "GoalKick", "free kick": "FreeKick",
        "kick off (start)": "KickOff", "kick off (after goal)": "KickOff"}


def _norm(s):
    return (s or "").strip().lower()


def parse(path, fps=FPS):
    """Parse one SportsCode XML at `path` (no guard here: callers go through load_restarts)."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        if _norm(inst.findtext("code")) != SET_PIECE_CODE:
            continue
        tags = {_norm(lb.findtext("group")): (lb.findtext("text") or "").strip() for lb in inst.findall("label")}
        typ = tags.get("type", "")
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        out.append({
            "frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "type": TYPE.get(_norm(typ), typ),
            "team": TEAM.get(_norm(tags.get("team"))),     # N/A, empty or an unknown club -> None
            "half": tags.get("half", ""),
        })
    out.sort(key=lambda r: r["frame"])
    return out


def load_restarts(path=None, game=None, data_dir=None):
    """Restarts of the competition match: [{frame, end_frame, type, team 'H'/'A'/None, half}], by frame."""
    mtah_guard.refuse_under_claude()
    import games
    game = game or games.GAME
    if path is None:
        path = games.require(["smart_tagging"], game, data_dir)[0]
    return parse(path, games.GAMES[game]["fps"])


def to_tags(restarts):
    """-> [{type, frame, team 0 (H) / 1 (A) / None}] for restarts.adjust."""
    return [{"type": r["type"], "frame": r["frame"], "team": {"H": 0, "A": 1}.get(r["team"])} for r in restarts]
