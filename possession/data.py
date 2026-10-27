"""The one representation every method in this package reads, and the IDSSE loader that fills it.

A match is a list of halves. Each half is a dict:

    ball   (T, 2) float32, metres, pitch centred on (0, 0). NaN where the ball is not observed.
    xy     (T, S, 2) float32, metres. One column per track slot, NaN where the slot is empty.
    team   (S,) int8, 0 = home, 1 = away. Fixed per slot, as in Metrica's ATD.
    truth  (T,) int8 in possession_metrics codes (H 0, A 1, D 2), or None when there is none.

IDSSE truth is DFL BallStatus (0 = dead -> D) and BallPossession (1 = home -> H, 2 = away -> A).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "idsse-tests"))
sys.path.insert(0, str(ROOT / "evaluation"))
import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
HALF_LEN, HALF_WID = idsse.HALF_LEN, idsse.HALF_WID
LOCKED = ("J03WQQ", "J03WR9")  # final check only: never tune on these
TUNE = tuple(m for m in idsse.MATCHES if m not in LOCKED)


def truth_codes(status, poss):
    out = np.full(len(status), pm.D, np.int8)
    alive = status != 0
    out[alive & (poss == 1)] = pm.H
    out[alive & (poss == 2)] = pm.A
    return out


def load_idsse(mid):
    """Clean IDSSE match: ball always seen, one slot per player, truth attached."""
    m = idsse.load_match(mid)
    halves = []
    for h in m["halves"]:
        home, away = h["teams"][m["home"]]["xy"], h["teams"][m["away"]]["xy"]
        halves.append({
            "ball": h["ball"].copy(),
            "xy": np.concatenate([home, away], axis=1),
            "team": np.r_[np.zeros(home.shape[1], np.int8), np.ones(away.shape[1], np.int8)],
            "truth": truth_codes(h["status"], h["poss"]),
        })
    return halves
