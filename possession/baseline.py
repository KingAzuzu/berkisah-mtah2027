"""The organiser's baseline (nearest tracked player, centred majority vote) on our representation.

Same algorithm as starter/baseline.py, and its smooth() is imported rather than copied, so the only
difference is where the positions come from. Used as a calibration anchor: on BAR-ATH the published
baseline_smooth50 scores spellF1 0.2433, so degraded IDSSE should land near that if degrade.py is
realistic.
"""
import sys

import numpy as np

from .data import ROOT, pm

sys.path.insert(0, str(ROOT / "starter"))
from baseline import NOBODY, smooth  # noqa: E402


def nearest_team(half):
    """0 / 1 = team of the slot nearest the ball, NOBODY where the ball or every slot is missing."""
    d = np.linalg.norm(half["xy"] - half["ball"][:, None, :], axis=2)
    ok = ~np.all(np.isnan(d), axis=1)
    out = np.full(len(d), NOBODY, np.int8)
    out[ok] = half["team"][np.nanargmin(d[ok], axis=1)]
    return out


def predict(half, window=50):
    return pm.label_codes(smooth(nearest_team(half), window))
