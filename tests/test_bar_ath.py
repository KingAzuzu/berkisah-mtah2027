"""ATD parser round trip on a synthetic file built from degraded IDSSE, and the competition-data guard.

    idsse-tests\\.venv\\Scripts\\python tests\\test_bar_ath.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from possession import safety  # noqa: E402
from possession.bar_ath import parse_raw  # noqa: E402
from possession.run_idsse import load  # noqa: E402

PITCH = (105.3, 68.0)


def fmt(v):
    return "NaN" if np.isnan(v) else f"{v:.6f}"


def write_atd(path, ball, xy):
    size = np.array(PITCH)
    nb, nxy = (ball + size / 2) / size, (xy + size / 2) / size
    with open(path, "w") as f:
        for t in range(len(ball)):
            players = ";".join(f"{fmt(x)},{fmt(y)},NaN" for x, y in nxy[t])
            f.write(f"{t + 1}:{players}:{fmt(nb[t, 0])},{fmt(nb[t, 1])},NaN\n")


def test_round_trip():
    h = load("J03WMX")[0]
    T = 5000
    ball, xy = h["ball"][:T], h["xy"][:T]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "synthetic_atd.txt")
        write_atd(p, ball, xy)
        b2, xy2 = parse_raw(p, T, xy.shape[1], PITCH)
    assert np.array_equal(np.isnan(b2), np.isnan(ball)) and np.array_equal(np.isnan(xy2), np.isnan(xy))
    assert np.nanmax(np.abs(b2 - ball)) < 1e-3 and np.nanmax(np.abs(xy2 - xy)) < 1e-3
    # Compare features, not labels: 6-decimal rounding (~1e-4 m) can flip a near-tie in the Viterbi
    # and move one boundary (seen once: a 41-frame block), which says nothing about the parser.
    f1 = safety.features({"ball": ball, "xy": xy, "team": h["team"]})
    f2 = safety.features({"ball": b2, "xy": xy2, "team": h["team"]})
    for k in f1:
        x, y = np.asarray(f1[k], float), np.asarray(f2[k], float)
        assert np.array_equal(np.isnan(x), np.isnan(y)) and np.nanmax(np.abs(x - y)) < 1e-3, k
    print("ok   parse_raw round trip: positions and safety features agree to 1e-3")


def test_guard():
    env = {**os.environ, "CLAUDECODE": "1"}
    r = subprocess.run([sys.executable, "-m", "possession.bar_ath", "--out", os.devnull],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "Refusing" in r.stderr, r.stderr
    print("ok   bar_ath refuses under CLAUDECODE")


if __name__ == "__main__":
    test_round_trip()
    test_guard()
