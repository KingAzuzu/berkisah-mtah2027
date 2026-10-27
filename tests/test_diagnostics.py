"""Tests for possession/atd_stats.py and possession/camera_centre.py on synthetic files only.

    idsse-tests\\.venv\\Scripts\\python tests\\test_diagnostics.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "tests"))
from possession import atd_stats, camera_centre  # noqa: E402
from possession.bar_ath import parse_raw  # noqa: E402
from possession.run_idsse import load  # noqa: E402
from games import asset_paths  # noqa: E402
from load_homography import MATRIX_COLUMNS, load as load_h  # noqa: E402
from test_bar_ath import write_atd  # noqa: E402

PITCH = (105.3, 68.0)
T = 6000
BLOCKS = [(1, 500, 2999), (2, 3500, 5999)]
FPS = 25


def synthetic_inputs():
    """First T frames of degraded IDSSE (ball gaps, tracklet slots) for the ATD side."""
    h = load("J03WMX")[0]
    return h["ball"][:T].copy(), h["xy"][:T].copy(), h["team"]


def make_homographies(T, rng):
    """Slowly varying perspective maps from normalised pitch to 1920x1080 pixels."""
    base = np.array([[1700.0, -200.0, 100.0], [60.0, 650.0, 150.0], [0.0, -0.15, 1.0]])
    H = np.repeat(base[None], T, axis=0)
    H[:, 0, 2] += rng.normal(0, 30, T).cumsum() * 0.1
    H[:, 1, 2] += rng.normal(0, 10, T).cumsum() * 0.1
    return H


def write_homography(path, H, valid, err):
    with open(path, "w") as f:
        f.write("frame," + ",".join(MATRIX_COLUMNS) + ",reprojection_error,valid\n")
        for t in range(len(H)):
            m = ",".join("nan" if not valid[t] else f"{v:.10g}" for v in H[t].ravel())
            f.write(f"{t},{m},{err[t]:.4f},{int(valid[t])}\n")


def screen_centre_m(H):
    """Pitch position in metres (centred) under pixel (960, 540), per frame."""
    p = np.linalg.solve(H, np.tile([960.0, 540.0, 1.0], (len(H), 1))[:, :, None])[:, :, 0]
    return (p[:, :2] / p[:, 2:3]) * np.array(PITCH) - np.array(PITCH) / 2


def test_atd_stats():
    ball, xy, team = synthetic_inputs()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "tactical"))
        raw = asset_paths(data_dir=d)["atd_raw"]
        write_atd(raw, ball, xy)
        b2, xy2 = parse_raw(raw, T, xy.shape[1], PITCH)
    st = atd_stats.compute_stats(b2, xy2, team, BLOCKS, PITCH)

    graded = np.zeros(T, bool)
    for _, a, b in BLOCKS:
        graded[a:b + 1] = True
    seen = ~np.isnan(ball[:, 0])
    assert st["graded_frames"] == graded.sum()
    assert abs(st["ball_seen_fraction"] - seen[graded].mean()) < 1e-3
    # run lengths: seen + missing frames across runs add back to the graded frames
    sr, mr = st["ball_seen_run_s"], st["ball_missing_run_s"]
    assert abs(sr["mean"] * sr["count"] + mr["mean"] * mr["count"] - graded.sum() / FPS) < 0.05
    assert 0 < st["objects_per_frame"]["all"]["mean"] <= xy.shape[1]
    assert abs(st["objects_per_frame"]["home"]["mean"] + st["objects_per_frame"]["away"]["mean"]
               - st["objects_per_frame"]["all"]["mean"]) < 1e-2
    assert 0 < st["slots_ever_active"] <= st["slots_total"] == 46
    assert st["slot_tracklet_s"]["count"] >= st["slots_ever_active"]
    assert 0 <= st["share_speed_over_40ms"] < 0.05 and 0 <= st["share_seen_outside_pitch_1m"] < 0.05
    assert st["ball_speed_ms_nearly_stationary"]["p50"] < 1.0
    text = atd_stats.format_report(st)
    assert len(text.splitlines()) < 20
    print("ok   atd_stats on synthetic ATD: seen", st["ball_seen_fraction"], "objects",
          st["objects_per_frame"]["all"]["mean"], "slots", st["slots_ever_active"])

    # injected faults must show up: one teleport and one ball far outside the pitch
    b3 = b2.copy()
    i = int(np.flatnonzero(~np.isnan(b3[:, 0]) & graded)[10])
    b3[i, 0] += 80.0
    s3 = atd_stats.compute_stats(b3, xy2, team, BLOCKS, PITCH)
    assert s3["share_speed_over_40ms"] > st["share_speed_over_40ms"]
    assert s3["share_seen_outside_pitch_1m"] >= st["share_seen_outside_pitch_1m"]
    print("ok   atd_stats flags an injected 80 m jump")


def test_camera_centre():
    ball, _, _ = synthetic_inputs()
    rng = np.random.default_rng(0)
    H = make_homographies(T, rng)
    valid = np.ones(T, bool)
    valid[1000:1200] = False
    err = rng.uniform(0.1, 0.5, T)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "tactical"))
        write_homography(asset_paths(data_dir=d)["homography"], H, valid, err)
        Hl, _, vl = load_h(data_dir=d)
    assert np.array_equal(vl, valid) and np.isnan(Hl[1100]).all()

    # ball sits exactly at the projected screen centre on every frame -> distance ~ 0
    centre = np.zeros((T, 2), np.float32)
    ok = valid
    centre[ok] = screen_centre_m(H[ok])
    centre[~ok] = np.nan
    dist, bx = camera_centre.centre_distances(Hl, vl, centre, BLOCKS, PITCH)
    graded_valid = sum(int(valid[a:b + 1].sum()) for _, a, b in BLOCKS)
    assert len(dist) == graded_valid, (len(dist), graded_valid)
    assert dist.max() < 0.01, dist.max()  # float32 ball storage
    st = camera_centre.summarise(dist, bx, PITCH)
    assert st["pass"] is True and st["median_m"] < 0.01
    print("ok   camera centre: ball at screen centre -> max distance", round(float(dist.max()), 5), "m")

    # ball offset by a known 12 m in x -> median 12 m, bar FAILS; missing ball frames are skipped
    off = centre.copy()
    off[:, 0] += 12.0
    off[2000:2100] = np.nan
    dist2, bx2 = camera_centre.centre_distances(Hl, vl, off, BLOCKS, PITCH)
    st2 = camera_centre.summarise(dist2, bx2, PITCH)
    assert abs(st2["median_m"] - 12.0) < 0.05 and st2["pass"] is False
    assert len(dist2) == graded_valid - 100
    assert sum(st2["frames_by_ball_third"].values()) == len(dist2)
    print("ok   camera centre: 12 m offset -> median", st2["median_m"], "FAIL as expected;",
          camera_centre.format_report(st2).splitlines()[-1])

    # random synthetic ball (IDSSE degraded) gives a finite, sane report
    dist3, bx3 = camera_centre.centre_distances(Hl, vl, ball.astype(np.float32), BLOCKS, PITCH)
    st3 = camera_centre.summarise(dist3, bx3, PITCH)
    assert np.isfinite(st3["median_m"]) and st3["p25_m"] <= st3["median_m"] <= st3["p75_m"] <= st3["p90_m"]
    print("ok   camera centre on degraded IDSSE ball vs synthetic camera: median", st3["median_m"], "m")


def test_guards():
    env = {**os.environ, "CLAUDECODE": "1"}
    for mod in ("possession.atd_stats", "possession.camera_centre"):
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "Refusing" in r.stderr and not r.stdout, (mod, r.stdout, r.stderr)
        print(f"ok   {mod} refuses under CLAUDECODE")
    # the loaders refuse on their own too, not only main()
    os.environ["CLAUDECODE"] = "1"
    for fn in (atd_stats.load_inputs, camera_centre.load_inputs):
        try:
            fn()
        except SystemExit as e:
            assert "Refusing" in str(e)
        else:
            raise AssertionError(fn)
    print("ok   both load_inputs() refuse under CLAUDECODE")


if __name__ == "__main__":
    test_atd_stats()
    test_camera_centre()
    test_guards()
