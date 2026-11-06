"""Safety-net submission for BAR-ATH. Run by a human in their own terminal, never inside Claude Code.

    $env:MTAH_DATA_DIR = "D:\\mtah-data"
    idsse-tests\\.venv\\Scripts\\python -m possession.bar_ath --out evaluation\\results\\safety.csv
    idsse-tests\\.venv\\Scripts\\python evaluation\\validate_submission.py evaluation\\results\\safety.csv

Paste back only the printed aggregate lines (counts, validation verdict), never rows of the CSV.

Parameters come from possession/params.json (tuned on IDSSE only), else safety.PARAMS. Nothing in
this file is chosen by looking at BAR-ATH.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import safety
from .data import ROOT, pm
from .score import join

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
import mtah_guard  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402

FORCED_DEAD = (110808, 113741)  # graded, all D: second half kicks off at 113742 (CLAUDE.md)


def parse_raw(raw_path, T, S, pitch):
    """ATD raw text -> ball (T, 2), slots (T, S, 2), metres centred on the pitch. No guard: tested on
    synthetic files in tests/; load_atd() is the only caller that touches the competition data."""
    ball = np.full((T, 2), np.nan, np.float32)
    xy = np.full((T, S, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1  # ATD frame_id = video frame + 1
            b = parts[2].split(",")
            ball[t] = (float(b[0]), float(b[1]))
            vals = [float(v) for g in parts[1].split(";") for v in g.split(",")[:2]]
            xy[t] = np.array(vals, np.float32).reshape(S, 2)
    size = np.array(pitch, np.float32)
    return ball * size - size / 2, xy * size - size / 2


def load_atd():
    """Whole-video ball and slots in metres, plus the team of each slot."""
    mtah_guard.refuse_under_claude()
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    ball, xy = parse_raw(raw_path, source_frames("tactical"), len(team), GAMES[GAME]["pitch"])
    return ball, xy, team


def params():
    path = Path(__file__).with_name("params.json")
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    mtah_guard.refuse_under_claude()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=("v1", "v2", "v2_1", "v3", "v4", "v6", "v8"), default="v2",
                    help="v1 = safety net (params.json); v2 = rules + dead-ball model (params_v2.json); v2_1 = same with ball-feature model (params_v2_1.json); v3 = TCN + v2.1 fusion (params_v3.json, needs torch); v4 = TCN4 boundary-hook + v2.1 fusion (params_v4.json, needs torch); v6 = v4 with short-spell-weighted nets (params_v6.json, needs torch); v8 = v6 on touch-merged labels (params_v8.json, needs torch)")
    args = ap.parse_args()

    ball, xy, team = load_atd()
    labels = np.full(len(ball), pm.D, np.int8)  # ungraded frames are ignored by the scorer
    if args.method in ("v2", "v2_1"):
        from . import pipeline
        sfx = "" if args.method == "v2" else "_1"
        p = pipeline.load_params(Path(__file__).with_name(f"params_v2{sfx}.json"))
        models = pipeline.load_models(Path(__file__).with_name("models") / f"dead_v2{sfx}.joblib")
        predict = lambda half: pipeline.predict(half, p, models)  # noqa: E731
    elif args.method == "v3":
        from . import pipeline
        p, predict = pipeline.v3_predictor()
    elif args.method == "v4":
        from . import pipeline
        p, predict = pipeline.v4_predictor()
    elif args.method == "v6":
        from . import pipeline
        p, predict = pipeline.v6_predictor()
    elif args.method == "v8":
        from . import pipeline
        p, predict = pipeline.v8_predictor()
    else:
        p = params()
        predict = lambda half: safety.predict(half, p)  # noqa: E731
    for _, first, last in half_video_blocks(GAME):
        half = {"ball": ball[first:last + 1], "xy": xy[first:last + 1], "team": team}
        labels[first:last + 1] = predict(half)
    labels[FORCED_DEAD[0]:FORCED_DEAD[1] + 1] = pm.D

    letters = np.array(pm.LABELS)[labels]
    write_submission(args.out, letters)
    blocks = half_video_blocks(GAME)
    graded = join([labels[a:b + 1] for _, a, b in blocks])
    stats = pm.possession_stats(graded, pm.valid_runs(graded))
    seen = np.mean(np.concatenate([~np.isnan(ball[a:b + 1, 0]) for _, a, b in blocks]))
    print(f"method {args.method}  params {json.dumps(p) if p else 'defaults'}")
    print(f"rows {len(letters)}  graded H {np.sum(graded == pm.H)}  A {np.sum(graded == pm.A)}  D {np.sum(graded == pm.D)}")
    print(f"possessions {stats['n_spells']}  mean {stats['mean_duration_s']:.1f} s  ball seen (graded) {seen:.3f}")


if __name__ == "__main__":
    main()
