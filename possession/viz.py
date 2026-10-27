"""One-half data viewer as a self-contained local HTML file: pitch replay, label timeline, durations.

    python -m possession.viz --idsse J03WMX --half 0          # public IDSSE, degraded like the ATD
    python -m possession.viz --idsse J03WMX --half 0 --clean  # the same half before degrading
    python -m possession.viz --bar-ath --half 0               # human only, in your own terminal

Writes IDSSE pages to idsse-tests/data/viz/ and BAR-ATH pages to data/viz/ (both gitignored). Open the
file in a browser. The BAR-ATH page holds competition
data: keep it on this machine and never paste it, or any part of it, into an AI tool.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from . import safety
from .data import FPS, ROOT, load_idsse, pm

TEMPLATE = Path(__file__).with_name("viz_template.html")
OUT_DIRS = {"idsse": ROOT / "idsse-tests" / "data" / "viz",  # public data, next to the IDSSE files
            "bar_ath": ROOT / "data" / "viz"}                 # competition data stays beside the bundle
PITCH_FPS = 5  # replay frame rate; labels stay at 25 fps
NAN16 = -32768


def _rle(codes):
    """[[start, stop, code], ...] runs of a per-frame int array."""
    codes = np.asarray(codes)
    b = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts, stops = np.r_[0, b], np.r_[b, len(codes)]
    return [[int(a), int(z), int(codes[a])] for a, z in zip(starts, stops)]


def _b64_int16(a):
    q = np.where(np.isnan(a), NAN16, np.round(a * 10)).astype("<i2")  # decimetres
    return base64.b64encode(q.tobytes()).decode("ascii")


def payload(half, title, pitch, preds, source):
    step = FPS // PITCH_FPS
    xy = half["xy"][::step]
    occupied = ~np.all(np.isnan(xy[:, :, 0]), axis=0)  # drop slots never used in this half
    xy, team = xy[:, occupied], half["team"][occupied]
    tracks = {"truth": _rle(half["truth"])} if half.get("truth") is not None else {}
    tracks.update({name: _rle(p) for name, p in preds.items()})
    return {
        "title": title, "source": source, "fps": FPS, "pitchFps": PITCH_FPS, "T": len(half["ball"]),
        "pitch": list(pitch), "nSlots": int(xy.shape[1]), "team": team.astype(int).tolist(),
        "xy": _b64_int16(xy), "ball": _b64_int16(half["ball"][::step]),
        "ballSeen": _rle(~np.isnan(half["ball"][:, 0])),
        "tracks": tracks,
    }


def write(p, name, kind):
    out_dir = OUT_DIRS[kind]
    out_dir.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(p, separators=(",", ":")))
    out = out_dir / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    return out


def predict(half):
    params = json.loads(Path(__file__).with_name("params.json").read_text())
    return safety.predict(half, params)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--idsse", metavar="MATCH")
    src.add_argument("--bar-ath", action="store_true")
    ap.add_argument("--half", type=int, default=0, choices=(0, 1))
    ap.add_argument("--clean", action="store_true", help="IDSSE only: before degrading")
    args = ap.parse_args()

    if args.bar_ath:
        from .bar_ath import load_atd, mtah_guard
        from games import GAME, GAMES, half_video_blocks
        mtah_guard.refuse_under_claude()
        ball, xy, team = load_atd()
        _, a, b = half_video_blocks(GAME)[args.half]
        half = {"ball": ball[a:b + 1], "xy": xy[a:b + 1], "team": team, "truth": None}
        p = payload(half, f"BAR-ATH, half {args.half + 1}", GAMES[GAME]["pitch"],
                    {"safety net": predict(half)}, "Metrica ATD (tactical)")
        name = f"bar_ath_half{args.half + 1}"
    else:
        from .data import LOCKED
        from .run_idsse import load
        if args.idsse in LOCKED:
            raise SystemExit(f"{args.idsse} is locked for the final check; pick a tuning match")
        half = (load_idsse(args.idsse) if args.clean else load(args.idsse))[args.half]
        kind = "clean" if args.clean else "degraded"
        p = payload(half, f"IDSSE {args.idsse}, half {args.half + 1} ({kind})", (105.0, 68.0),
                    {"safety net": predict(half)}, "IDSSE / DFL, CC BY 4.0 (Bassek et al. 2025)")
        name = f"idsse_{args.idsse}_half{args.half + 1}_{kind}"
    out = write(p, name, "bar_ath" if args.bar_ath else "idsse")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
