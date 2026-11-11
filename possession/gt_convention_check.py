"""Which label convention does the ground truth follow? Human only: reads the PUBLIC ground-truth CSV, prints 5 aggregate numbers.

    python -m possession.gt_convention_check evaluation\\gt\\<public ground truth csv>

Purpose (organiser: "inspecting the provided data ... to better understand what counts as possession" is allowed; CHALLENGE.md:
use the public ground truth to understand how cases are labelled). It does NOT score or tune anything: it only counts how many
brief opponent touches the truth contains, to choose between frozen methods v6 / v8 / v10 by a rule fixed before looking.

Short X-Y-X = an H/A spell shorter than 2 s directly between two spells of the other team (no D in between): a touch that flips
the owner for a moment. Reference, IDSSE DFL truth, first 300 s of each of the 10 TUNE halves:
    spells 22-38, spells < 2 s 5-20 (never below 5), short X-Y-X 0-18 (median 6; 0 in 1 of 10 windows).
Rule (fixed 11 Nov 2026, before the count was seen):
    short X-Y-X == 0            -> touches do not flip the owner  -> v8
    short X-Y-X >= 3            -> touches flip like DFL           -> v6
    1 or 2                      -> unclear                         -> v10 (hedge)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mtah_guard  # noqa: E402

FPS = 25


def counts(lab):
    """lab: array of 'H'/'A'/'D' (graded frames only, in order). -> dict of aggregates."""
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(lab)]
    v = lab[st]
    sp = [j for j in range(len(v)) if v[j] in ("H", "A")]
    dur = np.array([(en[j] - st[j]) / FPS for j in sp])
    xyx = sum(1 for j in sp if 0 < j < len(v) - 1 and v[j - 1] == v[j + 1] and v[j - 1] in ("H", "A")
              and v[j - 1] != v[j] and en[j] - st[j] < 2 * FPS)
    return {"graded": len(lab), "spells": len(sp), "mean_s": float(dur.mean()) if len(dur) else 0.0,
            "min_s": float(dur.min()) if len(dur) else 0.0, "short": int((dur < 2).sum()), "xyx": int(xyx)}


def verdict(xyx):
    return "v8" if xyx == 0 else ("v6" if xyx >= 3 else "v10")


def main(path):
    mtah_guard.refuse_under_claude()
    lab = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1, dtype=str)
    lab = lab[lab != "X"]
    c = counts(lab)
    print(f"graded frames {c['graded']}; possessions {c['spells']}, mean {c['mean_s']:.1f} s, shortest {c['min_s']:.2f} s; "
          f"possessions < 2 s: {c['short']}; short X-Y-X touches: {c['xyx']}")
    print(f"IDSSE DFL reference (10 windows of 300 s): possessions < 2 s 5-20, short X-Y-X 0-18 (median 6)")
    print(f"rule fixed beforehand -> {verdict(c['xyx'])}")


if __name__ == "__main__":
    main(sys.argv[1])
