"""Label-free check: does the Smart Tagging `Team` tag agree with our own v4 labels? Human only.

    python -m possession.tag_check evaluation\\results\\v4.csv

For each phase code: number of instances, share of frames inside the intervals that v4 calls D, and,
on the frames v4 calls H/A, the share whose team equals the tag's team. ~0.8+ means the tag names the
team in possession; ~0.2 means it names the other team (e.g. DEFENDING codes); ~0.5 means no signal.
Prints aggregates only.
"""
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from .smart_tagging import FPS, TEAM, _norm, mtah_guard


def parse_all(path, fps=FPS):
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {_norm(lb.findtext("group")): (lb.findtext("text") or "").strip() for lb in inst.findall("label")}
        out.append((_norm(inst.findtext("code")), int(round(float(inst.findtext("start")) * fps)),
                    int(round(float(inst.findtext("end")) * fps)), TEAM.get(_norm(tags.get("team")))))
    return out


def main(csv):
    mtah_guard.refuse_under_claude()
    import games
    path = games.require(["smart_tagging"], games.GAME)[0]
    lab = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=1, dtype=str)
    acc = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, frames, D frames, alive frames, agree
    for code, a, b, team in parse_all(path):
        seg = lab[max(a, 0):b + 1]
        seg = seg[seg != "X"]
        r = acc[code]
        r[0] += 1; r[1] += len(seg); r[2] += int(np.sum(seg == "D"))
        if team:
            alive = seg[(seg == "H") | (seg == "A")]
            r[3] += len(alive); r[4] += int(np.sum(alive == team))
    print(f"{'code':<32}{'n':>5}{'D share':>9}{'agree':>8}{'alive fr':>10}")
    for code, (n, fr, d, al, ag) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
        print(f"{code:<32}{n:>5}{d / max(fr, 1):>9.3f}{(ag / al if al else float('nan')):>8.3f}{al:>10}")


if __name__ == "__main__":
    main(sys.argv[1])
