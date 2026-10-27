"""Print markdown tables for results/restarts.md from the restarts_*.json logs and run txt files."""
import json
import re
from pathlib import Path

import numpy as np

L = Path(__file__).with_name("logs")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
BASE = {"safety": "reference_safety", "rules": "rules_final"}
SIGMAS, SEEDS = ["0.5", "1.5"], [0, 1, 2]


def row(label, r):
    h = [r["held_out"][m]["spell"] for m in MATCHES]
    return (f"| {label} | {r['lomo_spell_mean']:.4f} | {r['lomo_spell_min']:.4f} | {r['lomo_spell_max']:.4f} | "
            + " | ".join(f"{x:.4f}" for x in h) + f" | {r['lomo_macro_mean']:.4f} |")


def cells(path):
    out = []
    for line in path.read_text().splitlines():
        mt = re.match(r"(\{.*?\}) ((?:[\d.]+ ){5})mean ([\d.]+)", line)
        if mt:
            out.append((json.loads(mt.group(1)), [float(x) for x in mt.group(2).split()], float(mt.group(3))))
    return out


def main():
    head = ("| variant | LOMO mean | min | max | " + " | ".join(MATCHES) + " | macroF1 |\n|---|---|---|---|"
            + "---|" * 5 + "---|")
    for base in BASE:
        b = json.loads((L / f"{BASE[base]}.json").read_text())
        for s in SIGMAS:
            print(f"\n### base {base}, jitter sigma {s} s\n\n{head}")
            print(row(f"base ({BASE[base]})", b))
            for seed in SEEDS:
                r = json.loads((L / f"restarts_{base}_s{s}_seed{seed}.json").read_text())
                print(row(f"+ restarts, seed {seed}", r))
            rs = [json.loads((L / f"restarts_{base}_s{s}_seed{k}.json").read_text()) for k in SEEDS]
            d = [x["lomo_spell_mean"] - b["lomo_spell_mean"] for x in rs]
            print(f"\nmean gain over base across seeds: {np.mean(d):+.4f} (per seed {', '.join(f'{x:+.4f}' for x in d)}); "
                  f"per-match gain seed 0: "
                  + ", ".join(f"{m} {rs[0]['held_out'][m]['spell'] - b['held_out'][m]['spell']:+.4f}" for m in MATCHES))
            print("tag stats seed 0:", rs[0]["tag_stats"])
            print("chosen params per held-out match (seed 0):",
                  {m: {k: v for k, v in rs[0]["held_out"][m]["params"].items()} for m in MATCHES})
            c = cells(L / f"restarts_run_s{s}_seed0.txt")
            if base == "rules":  # both bases print into the same txt: second half of the lines is rules
                c = c[len(c) // 2:]
            else:
                c = c[:len(c) // 2]
            print("\nin-sample grid cells (mean over 5 matches, seed 0; chosen on the same 5, so descriptive only):")
            for name, sel in [("base (W_D 0, W_T 0)", lambda p: p["W_D"] == 0 and p["W_T"] == 0),
                              ("D only (W_T 0)", lambda p: p["W_D"] > 0 and p["W_T"] == 0),
                              ("team only (W_D 0)", lambda p: p["W_D"] == 0 and p["W_T"] > 0),
                              ("both", lambda p: p["W_D"] > 0 and p["W_T"] > 0)]:
                best = max((x for x in c if sel(x[0])), key=lambda x: x[2])
                print(f"- {name}: best {best[2]:.4f} at {json.dumps(best[0])}  per match {best[1]}")


if __name__ == "__main__":
    main()
