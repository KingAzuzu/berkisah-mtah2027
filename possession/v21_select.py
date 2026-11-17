"""v21 selection: applies the v21_plan gates in the order fixed beforehand, on whatever LOMO outputs exist, and writes the winner.

  REF = r2 nets + dead model trained on matched2 (ball features)       (= what v10 is, evaluated on matched3)
  A   = r2 nets + dead model trained on matched3 (ball)
  A'  = r2 nets + dead model trained on matched3 (players only)
  B   = r3 nets (matched3 + DR) + best dead so far
  C   = r3x (f32 features), D = r3xl (f32, long context)
  then the decoder: nested LOMO costs (v21_decoder) on the winner, same gate.
Gate for every step: mean spellF1 (matched3 evaluation) >= best + 0.01 and >= 4/5 match wins over the current best.
A candidate whose nets are missing (not trained yet) is skipped and reported as such.

python -m possession.v21_select [SEEDS] -> logs/v21/select.json, results/v21.md"""
import json
import sys

import numpy as np

from . import v21_decoder as Dc, v21_lomo as H
from .data import TUNE


def have(recipe, seeds):
    return recipe == "none" or all((H.LOG / "nets" / f"{recipe}_s{s}_{m}.npz").exists() for s in seeds for m in TUNE)


def evaluate(recipe, tsim, variant, seeds, pdead):
    out = {}
    for esim in ("m3", "m2"):
        nets = None if recipe == "none" else H.net_outputs(recipe, esim, seeds)
        out[esim] = H.metrics(H.decode(nets, pdead[(tsim, variant)][esim], esim), esim)
    return out


def main(seeds=(20, 21, 22)):
    pdead = H.dead_tables()
    log, rows = [], []
    best = {"name": "REF", "recipe": "r2", "tsim": "m2", "variant": "ball"}
    if not have("r2", seeds):
        sys.exit("r2 nets missing for seeds %s" % (seeds,))
    best["res"] = evaluate("r2", "m2", "ball", seeds, pdead)
    rows.append(("REF", "r2", "m2/ball", best["res"], None))

    def step(name, recipe, tsim, variant):
        nonlocal best
        if not have(recipe, seeds):
            rows.append((name, recipe, f"{tsim}/{variant}", None, "skipped: nets missing"))
            return
        res = evaluate(recipe, tsim, variant, seeds, pdead)
        g = H.gate(res["m3"], best["res"]["m3"])
        rows.append((name, recipe, f"{tsim}/{variant}", res, f"vs {best['name']}: {g['delta']:+.4f}, {g['wins']}/5 -> "
                     + ("PASS" if g["pass"] else "FAIL")))
        if g["pass"]:
            best = {"name": name, "recipe": recipe, "tsim": tsim, "variant": variant, "res": res}

    step("A", "r2", "m3", "ball")
    step("A'", "r2", "m3", "players")
    for name, recipe in (("B", "r3"), ("C", "r3x"), ("D", "r3xl")):
        step(name, recipe, best["tsim"] if best["tsim"] == "m3" else "m3", best["variant"])
    # decoder, nested, on the winner
    dec = Dc.run(best["recipe"], best["tsim"], best["variant"], seeds)
    wins = sum(a > b for a, b in zip(dec["nested_per_match"], dec["ref_per_match"]))
    dpass = dec["nested_mean"] >= dec["ref_mean"] + 0.01 and wins >= 4
    vals, cnt = np.unique(np.array(dec["picks"]), axis=0, return_counts=True)
    costs = vals[int(np.argmax(cnt))].tolist() if dpass else [20.0, 5.0, 5.0, 0.0]
    win = {k: best[k] for k in ("name", "recipe", "tsim", "variant")}
    win["decoder"] = {"st": costs[0], "sd_in": costs[1], "sd_out": costs[2], "same": costs[3], "nested": dec, "pass": dpass,
                      "wins": wins}
    json.dump({"winner": win, "seeds": list(seeds),
               "rows": [{"name": r[0], "recipe": r[1], "dead": r[2], "res": r[3], "gate": r[4]} for r in rows]},
              open(H.LOG / "select.json", "w"), indent=1)
    md = ["# v21 LOMO result (IDSSE TUNE, matched3 evaluation; gates from v21_plan.md)", "",
          f"Seeds {list(seeds)}. spellF1 vs DFL truth, mean over the 5 held-out matches [min, max]; matched2 column reported, not gated.", "",
          "| step | nets | dead model | matched3 spellF1 | macroF1 | D recall (ball seen / missing) | matched2 spellF1 | gate |",
          "|---|---|---|---|---|---|---|---|"]
    for name, recipe, dm, res, g in rows:
        if res is None:
            md.append(f"| {name} | {recipe} | {dm} | | | | | {g} |")
            continue
        a, b = res["m3"], res["m2"]
        md.append(f"| {name} | {recipe} | {dm} | {a['spell']:.4f} [{min(a['per_match']):.3f}, {max(a['per_match']):.3f}] | "
                  f"{a['macro']:.4f} | {a['D_recall_seen']:.3f} / {a['D_recall_missing']:.3f} | {b['spell']:.4f} | {g or 'reference'} |")
    md += ["", f"Decoder (nested LOMO on {win['name']}): v10 costs {dec['ref_mean']:.4f}, nested {dec['nested_mean']:.4f} "
           f"({dec['nested_mean'] - dec['ref_mean']:+.4f}, {wins}/5) -> {'PASS' if dpass else 'FAIL'}; picks per fold {dec['picks']}.",
           f"Winner: {win['name']} = nets {win['recipe']}, dead {win['tsim']}/{win['variant']}, costs st/sd_in/sd_out/same {costs}."]
    (H.pl.HERE / "results" / "v21.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    H.prog("\n".join(md))


if __name__ == "__main__":
    main(tuple(int(s) for s in sys.argv[1].split(",")) if len(sys.argv) > 1 else (20, 21, 22))
