"""Error analysis of reference R (v2.1 under matched2, LOMO). Writes possession/results/error_analysis.md.
IDSSE TUNE matches only."""
import joblib
import numpy as np
from . import m2_lib as L
from .data import FPS, TUNE, pm
from .score import score

LEN_BINS = [(0, 2, "<2s"), (2, 5, "2-5s"), (5, 15, "5-15s"), (15, 1e9, ">15s")]
LABELS = [l for *_, l in LEN_BINS]


def lb(n):
    s = n / FPS
    return next(l for a, b, l in LEN_BINS if a <= s < b)


def ffill_team(codes):
    """Replace D by nearest previous non-D (back-fill at the start)."""
    out = codes.copy()
    idx = np.where(out != 2, np.arange(len(out)), -1)
    np.maximum.accumulate(idx, out=idx)
    ok = idx >= 0
    out[ok] = codes[idx[ok]]
    if (~ok).any() and ok.any():
        out[~ok] = codes[np.flatnonzero(ok)[0]]
    return out


def oracle_D(truth, pred):
    """truth D -> D; else our team (our D filled with our previous team)."""
    o = ffill_team(pred)
    o[truth == 2] = 2
    return o


def oracle_team(truth, pred):
    """our D -> D; else the true team (truth D filled with last true team)."""
    o = ffill_team(truth)
    o[pred == 2] = 2
    return o


def trans(c, kind):
    a, b = c[:-1], c[1:]
    if kind == "dstart":
        return np.flatnonzero((a != 2) & (b == 2)) + 1
    if kind == "dend":
        return np.flatnonzero((a == 2) & (b != 2)) + 1
    return np.flatnonzero((a != 2) & (b != 2) & (a != b)) + 1


def offsets(g, p):
    if not len(g) or not len(p):
        return np.array([])
    j = np.searchsorted(p, g)
    c = np.stack([p[np.clip(j - 1, 0, len(p) - 1)], p[np.clip(j, 0, len(p) - 1)]])
    k = np.argmin(np.abs(c - g), axis=0)
    return (c[k, np.arange(len(g))] - g) / FPS


def q(x, qs=(10, 50, 90)):
    return "n/a" if not len(x) else "/".join(f"{np.percentile(x, v):.2f}" for v in qs)


def new_acc():
    z = lambda: dict.fromkeys(LABELS, 0)
    return {"gt": z(), "m03": z(), "miss": z(), "loose": 0, "cause": {}, "cause_len": {},
            "pred": 0, "fp": 0, "fp_short": 0, "fpc": {}, "off": {"dstart": [], "dend": [], "ha": []}}


def analyse_half(truth, pred, acc):
    runs = [(0, len(truth))]
    g = pm.possession_spells(truth, runs)
    p = pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(g, p)
    f03 = pm.matched_gt_flags(ov, len(g), 0.3)
    f07 = pm.matched_gt_flags(ov, len(g), 0.7)
    mp03, used_g = set(), set()
    for iou, i, j in ov:
        if iou < 0.3:
            break
        if i in used_g or j in mp03:
            continue
        used_g.add(i)
        mp03.add(j)
    for i, (a, b, k) in enumerate(g):
        L_ = lb(b - a)
        acc["gt"][L_] += 1
        if f03[i]:
            acc["m03"][L_] += 1
            acc["loose"] += int(not f07[i])
            continue
        acc["miss"][L_] += 1
        seg = pred[a:b]
        same, other, dd = np.mean(seg == k), np.mean((seg != k) & (seg != 2)), np.mean(seg == 2)
        n_ps = sum(1 for (c, d, kk) in p if kk == k and c < b and d > a)
        if n_ps >= 2:
            cause = "fragmented (>=2 same-team pred spells overlap)"
        elif other > 0.5:
            cause = "wrong team"
        elif dd > 0.5:
            cause = "predicted D"
        elif same > 0.5:
            cause = "one same-team spell, IoU<0.3 (boundary/merge)"
        else:
            cause = "mixed"
        acc["cause"][cause] = acc["cause"].get(cause, 0) + 1
        key = cause + "|" + L_
        acc["cause_len"][key] = acc["cause_len"].get(key, 0) + 1
    for j, (a, b, k) in enumerate(p):
        acc["pred"] += 1
        if j in mp03:
            continue
        acc["fp"] += 1
        acc["fp_short"] += int((b - a) / FPS < 2)
        seg = truth[a:b]
        tD, ts, to = np.mean(seg == 2), np.mean(seg == k), np.mean((seg != k) & (seg != 2))
        c = "mostly truth D" if tD > 0.5 else ("wrong team" if to > 0.5 else ("right team, bad IoU" if ts > 0.5 else "mixed"))
        acc["fpc"][c] = acc["fpc"].get(c, 0) + 1
    for kind in acc["off"]:
        acc["off"][kind].append(offsets(trans(truth, kind), trans(pred, kind)))


def add(tot, acc):
    for k in ("gt", "m03", "miss"):
        for l in LABELS:
            tot[k][l] += acc[k][l]
    for k in ("loose", "pred", "fp", "fp_short"):
        tot[k] += acc[k]
    for k in ("cause", "cause_len", "fpc"):
        for a, v in acc[k].items():
            tot[k][a] = tot[k].get(a, 0) + v
    for k in tot["off"]:
        tot["off"][k] += acc["off"][k]


def main():
    data, R = L.data2(), joblib.load(L.CACHE / "matched2_R.pkl")
    tot = new_acc()
    rows, orows = [], []
    for m in TUNE:
        acc = new_acc()
        preds = R[m][1]
        for h, p in zip(data[m], preds):
            analyse_half(h["truth"], p, acc)
        add(tot, acc)
        base = score(data[m], preds)["spell_f1_mean"]
        so = score(data[m], [oracle_D(h["truth"], p) for h, p in zip(data[m], preds)])["spell_f1_mean"]
        st = score(data[m], [oracle_team(h["truth"], p) for h, p in zip(data[m], preds)])["spell_f1_mean"]
        orows.append((m, base, so, st))
        rows.append((m, acc["pred"], sum(acc["gt"].values()), sum(acc["miss"].values()), acc["fp"]))
    out = ["# Error analysis of R (v2.1 under matched2, LOMO, TUNE, seed 0)", "",
           "Script `possession/error_analysis.py`, reference preds from `possession/m2_lib.py`. Spell matching at IoU 0.3 "
           "(loosest threshold) unless stated; pooled over 5 matches.", "",
           "## Spell counts per match", "| match | pred spells | true spells | true missed @0.3 | pred unmatched @0.3 |",
           "|---|---|---|---|---|"]
    out += [f"| {m} | {a} | {b} | {c} | {d} |" for m, a, b, c, d in rows]
    out += ["", "## True spells by length (pooled)", "| length | n true | matched @0.3 | missed | miss rate |", "|---|---|---|---|---|"]
    for l in LABELS:
        n = tot["gt"][l]
        out.append(f"| {l} | {n} | {tot['m03'][l]} | {tot['miss'][l]} | {tot['miss'][l] / max(n, 1):.2f} |")
    n_m = sum(tot["m03"].values())
    out.append(f"\nMatched at 0.3 but not at 0.7 (boundary loss): {tot['loose']} of {n_m} ({tot['loose'] / n_m:.2f}).")
    nm = sum(tot["miss"].values())
    out += ["", "## Why true spells are missed (pooled)", "| cause | n | share of missed |", "|---|---|---|"]
    for c, v in sorted(tot["cause"].items(), key=lambda x: -x[1]):
        out.append(f"| {c} | {v} | {v / nm:.2f} |")
    out += ["", "## Predicted spells unmatched @0.3 (pooled)",
            f"{tot['fp']} of {tot['pred']} predicted spells unmatched; {tot['fp_short']} of those are <2 s.", "",
            "| truth under the spell | n | share |", "|---|---|---|"]
    for c, v in sorted(tot["fpc"].items(), key=lambda x: -x[1]):
        out.append(f"| {c} | {v} | {v / tot['fp']:.2f} |")
    out += ["", "## Boundary offsets, predicted minus truth (nearest predicted transition of the same type), seconds", "",
            "| boundary | n true | signed p10/p50/p90 | abs<=1 s | abs<=3 s | median abs |", "|---|---|---|---|---|---|"]
    names = {"dstart": "alive -> D (D start)", "dend": "D -> alive (D end)", "ha": "direct H<->A change"}
    for k, nme in names.items():
        o = np.concatenate(tot["off"][k])
        a = np.abs(o)
        out.append(f"| {nme} | {len(o)} | {q(o)} | {np.mean(a <= 1):.2f} | {np.mean(a <= 3):.2f} | {np.median(a):.2f} |")
    out += ["", "## Oracle tests (spellF1 of the same decoded preds)", "| match | R | truth D, our team | truth team, our D |",
            "|---|---|---|---|"]
    out += [f"| {m} | {a:.3f} | {b:.3f} | {c:.3f} |" for m, a, b, c in orows]
    ar = np.array([r[1:] for r in orows])
    out.append(f"| mean | {ar[:, 0].mean():.3f} | {ar[:, 1].mean():.3f} | {ar[:, 2].mean():.3f} |")
    out += ["", "Truth D, our team: frames where truth is D become D, others take our team (our D filled with our previous team). "
            "Truth team, our D: frames where we say D stay D, others take the true team (truth D filled with last true team)."]
    (L.pl.HERE / "results" / "error_analysis.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
