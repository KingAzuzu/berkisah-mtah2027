"""Upper bound for ball filling. python -m possession.ball_fill_bound  (IDSSE TUNE, frozen v4 = in-sample)."""
import copy, time, json
import numpy as np
from . import m2_lib as L, pipeline as pl, degrade_matched as M, degrade_matched2 as M2
from .data import TUNE
from .score import score

LOG = pl.HERE / "logs" / "v5b_progress.txt"
t0 = time.time()
def prog(pct, step):
    eta = (time.time() - t0) / max(pct, 1) * (100 - pct)
    LOG.write_text(f"{pct}% | {step} | ETA {eta/60:.0f} min\n")

def noise_and_mask(mid, hi, half, seed=0):
    rec = {}
    old = M._pick_mask
    def pm(h, crowd, rng):
        m = M2._pick_mask(h, crowd, rng)
        rec["mask"], rec["noise"] = m, copy.deepcopy(rng).standard_normal((len(h["ball"]), 2))
        return m
    M._pick_mask = pm
    try:
        d = M.degrade(half, mid, hi, seed)
    finally:
        M._pick_mask = old
    return d, rec["mask"], (rec["noise"] * M.BALL_JITTER_M).astype(np.float32)

prog(1, "load")
data = L.data2()
from .data import load_idsse
res = {"a": {}, "b": {}, "c": {}}; frac = {}
p, predict = pl.v4_predictor()
for k, mid in enumerate(TUNE):
    raw = load_idsse(mid)
    cond = {"a": [], "b": [], "c": []}
    for hi in range(2):
        d, mask, nz = noise_and_mask(mid, hi, raw[hi])
        base = data[mid][hi]
        assert np.allclose(base["ball"][mask], d["ball"][mask], equal_nan=True)
        restore = ~mask & (raw[hi]["truth"] != 2)
        full = d["ball"].copy(); full[restore] = raw[hi]["ball"][restore] + nz[restore]
        # 50%: random runs, seed 0
        e = np.diff(np.r_[False, restore, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))
        rng = np.random.default_rng(0); rng.shuffle(runs)
        half_r = np.zeros(len(mask), bool); tot = restore.sum(); got = 0
        for a, b in runs:
            if got >= 0.5 * tot: break
            half_r[a:b] = True; got += b - a
        part = d["ball"].copy(); part[half_r] = raw[hi]["ball"][half_r] + nz[half_r]
        frac[(mid, hi)] = (len(mask), int((~np.isnan(d["ball"][:, 0])).sum()), int(restore.sum()), int(got))
        for c, bl in (("a", d["ball"]), ("b", full), ("c", part)):
            h = dict(d); h["ball"] = bl
            cond[c].append(predict(h))
    for c in cond:
        r = score([{**data[mid][i]} for i in range(2)], cond[c]); res[c][mid] = r["spell_f1_mean"]
    print(mid, {c: round(res[c][mid], 4) for c in res}, flush=True)
    prog(int(100 * (k + 1) / len(TUNE)), f"fold {mid}")
mean = {c: float(np.mean(list(res[c].values()))) for c in res}
tt = sum(v[0] for v in frac.values()); seen = sum(v[1] for v in frac.values())
rs = sum(v[2] for v in frac.values()); gt = sum(v[3] for v in frac.values())
names = {"a": "(a) baseline matched2", "b": "(b) restore all hidden-alive", "c": "(c) restore ~50% (random runs, seed 0)"}
md = ["# Ball-fill upper bound (IDSSE TUNE, matched2, frozen v4)", "",
      "Caveat: per-fold v4 models were not saved (only log-prob npz); frozen models trained on all 5 TUNE matches are used, so the result is IN-SAMPLE (absolute levels inflated; compare conditions only).",
      f"Ball seen: baseline {seen/tt:.3f}; restored in (b) {rs/tt:.3f} of frames (alive and hidden); (c) {gt/tt:.3f}. Restored = truth + same jitter; dead-ball hiding kept. Code: possession/ball_fill_bound.py.", "",
      "| condition | " + " | ".join(TUNE) + " | mean |", "|---|" + "---|" * (len(TUNE) + 1)]
for c in res:
    md.append(f"| {names[c]} | " + " | ".join(f"{res[c][m]:.3f}" for m in TUNE) + f" | {mean[c]:.4f} |")
md += ["", f"(b) - (a) = {mean['b']-mean['a']:+.4f}; (c) - (a) = {mean['c']-mean['a']:+.4f}; threshold 0.02: {'yes' if mean['b']-mean['a']>=0.02 else 'no'}.",
       "Single-match differences below ~0.04 are noise; one degrade seed (0)."]
(pl.HERE / "results" / "ball_fill_bound.md").write_text("\n".join(md) + "\n")
LOG.write_text("100% | done | ETA 0\nDONE\n")
print("\n".join(md))
