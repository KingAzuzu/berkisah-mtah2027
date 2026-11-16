"""v17 step 1 (CPU): upside of filling missing ball positions from a video ball detector, simulated on IDSSE.

Frozen v10 (final nets, trained on TUNE) on the matched2 seed-0 TUNE halves, so both arms are IN-SAMPLE: read the delta only.
Locked matches are not touched. Simulated detector: on frames where the matched2 ball is missing and the clean DFL ball exists
(alive AND dead frames, as a camera sees the ball in both), 1 s blocks are detected with probability p; detected positions are
clean + smooth error (1 Hz Gaussian offsets of std sigma per axis, linearly interpolated, like a Kalman-smoothed track).
Configs fixed beforehand: (p, sigma) in (0.3,1) (0.5,0.5) (0.5,1) (0.5,2) (0.7,1) (1,0.5) (1,1) (1,2) and oracle (1,0).
Gate fixed beforehand on the realistic cell (0.5, 1): mean delta vs v10 >= +0.01 on BOTH truths and >= 4/5 matches better on avg.
python -m possession.v17_fill_sim -> logs/v17/fill_sim.json"""
import json
import sys
import numpy as np
from . import m2_lib as L
from .data import TUNE, load_idsse
from .pipeline import v10_predictor
from .score import score
from .v8_convention import merge
from .v7_decoder import LOG

CFGS = [(0.3, 1.0), (0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (0.7, 1.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (1.0, 0.0)]
GATE = (0.5, 1.0)


def fill(deg, clean, p, sigma, seed):
    rng = np.random.default_rng(seed)
    ball = deg["ball"].copy()
    T = len(ball)
    cand = np.isnan(ball[:, 0]) & ~np.isnan(clean["ball"][:, 0])
    det = np.repeat(rng.random(T // 25 + 1) < p, 25)[:T]
    knots = np.arange(0, T + 25, 25)
    off = np.stack([np.interp(np.arange(T), knots, rng.normal(0, sigma, len(knots))) for _ in range(2)], 1)
    m = cand & det
    ball[m] = clean["ball"][m, :2] + off[m]
    return dict(deg, ball=ball), float(m.sum() / max(cand.sum(), 1))


def run(predict, data, clean, cfg):
    preds, frac = {}, []
    for k, m in enumerate(TUNE):
        hs = []
        for i, (d, c) in enumerate(zip(data[m], clean[m])):
            if cfg is not None:
                d, f = fill(d, c, *cfg, seed=1000 * k + i)
                frac.append(f)
            hs.append(predict(d))
        preds[m] = hs
    out = {}
    for tn in ("dfl", "merged"):
        s = []
        for m in TUNE:
            halves = [dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in data[m]]
            s.append(float(score(halves, preds[m])["spell_f1_mean"]))
        out[tn] = s
    out["filled_frac"] = float(np.mean(frac)) if frac else 0.0
    return out


def task(cfg):
    _, predict = v10_predictor()
    data = L.data2()
    clean = {m: load_idsse(m) for m in TUNE}
    for m in TUNE:
        assert all(len(a["ball"]) == len(b["ball"]) for a, b in zip(data[m], clean[m]))
    r = run(predict, data, clean, cfg)
    print("done", cfg, flush=True)
    return r


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        outs = pool.map(task, [None] + CFGS, chunksize=1)
    res = {"base": outs[0]}
    b = res["base"]
    print(f"base dfl {np.mean(b['dfl']):.4f} merged {np.mean(b['merged']):.4f}", flush=True)
    for cfg, r in zip(CFGS, outs[1:]):
        d = {tn: [x - y for x, y in zip(r[tn], b[tn])] for tn in ("dfl", "merged")}
        wins = sum((a + c) > 0 for a, c in zip(d["dfl"], d["merged"]))
        r.update(delta_dfl=float(np.mean(d["dfl"])), delta_merged=float(np.mean(d["merged"])), wins=wins)
        res[f"{cfg[0]}|{cfg[1]}"] = r
        print(f"p={cfg[0]} sigma={cfg[1]} filled {r['filled_frac']:.2f} d_dfl {r['delta_dfl']:+.4f} d_merged {r['delta_merged']:+.4f} "
              f"wins {wins}/5 per-match d_dfl " + " ".join(f"{x:+.3f}" for x in d["dfl"]), flush=True)
    g = res[f"{GATE[0]}|{GATE[1]}"]
    res["gate"] = bool(g["delta_dfl"] >= 0.01 and g["delta_merged"] >= 0.01 and g["wins"] >= 4)
    print("gate (0.5, 1):", "PASS" if res["gate"] else "FAIL")
    (LOG / "v17").mkdir(exist_ok=True)
    json.dump(res, open(LOG / "v17" / "fill_sim.json", "w"), indent=1)
