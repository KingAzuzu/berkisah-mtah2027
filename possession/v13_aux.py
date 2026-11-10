"""v13 C2: v8 TCN4 + auxiliary head (true ball xy Huber, nearest-team CE) trained from clean IDSSE ball/players; inference = v8.
python -m possession.v13_aux smoke | train SEED | score -> logs/v13/aux_s{seed}_{match}.npz, score_C2.json"""
import sys, os, time, json
import numpy as np, joblib, torch, torch.nn as nn, torch.nn.functional as F
from . import m2_lib as L, tcn_lib as T, v4_lib as V, v7_decoder as E
from .data import TUNE, idsse, load_idsse
from .v8_convention import merge
from .score import score

OUT = os.path.join(os.path.dirname(__file__), "logs", "v13")
W_XY, W_NT = 0.3, 0.3
NORM = np.array([52.5, 34.0], np.float32)


class TCN4Aux(V.TCN4):
    """out channels: 0-2 class, 3 boundary, 4-5 ball xy, 6-7 nearest-team logits."""
    def __init__(self, nin=26, c=64, p=0.2):
        super().__init__(nin, c, p)
        self.oa = nn.Conv1d(c, 4, 1)

    def forward(self, x):
        h = self.i(x)
        for b in self.b:
            h = b(h)
        h = F.gelu(h)
        return torch.cat([self.o(h), self.ob(h), self.oa(h)], 1)


def aux_targets(clean_half, merged_y5):
    """-> xy (T5,2) normalised (NaN unknown), nt (T5,) int64 (-1 dead/unknown), sampled at the label frame."""
    n = len(merged_y5) * T.SUB
    ball = clean_half["ball"][:n].reshape(-1, T.SUB, 2)[:, T.SUB // 2]
    P = clean_half["xy"]
    xy = P[:n].reshape(-1, T.SUB, P.shape[1], 2)[:, T.SUB // 2]
    team = clean_half["team"]
    with np.errstate(all="ignore"):
        d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dm = [np.where(team == t, d, np.inf).min(1) for t in (0, 1)]
    nt = (dm[1] < dm[0]).astype(np.int64)
    ok = np.isfinite(np.minimum(dm[0], dm[1])) & (merged_y5 != 2) & (merged_y5 >= 0)
    nt = np.where(ok, nt, -1)
    return (ball / NORM).astype(np.float32), nt


def train_aux(items, seed=0, steps=1500, crop=300, bs=64, dev="cuda", short_w=3.0, log=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    allX = np.concatenate([it["feats"][0] for it in items])
    mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    for it in items:
        it["Z"] = [torch.tensor(T.prep(f, mu, sd)).to(dev) for f in it["feats"]]
        it["Zm"] = [torch.tensor(T.prep(V.mirror(f), mu, sd)).to(dev) for f in it["feats"]]
        it["Y"] = torch.tensor(it["y"]).to(dev)
        it["B"] = torch.tensor(V.boundary_target(it["y"])).to(dev)
        it["W"] = torch.tensor(V.short_weights(it["y"], short_w)).to(dev)
        it["XY"] = torch.tensor(it["xy"]).to(dev)
        it["NT"] = torch.tensor(it["nt"]).to(dev)
    size = np.array([len(it["y"]) for it in items], float)
    size /= size.sum()
    swap = torch.tensor(T.SWAP).to(dev)
    net = TCN4Aux().to(dev)
    ema = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for s in range(steps):
        xb, yb, bb, wb, xyb, ntb = [], [], [], [], [], []
        for _ in range(bs):
            it = items[rng.choice(len(items), p=size)]
            v = rng.integers(len(it["Z"]))
            a = rng.integers(0, len(it["Y"]) - crop)
            mir = rng.random() < 0.5
            Z = it["Zm"][v] if mir else it["Z"][v]
            x, y, b, w = Z[a:a + crop], it["Y"][a:a + crop], it["B"][a:a + crop], it["W"][a:a + crop]
            xy, nt = it["XY"][a:a + crop], it["NT"][a:a + crop]
            if mir:
                xy = -xy
            if rng.random() < 0.5:
                x, y = x[:, swap], torch.where((y == 0) | (y == 1), 1 - y, y)
                nt = torch.where(nt >= 0, 1 - nt, nt)
            xb.append(x.T), yb.append(y), bb.append(b), wb.append(w), xyb.append(xy), ntb.append(nt)
        xb = torch.stack(xb)
        xb = xb + 0.05 * torch.randn_like(xb)
        yb, bb, wb, xyb, ntb = [torch.stack(a) for a in (yb, bb, wb, xyb, ntb)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(xb)
        out = out.float()
        lp3 = F.log_softmax(out[:, :3], 1)
        yc = yb.clamp(min=0)
        ce3 = -lp3.gather(1, yc[:, None])[:, 0]
        known = yb >= 0
        loss = (ce3 * known * wb).sum() / known.sum().clamp(min=1)
        loss = loss + V.BD_LAMBDA * (F.binary_cross_entropy_with_logits(out[:, 3], bb, reduction="none") * wb).mean()
        pxy = out[:, 4:6].transpose(1, 2)
        kx = ~torch.isnan(xyb).any(-1)
        hub = F.huber_loss(pxy, torch.nan_to_num(xyb), reduction="none", delta=0.1).sum(-1)
        l_xy = (hub * kx).sum() / kx.sum().clamp(min=1)
        kn = ntb >= 0
        l_nt = (F.cross_entropy(out[:, 6:8], ntb.clamp(min=0), reduction="none") * kn).sum() / kn.sum().clamp(min=1)
        loss = loss + W_XY * l_xy + W_NT * l_nt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log and s % 50 == 0:
            print(s, round(float(loss), 4), "xy", round(float(l_xy), 4), "nt", round(float(l_nt), 4), flush=True)
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.lerp_(p.detach(), 1 - V.EMA)
    with torch.no_grad():
        for e, p in zip(ema, net.parameters()):
            p.copy_(e)
    return {"net": net.cpu().eval(), "mu": mu, "sd": sd}


def build():
    data = L.data2()
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    base = {}
    for m in TUNE:
        clean = load_idsse(m)
        base[m] = []
        for i in range(2):
            y = T.labels5(dict(data[m][i], truth=merge(data[m][i]["truth"], 2)))
            xy, nt = aux_targets(clean[i], y)
            assert len(xy) == len(y) == len(canon[m][i]), (len(xy), len(y), len(canon[m][i]))
            base[m].append({"src": "idsse", "feats": [canon[m][i]] + [dr[m][v][i] for v in range(len(dr[m]))], "y": y, "xy": xy, "nt": nt})
        del clean
    return base, canon


def run(seeds, smoke=False):
    base, canon = build()
    folds = TUNE[:1] if smoke else TUNE
    for seed in seeds:
        for m in folds:
            f = os.path.join(OUT, f"aux_s{seed}_{m}.npz")
            if os.path.exists(f) and not smoke:
                continue
            t0 = time.time()
            items = [dict(it) for k in TUNE if k != m for it in base[k]]
            mod = train_aux(items, seed=seed, steps=300 if smoke else 1500, log=smoke)
            res = {}
            for i in range(2):
                lp, b = V.predict4(mod, canon[m][i])
                res[f"lp{i}"], res[f"b{i}"] = lp, b
            if smoke:
                print("shapes", lp.shape, b.shape, canon[m][1].shape)
                return
            np.savez(f, **res)
            del items, mod
            torch.cuda.empty_cache()
            print(seed, m, f"{time.time()-t0:.0f}s", flush=True)
        open(os.path.join(OUT, "progress.txt"), "w").write(f"C2 seed {seed} ({(seed-19)*5}/25)\n")


def dec(pref, D, c):
    out = {}
    for m in TUNE:
        fs = [np.load(D / f"{pref}_s{s}_{m}.npz") for s in range(20, 25)]
        p = []
        for i in range(2):
            n = len(c[m][i]["truth"])
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
            p.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
        out[m] = p
    return out


def main_score():
    from pathlib import Path
    c = joblib.load(E.CACHE)
    P = {"v8": dec("m2", E.LOG / "v8", c), "aux": dec("aux", Path(OUT), c)}
    res = {}
    for tn in ("merged", "dfl"):
        for k in P:
            s = [float(score([dict(h, truth=merge(h["truth"], 2) if tn == "merged" else h["truth"]) for h in c[m]], P[k][m])["spell_f1_mean"]) for m in TUNE]
            res[f"{k} vs {tn}"] = {"mean": float(np.mean(s)), "per_match": s}
    a, r = np.array(res["aux vs merged"]["per_match"]), np.array(res["v8 vs merged"]["per_match"])
    res["gate"] = {"wins": int((a > r).sum()), "pass": bool(a.mean() >= 0.7663 and (a > r).sum() >= 4)}
    json.dump(res, open(os.path.join(OUT, "score_C2.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    open(os.path.join(OUT, "progress.txt"), "w").write(f"C2 scored {a.mean():.4f} {'PASS' if res['gate']['pass'] else 'FAIL'}\n")


if __name__ == "__main__":
    a = sys.argv[1]
    run([20], True) if a == "smoke" else run([int(x) for x in sys.argv[2:]]) if a == "train" else main_score()
