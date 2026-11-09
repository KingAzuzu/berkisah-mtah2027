"""v11 E2: distillation from a clean-feature teacher (cross-fitted, leak-free), IDSSE TUNE only, LOMO.
python -m possession.v11_distill teachers | student SEED. Design in results/v11.md.
Teacher(excl {m,k}) = v6 recipe on CLEAN feats of the other 3 TUNE matches, merged labels, seed 20; it gives soft log-probs
(mirror TTA) for k (used when m is held out) and for m (used when k is held out). Student for held-out m = v8 recipe on degraded+DR
feats of the 4 others; loss = CE(merged) + 0.5 * T^2 * KL(teacher_T || student_T), T = 2, over the 3 classes (+ unchanged boundary BCE)."""
import sys, time, os, itertools
import numpy as np, joblib, torch
import torch.nn.functional as F
from pathlib import Path

from . import m2_lib as L, tcn_lib as T, v4_lib as V
from .data import TUNE, idsse
from .v8_convention import merge

OUT = Path(__file__).parent / "logs" / "v11"
OUT.mkdir(exist_ok=True)
STEPS, SHORT_W, KD_W, TEMP = 1500, 3.0, 0.5, 2.0
data = L.data2()
labs = {m: [T.labels5(dict(h, truth=merge(h["truth"], 2))) for h in data[m]] for m in TUNE}


def tfile(a, b):
    a, b = sorted((a, b))
    return OUT / f"teach_{a}_{b}.npz"


def teachers():
    cf = joblib.load(idsse.DATA / "cache" / "tcn_feats_clean.pkl")
    for n, (a, b) in enumerate(itertools.combinations(TUNE, 2), 1):
        f = tfile(a, b)
        if f.exists():
            continue
        t0 = time.time()
        items = [{"src": "idsse", "feats": [cf[k][i]], "y": labs[k][i]} for k in TUNE if k not in (a, b) for i in range(2)]
        mod = V.train4(items, seed=20, steps=STEPS, short_w=SHORT_W)
        res = {}
        for k in (a, b):
            for i in range(2):
                res[f"{k}_{i}"] = V.predict4(mod, cf[k][i])[0].astype(np.float32)
        np.savez(f, **res)
        del mod, items
        torch.cuda.empty_cache()
        print("teacher", a, b, f"{time.time()-t0:.0f}s", flush=True)
        (OUT / "progress.txt").write_text(f"E2 teachers {n}/10\n")


def train_kd(items, seed, steps=STEPS, crop=300, bs=64, dev="cuda", short_w=SHORT_W):
    """V.train4 (cB: only 'idsse' items) plus the distillation term. Each item carries Q (T,3) teacher probs at temperature TEMP."""
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
        it["Q"] = torch.tensor(it["q"]).to(dev)
    size = np.array([len(it["y"]) for it in items], float)
    swap = torch.tensor(T.SWAP).to(dev)
    net = V.TCN4().to(dev)
    ema = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    sw = torch.tensor([1, 0, 2], device=dev)
    for s in range(steps):
        xb, yb, bb, wb, qb = [], [], [], [], []
        for _ in range(bs):
            it = items[rng.choice(len(items), p=size / size.sum())]
            v = rng.integers(len(it["Z"]))
            a = rng.integers(0, len(it["Y"]) - crop)
            Z = it["Zm"][v] if rng.random() < 0.5 else it["Z"][v]
            x, y, b, w, q = Z[a:a + crop], it["Y"][a:a + crop], it["B"][a:a + crop], it["W"][a:a + crop], it["Q"][a:a + crop]
            if rng.random() < 0.5:
                x, y = x[:, swap], torch.where((y == 0) | (y == 1), 1 - y, y)
                q = q[:, sw]
            xb.append(x.T), yb.append(y), bb.append(b), wb.append(w), qb.append(q)
        xb = torch.stack(xb)
        xb = xb + 0.05 * torch.randn_like(xb)
        yb, bb, wb, qb = torch.stack(yb), torch.stack(bb), torch.stack(wb), torch.stack(qb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(xb)
        out = out.float()
        lp3 = F.log_softmax(out[:, :3], 1)
        ce3 = -lp3.gather(1, yb.clamp(min=0)[:, None])[:, 0]
        valid = yb >= 0
        loss = (ce3 * valid * wb).sum() / valid.sum().clamp(min=1)
        ls = F.log_softmax(out[:, :3] / TEMP, 1)                       # (B,3,T)
        kl = (qb.permute(0, 2, 1) * (torch.log(qb.permute(0, 2, 1).clamp(min=1e-8)) - ls)).sum(1)
        loss = loss + KD_W * TEMP ** 2 * (kl * wb).mean()
        bce = F.binary_cross_entropy_with_logits(out[:, 3], bb, reduction="none")
        loss = loss + V.BD_LAMBDA * (bce * wb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.lerp_(p.detach(), 1 - V.EMA)
    with torch.no_grad():
        for e, p in zip(ema, net.parameters()):
            p.copy_(e)
    return {"net": net.cpu().eval(), "mu": mu, "sd": sd}


def tprobs(lp):
    """teacher log-probs (T,3) -> softmax at temperature TEMP"""
    z = lp / TEMP
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def student(seed):
    canon = joblib.load(idsse.DATA / "cache" / "tcn_feats_matched2.pkl")
    dr = joblib.load(idsse.DATA / "cache" / "tcn_feats_dr.pkl")
    for m in TUNE:
        f = OUT / f"D_s{seed}_{m}.npz"
        if f.exists():
            continue
        t0 = time.time()
        items = []
        for k in TUNE:
            if k == m:
                continue
            tz = np.load(tfile(m, k))
            for i in range(2):
                q = tprobs(tz[f"{k}_{i}"])
                assert len(q) == len(labs[k][i]), (len(q), len(labs[k][i]))
                items.append({"src": "idsse", "feats": [canon[k][i]] + [dr[k][v][i] for v in range(len(dr[k]))],
                              "y": labs[k][i], "q": q})
        mod = train_kd(items, seed)
        res = {}
        for i in range(2):
            res[f"lp{i}"], res[f"b{i}"] = V.predict4(mod, canon[m][i])
        np.savez(f, **res)
        del mod, items
        torch.cuda.empty_cache()
        print("D", seed, m, f"{time.time()-t0:.0f}s", flush=True)
        n = len(list(OUT.glob("D_s*.npz")))
        (OUT / "progress.txt").write_text(f"E2 student seed {seed} ({n}/25)\n")


if __name__ == "__main__":
    teachers() if sys.argv[1] == "teachers" else student(int(sys.argv[2]))
