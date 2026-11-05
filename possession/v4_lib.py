"""v4 candidate: TCN + boundary head + EMA + multi-source data (IDSSE with domain-randomised degrade, Metrica, SkillCorner),
mirror augmentation and TTA. Training sees only the listed sources; D is masked for SkillCorner (team loss only)."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tcn_lib as T

MIRROR_COLS = [0, 1, 10, 11, 20, 21]
BD_SIGMA = 2.0        # steps at 5 Hz = 0.4 s
BD_LAMBDA = 0.5
EMA = 0.995
W_SRC = {"idsse": 0.60, "met": 0.15, "sc": 0.25}   # fixed a priori sampling mass per source
HOOK = 0.6            # Viterbi switch cost scaled by (1 - HOOK * boundary prob)


class TCN4(T.TCN):
    """TCN with a 4th output channel: boundary logit (change of H/A/D label)."""
    def __init__(self, nin=26, c=64, p=0.2):
        super().__init__(nin, c, p)
        self.ob = nn.Conv1d(c, 1, 1)

    def forward(self, x):
        h = self.i(x)
        for b in self.b:
            h = b(h)
        h = F.gelu(h)
        return torch.cat([self.o(h), self.ob(h)], 1)


def mirror(X):
    X = X.copy()
    X[:, MIRROR_COLS] *= -1
    return X


def boundary_target(y):
    """y (T,) int labels (-1 unknown) at 5 Hz -> Gaussian-smoothed change indicator in [0,1]."""
    ch = np.zeros(len(y), np.float32)
    ch[1:] = ((y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)).astype(np.float32)
    k = np.exp(-0.5 * (np.arange(-6, 7) / BD_SIGMA) ** 2).astype(np.float32)
    return np.minimum(np.convolve(ch, k, "same"), 1.0)


def short_weights(y, factor, near=5, short=15):
    """(T,) loss weights: factor within +-near steps (1 s at 5 Hz) of a true H<->A change where either adjacent spell
    (constant-label run) is shorter than `short` steps (3 s); 1 elsewhere. factor 1 -> all ones."""
    w = np.ones(len(y), np.float32)
    if factor == 1.0:
        return w
    cuts = np.flatnonzero(np.diff(y) != 0) + 1                  # run starts (index of first frame of new run)
    starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(y)]])
    for k in range(1, len(starts)):
        a, b = y[starts[k - 1]], y[starts[k]]
        if a in (0, 1) and b in (0, 1) and a != b and min(ends[k - 1] - starts[k - 1], ends[k] - starts[k]) < short:
            w[max(0, starts[k] - near):starts[k] + near] = factor
    return w


def train4(items, seed=0, steps=1500, crop=300, bs=64, bd_lambda=BD_LAMBDA, dev="cuda", short_w=1.0):
    """items: list of dict(src, feats=[variant feature arrays (T,26)], y=(T,) int64 (-1 unknown))."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    allX = np.concatenate([it["feats"][0] for it in items])
    mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    for it in items:
        it["Z"] = [torch.tensor(T.prep(f, mu, sd)).to(dev) for f in it["feats"]]
        it["Zm"] = [torch.tensor(T.prep(mirror(f), mu, sd)).to(dev) for f in it["feats"]]
        it["Y"] = torch.tensor(it["y"]).to(dev)
        it["B"] = torch.tensor(boundary_target(it["y"])).to(dev)
        it["W"] = torch.tensor(short_weights(it["y"], short_w)).to(dev)
    srcs = sorted({it["src"] for it in items})
    wsrc = np.array([W_SRC[s] for s in srcs]); wsrc = wsrc / wsrc.sum()
    by = {s: [it for it in items if it["src"] == s] for s in srcs}
    size = {s: np.array([len(it["y"]) for it in by[s]], float) for s in srcs}
    swap = torch.tensor(T.SWAP).to(dev)
    net = TCN4().to(dev)
    ema = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for s in range(steps):
        xb, yb, bb, sc, wb = [], [], [], [], []
        for _ in range(bs):
            src = srcs[rng.choice(len(srcs), p=wsrc)]
            it = by[src][rng.choice(len(by[src]), p=size[src] / size[src].sum())]
            v = rng.integers(len(it["Z"]))
            a = rng.integers(0, len(it["Y"]) - crop)
            Z = it["Zm"][v] if rng.random() < 0.5 else it["Z"][v]
            x, y, b, w = Z[a:a + crop], it["Y"][a:a + crop], it["B"][a:a + crop], it["W"][a:a + crop]
            if rng.random() < 0.5:
                x, y = x[:, swap], torch.where((y == 0) | (y == 1), 1 - y, y)
            xb.append(x.T), yb.append(y), bb.append(b), wb.append(w), sc.append(src == "sc")
        xb = torch.stack(xb)
        xb = xb + 0.05 * torch.randn_like(xb)
        yb, bb, wb = torch.stack(yb), torch.stack(bb), torch.stack(wb)
        scm = torch.tensor(sc, device=dev)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(xb)
        out = out.float()
        lp3 = F.log_softmax(out[:, :3], 1)
        lp2 = F.log_softmax(out[:, :2], 1)
        yc = yb.clamp(min=0)
        ce3 = -lp3.gather(1, yc[:, None])[:, 0]
        ce2 = -lp2.gather(1, yc.clamp(max=1)[:, None])[:, 0]
        known = yb >= 0
        ce = torch.where(scm, ce2, ce3)
        valid = known & (~scm | (yb < 2))
        loss = (ce * valid * wb).sum() / valid.sum().clamp(min=1)
        if bd_lambda > 0:
            bce = F.binary_cross_entropy_with_logits(out[:, 3], bb, reduction="none")
            loss = loss + bd_lambda * (bce * wb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.lerp_(p.detach(), 1 - EMA)
    with torch.no_grad():
        for e, p in zip(ema, net.parameters()):
            p.copy_(e)
    return {"net": net.cpu().eval(), "mu": mu, "sd": sd}


@torch.no_grad()
def predict4(model, feats, tta=True):
    """-> (lp (T5,3) log-probs, b (T5,) boundary prob) at 5 Hz, CPU float32; mirror TTA averages log-probs and probs."""
    torch.set_num_threads(1)
    outs = []
    for X in ([feats, mirror(feats)] if tta else [feats]):
        z = torch.tensor(T.prep(X, model["mu"], model["sd"])).T[None]
        o = model["net"](z)[0].T
        outs.append((F.log_softmax(o[:, :3], 1).numpy().astype(np.float64), torch.sigmoid(o[:, 3]).numpy().astype(np.float64)))
    return np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0)


def upsample(a, n):
    up = np.repeat(a, T.SUB, axis=0)
    return np.concatenate([up, np.repeat(up[-1:], n - len(up), axis=0)]) if len(up) < n else up[:n]


def viterbi_scaled(e, st, sd, scale):
    """safety.viterbi with the switch cost at step t multiplied by scale[t] (scale = 1 reproduces it)."""
    T_ = len(e)
    back = np.empty((T_, 3), np.int8)
    score = list(e[0])
    rows, sc = e.tolist(), scale.tolist()
    for t in range(1, T_):
        k = sc[t]
        c = ((0.0, st * k, sd * k), (st * k, 0.0, sd * k), (sd * k, sd * k, 0.0))
        new, r = [0.0, 0.0, 0.0], rows[t]
        for j in range(3):
            best, arg = score[0] - c[0][j], 0
            for i in (1, 2):
                v = score[i] - c[i][j]
                if v > best:
                    best, arg = v, i
            new[j] = best + r[j]
            back[t, j] = arg
        score = new
    path = np.empty(T_, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T_ - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
