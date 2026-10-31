"""Candidate B: small dilated residual TCN on per-frame permutation-invariant features (5 Hz), IDSSE LOMO."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dead
from .safety import MAX_SPEED

FPS, SUB = 25, 5
DIL = (1, 2, 4, 8, 16, 4, 2)   # receptive field +-37 steps = +-7.4 s at 5 Hz


def _fill(a):
    """Linear interpolation of NaNs along axis 0 (edges held); all-NaN -> 0."""
    out = a.copy()
    idx = np.arange(len(a))
    for j in range(a.shape[1]):
        ok = ~np.isnan(a[:, j])
        out[:, j] = np.interp(idx, idx[ok], a[ok, j]) if ok.any() else 0.0
    return out


def frame_features(half):
    xy, ball, team = half["xy"], half["ball"], half["team"]
    seen = ~np.isnan(ball[:, 0])
    bf = _fill(ball)
    since, until = dead._since_seen(seen)
    d = np.linalg.norm(xy - bf[:, None, :], axis=2)
    with np.errstate(all="ignore"):
        v = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2) * FPS
        v[v > MAX_SPEED] = np.nan
        cols = []
        for t in (0, 1):
            m = team == t
            p, vt, dt = xy[:, m], v[:, m], np.sort(np.where(np.isnan(d[:, m]), np.inf, d[:, m]), axis=1)
            cx, cy = np.nanmean(p[..., 0], 1), np.nanmean(p[..., 1], 1)
            spread = np.sqrt(np.nanvar(p[..., 0], 1) + np.nanvar(p[..., 1], 1))
            cols += [cx, cy, spread, np.nanmean(vt, 1), np.nanmax(np.where(np.isnan(vt), -1, vt), 1),
                     np.minimum(dt[:, 0], 60), np.minimum(dt[:, 1], 60)]
            cols += [(dt < r).sum(1).astype(float) for r in (1, 3, 10)]
        bv = np.linalg.norm(np.diff(bf, axis=0, prepend=bf[:1]), axis=1) * FPS
        bv = np.minimum(np.convolve(bv, np.ones(FPS) / FPS, "same"), 40) * seen
        cols += [bf[:, 0], bf[:, 1], bv, (~seen).astype(float), np.log1p(since / FPS), np.log1p(until / FPS)]
    X = np.stack(cols, 1).astype(np.float32)
    T = len(X) // SUB * SUB
    Xs = np.nanmean(X[:T].reshape(-1, SUB, X.shape[1]), axis=1)
    return Xs   # NaN kept; filled after standardisation


SWAP = np.r_[np.arange(10, 20), np.arange(0, 10), np.arange(20, 26)]


def labels5(half):
    T = len(half["truth"]) // SUB * SUB
    return half["truth"][:T].reshape(-1, SUB)[:, SUB // 2].astype(np.int64)


class Block(nn.Module):
    def __init__(self, c, d, p):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.n = nn.GroupNorm(8, c)
        self.p = nn.Dropout(p)

    def forward(self, x):
        h = self.c2(self.p(F.gelu(self.n(self.c1(x)))))
        return x + h


class TCN(nn.Module):
    def __init__(self, nin=26, c=64, p=0.2):
        super().__init__()
        self.i = nn.Conv1d(nin, c, 1)
        self.b = nn.ModuleList(Block(c, d, p) for d in DIL)
        self.o = nn.Conv1d(c, 3, 1)

    def forward(self, x):          # (B, F, T) -> (B, 3, T)
        h = self.i(x)
        for b in self.b:
            h = b(h)
        return self.o(F.gelu(h))


def prep(X, mu, sd):
    return np.nan_to_num((X - mu) / sd).astype(np.float32)


def train(feats, labs, seed=0, steps=1500, crop=300, bs=64, dev="cuda"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    allX = np.concatenate(feats)
    mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    Z = [torch.tensor(prep(f, mu, sd)).to(dev) for f in feats]
    Y = [torch.tensor(l).to(dev) for l in labs]
    swap = torch.tensor(SWAP).to(dev)
    net = TCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    sizes = np.array([len(z) for z in Z], float)
    for s in range(steps):
        xb, yb = [], []
        for _ in range(bs):
            k = rng.choice(len(Z), p=sizes / sizes.sum())
            a = rng.integers(0, len(Z[k]) - crop)
            x, y = Z[k][a:a + crop], Y[k][a:a + crop]
            if rng.random() < 0.5:   # swap teams: features and H/A labels
                x, y = x[:, swap], torch.where(y == 2, y, 1 - y)
            xb.append(x.T), yb.append(y)
        xb = torch.stack(xb)
        xb = xb + 0.05 * torch.randn_like(xb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(xb)
        loss = -F.log_softmax(out.float(), 1).gather(1, torch.stack(yb)[:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return {"net": net.cpu().eval(), "mu": mu, "sd": sd}


@torch.no_grad()
def predict_logp(model, feats, T):
    """Per-frame (T,3) log-probs at 25 fps from CPU float32 inference."""
    torch.set_num_threads(1)
    z = torch.tensor(prep(feats, model["mu"], model["sd"])).T[None]
    lp = F.log_softmax(model["net"](z)[0].T, dim=1).numpy().astype(np.float64)
    up = np.repeat(lp, SUB, axis=0)
    return np.concatenate([up, np.repeat(up[-1:], T - len(up), axis=0)]) if len(up) < T else up
