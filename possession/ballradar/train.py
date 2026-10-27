"""Train the Ball Radar style model on degraded IDSSE tuning matches (GPU).

    python -m possession.ballradar.train --folds [--steps N]     # leave-one-match-out, caches held-out predictions
    python -m possession.ballradar.train --final [--steps N]     # all 5 tuning matches -> possession/models/ballradar.pt
    python -m possession.ballradar.train --bench                 # speed / memory probe, no outputs

Training data: cached 5 Hz arrays of degrade seeds 0..2 (features.py), random 20 s windows, random pitch
flips in x and y, random swap of the home/away labels, and extra random ball-hiding chunks.
Loss: team cross-entropy on every frame + ball Huber (metres) weighted 1 on hidden frames, 0.05 on seen.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import TUNE
from ..run_idsse import load
from . import features as fe
from .infer import MODEL_PATH, W, predict5, upsample
from .model import BallRadar

LOGS = Path(__file__).resolve().parent.parent / "logs"
DEV = "cuda"


class Pool:
    """All training halves concatenated on the GPU; windows never cross a half boundary."""

    def __init__(self, mids, seeds=fe.SEEDS):
        parts = [h for m in mids for s in seeds for h in fe.build(m, s)]
        cat = lambda k: np.concatenate([h[k] for h in parts])
        S = parts[0]["slot"].shape[1]
        assert all(h["slot"].shape[1] == S for h in parts)
        self.t = {k: torch.from_numpy(cat(k)).to(DEV) for k in
                  ("slot", "mask", "ball", "seen", "truth5", "ball_true", "ball_true_ok")}
        self.t["truth5"] = self.t["truth5"].long()
        self.team = torch.from_numpy(parts[0]["team"]).long().to(DEV)
        starts, off = [], 0
        for h in parts:
            n = len(h["seen"])
            starts.append(np.arange(off, off + n - W + 1))
            off += n
        self.starts = torch.from_numpy(np.concatenate(starts)).to(DEV)
        self.n_steps = off

    def batch(self, B, gen):
        s = self.starts[torch.randint(len(self.starts), (B,), device=DEV, generator=gen)]
        idx = s[:, None] + torch.arange(W, device=DEV)
        slot, mask = self.t["slot"][idx].clone(), self.t["mask"][idx]
        ball, seen = self.t["ball"][idx].clone(), self.t["seen"][idx].clone()
        truth, bt, btok = self.t["truth5"][idx].clone(), self.t["ball_true"][idx].clone(), self.t["ball_true_ok"][idx]
        team = self.team[None].expand(B, -1).clone()
        r = lambda: torch.rand(B, device=DEV, generator=gen) < 0.5
        for ax, flip in ((0, r()), (1, r())):  # pitch flips: positions, velocities, ball, target
            sg = torch.where(flip, -1.0, 1.0)
            slot[..., ax] *= sg[:, None, None]
            slot[..., 2 + ax] *= sg[:, None, None]
            ball[..., ax] *= sg[:, None]
            bt[..., ax] *= sg[:, None]
        sw = r()  # swap home/away roles
        team = torch.where(sw[:, None], 1 - team, team)
        truth = torch.where(sw[:, None] & (truth < 2), 1 - truth, truth)
        # extra ball-hiding chunks (2-10 s) on half of the windows
        for _ in range(3):
            on = r() & r() | (torch.rand(B, device=DEV, generator=gen) < 0.15)
            a = torch.randint(0, W, (B,), device=DEV, generator=gen)
            n = torch.randint(10, 51, (B,), device=DEV, generator=gen)
            t = torch.arange(W, device=DEV)[None]
            seen &= ~(on[:, None] & (t >= a[:, None]) & (t < (a + n)[:, None]))
        ball = ball * seen[..., None]
        return slot, mask, team, ball, seen, truth, bt, btok


def loss_fn(model, b):
    slot, mask, team, ball, seen, truth, bt, btok = b
    pb, logits = model(slot, mask, team, ball, seen)
    ce = F.cross_entropy(logits.reshape(-1, 3), truth.reshape(-1))
    hub = F.smooth_l1_loss(pb, bt, beta=2.0, reduction="none").mean(-1)
    w = torch.where(seen, 0.05, 1.0) * btok
    lb = (hub * w).sum() / w.sum().clamp(min=1.0)
    return ce + 0.1 * lb, ce.item(), lb.item()


def train(mids, steps, seed=0, batch=48, lr=1e-3, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    pool = Pool(mids)
    model = BallRadar().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    torch.cuda.reset_peak_memory_stats()
    t0, hist = time.time(), []
    model.train()
    for it in range(1, steps + 1):
        loss, ce, lb = loss_fn(model, pool.batch(batch, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append((ce, lb))
        if it % 250 == 0 or it == steps:
            h = np.mean(hist[-250:], axis=0)
            log(f"  step {it}/{steps} ce {h[0]:.4f} ball_huber {h[1]:.3f} {time.time() - t0:.0f}s")
    info = {"steps": steps, "batch": batch, "train_seconds": time.time() - t0,
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20, "train_steps_pool": pool.n_steps,
            "final_ce": float(np.mean([h[0] for h in hist[-250:]])),
            "final_ball_huber": float(np.mean([h[1] for h in hist[-250:]]))}
    del pool
    return model.eval(), info


def heldout(model, mid):
    """Held-out predictions at 25 fps on the degraded seed-0 halves, cached."""
    out = []
    for half in load(mid):
        p5, b5 = predict5(model, fe.prep(half), DEV)
        probs, ball = upsample(half, p5, b5)
        out.append({"probs": probs, "ball": ball})
    (fe.CACHE / f"heldout_{mid}.pkl").write_bytes(pickle.dumps(out, protocol=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--only", nargs="+", default=list(TUNE))
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    pr = lambda s: print(s, flush=True)
    pr(f"torch {torch.__version__} cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if a.bench:
        _, info = train(TUNE[1:], 100, log=pr)
        pr(json.dumps(info))
    if a.folds:
        for m in a.only:
            pr(f"fold: hold out {m}")
            model, info = train([x for x in TUNE if x != m], a.steps, log=pr)
            heldout(model, m)
            info["held_out"] = m
            (LOGS / f"ballradar_train_{m}.json").write_text(json.dumps(info, indent=1))
            pr(json.dumps(info))
    if a.final:
        pr("final: all tuning matches")
        model, info = train(list(TUNE), a.steps, log=pr)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        torch.save({"cfg": model.cfg, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "torch": torch.__version__, "steps": a.steps}, MODEL_PATH)
        (LOGS / "ballradar_train_final.json").write_text(json.dumps(info, indent=1))
        pr(json.dumps(info))


if __name__ == "__main__":
    main()
