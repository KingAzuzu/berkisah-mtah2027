"""Inference: windowed prediction of a whole half, and the deterministic CPU path.

    python -m possession.ballradar.infer          # CPU, deterministic, twice on one half + timing

Windows of W steps (20 s at 5 Hz) stride W/2, combined in a fixed order with a sine weight, in float64.
The 5 Hz outputs are interpolated to 25 fps; the observed ball replaces the prediction wherever it was seen.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .features import STRIDE, prep
from .model import BallRadar

W = 100
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ballradar.pt"
THREADS = 4


def load_model(path=MODEL_PATH, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    m = BallRadar(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def predict5(model, p, device="cpu", batch=16):
    """prepped half -> team probabilities (T5, 3) and ball (T5, 2) at 5 Hz."""
    T5 = len(p["seen"])
    w = min(W, T5)
    starts = list(range(0, T5 - w + 1, w // 2))
    if starts[-1] != T5 - w:
        starts.append(T5 - w)
    t = {k: torch.from_numpy(p[k]).to(device) for k in ("slot", "mask", "ball", "seen")}
    team = torch.from_numpy(p["team"]).long().to(device)
    win = np.sin(np.pi * (np.arange(w) + 0.5) / w) + 0.05
    acc_p, acc_b, acc_w = np.zeros((T5, 3)), np.zeros((T5, 2)), np.zeros(T5)
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        idx = torch.tensor([[s + j for j in range(w)] for s in ss], device=device)
        ball, logits = model(t["slot"][idx], t["mask"][idx], team, t["ball"][idx], t["seen"][idx])
        pr = torch.softmax(logits.double(), -1).cpu().numpy()
        bl = ball.double().cpu().numpy()
        for k, s in enumerate(ss):  # fixed order
            acc_p[s:s + w] += pr[k] * win[:, None]
            acc_b[s:s + w] += bl[k] * win[:, None]
            acc_w[s:s + w] += win
    return acc_p / acc_w[:, None], acc_b / acc_w[:, None]


def upsample(half, p5, b5):
    """5 Hz -> 25 fps. Returns probs (T, 3) float32 and ball (T, 2) float32 (observed ball kept where seen)."""
    T = len(half["ball"])
    x5 = np.arange(len(p5)) * STRIDE
    f = np.arange(T)
    probs = np.stack([np.interp(f, x5, p5[:, c]) for c in range(3)], 1).astype(np.float32)
    ball = np.stack([np.interp(f, x5, b5[:, c]) for c in range(2)], 1).astype(np.float32)
    obs = ~np.isnan(half["ball"][:, 0])
    ball[obs] = half["ball"][obs]
    return probs, ball


def predict_half(model, half, device="cpu"):
    p5, b5 = predict5(model, prep(half), device)
    return upsample(half, p5, b5)


def determinism_check():
    """Two CPU runs on one half of one tuning match must be bit-identical. Loaded here: degraded J03WMX half 1."""
    from ..run_idsse import load
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    model = load_model()
    half = load("J03WMX")[0]
    outs, times = [], []
    for _ in range(2):
        t0 = time.time()
        outs.append(predict_half(model, half))
        times.append(time.time() - t0)
    same = all(np.array_equal(a, b) for a, b in zip(outs[0], outs[1]))
    bytes_same = all(a.tobytes() == b.tobytes() for a, b in zip(outs[0], outs[1]))
    gpu = None
    if torch.cuda.is_available():
        g = predict_half(model.to("cuda"), half, "cuda")
        gpu = float(np.abs(g[0] - outs[0][0]).max())
    res = {"half_frames": int(len(half["ball"])), "threads": THREADS, "torch": torch.__version__,
           "identical": bool(same), "bytes_identical": bool(bytes_same), "cpu_seconds": times,
           "max_abs_prob_diff_cpu_vs_gpu": gpu}
    print(json.dumps(res))
    Path(__file__).resolve().parent.parent.joinpath("logs", "ballradar_infer.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    determinism_check()
