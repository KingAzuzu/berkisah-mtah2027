"""Human only. Stage B of the label-free YOLO check: how accurate is an off-the-shelf ball detector on the tactical video?

Separate venv (keeps the pinned submission venv untouched):
    python -m venv .venv-vision
    .venv-vision\\Scripts\\pip install ultralytics numpy
    .venv-vision\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
    .venv-vision\\Scripts\\python -m possession.yolo_check <tactical mp4> [model, default yolo11x.pt] [imgsz, default 1920]

Reference = ATD ball on frames where it is seen (jitter 1-2 cm, so it is the truth there). Detector: COCO class 32 "sports ball",
conf >= 0.05, never trained on this match (rule #2). Each box centre is mapped to the pitch with the inverse homography.
Prints aggregates only. Gate fixed beforehand (from the v17 simulation: only <= 0.5 m errors on nearly all gaps help):
on ATD-seen frames, share with a detection within 0.5 m >= 0.80. This is optimistic (seen frames are the easy ones), so FAIL closes
the idea for good; PASS means: build v17 (retrain with filled ball) and gate it on IDSSE.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mtah_guard  # noqa: E402

PREP = ROOT / "data" / "stats" / "yolo_prep.npz"


def to_pitch(H, uv, size):
    p = np.linalg.solve(H, np.c_[uv, np.ones(len(uv))].T).T
    return p[:, :2] / p[:, 2:3] * size - size / 2


def main(mp4, model="yolo11x.pt", imgsz=1920):
    mtah_guard.refuse_under_claude()
    import cv2
    from ultralytics import YOLO
    d = np.load(PREP)
    idx, Hs, ball, size = d["idx"], d["H"], d["ball"], d["pitch"]
    want = dict(zip(idx.tolist(), range(len(idx))))
    net = YOLO(model)
    cap = cv2.VideoCapture(mp4)
    import time
    total, t0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1, time.time()
    near = np.full(len(idx), np.inf)
    ndet = np.zeros(len(idx), int)
    far = 0
    batch, bk, f = [], [], 0

    def flush():
        nonlocal far
        if not batch:
            return
        for k, r in zip(bk, net.predict(batch, imgsz=imgsz, conf=0.05, classes=[32], verbose=False)):
            b = r.boxes.xywh.cpu().numpy()
            ndet[k] = len(b)
            if len(b) and not np.isnan(ball[k, 0]):
                dist = np.linalg.norm(to_pitch(Hs[k], b[:, :2], size) - ball[k], axis=1)
                near[k] = dist.min()
                far += int((dist > 5).sum())
        batch.clear(); bk.clear()

    while True:
        ok = cap.grab()
        if not ok:
            break
        if f in want:
            _, img = cap.retrieve()
            batch.append(img); bk.append(want[f])
            if len(batch) == 4:
                flush()
        f += 1
        if f % 5000 == 0:
            el = time.time() - t0
            print(f"{f}/{total} frames {100 * f / total:5.1f}%  elapsed {el / 60:5.1f} min  ETA {el / f * (total - f) / 60:5.1f} min", flush=True)
    flush()
    seen = ~np.isnan(ball[:, 0])
    e = near[seen]
    print(f"video frames read {f}; sampled {len(idx)}; ATD-seen {int(seen.sum())}; model {model} imgsz {imgsz}")
    print(f"ATD-seen frames: detection within 0.5 m {np.mean(e <= 0.5):.3f}, 1 m {np.mean(e <= 1):.3f}, 2 m {np.mean(e <= 2):.3f}, "
          f"any detection {np.mean(ndet[seen] > 0):.3f}; median error of matched (<=2 m) {np.median(e[e <= 2]) if np.any(e <= 2) else float('nan'):.2f} m; "
          f"far (>5 m) detections per frame {far / max(seen.sum(), 1):.3f}")
    print(f"ATD-missing frames: any detection {np.mean(ndet[~seen] > 0):.3f} ({int((~seen).sum())} frames)")
    print("GATE (within 0.5 m >= 0.80):", "PASS" if np.mean(e <= 0.5) >= 0.80 else "FAIL")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(a[1:2] or []), *([int(a[2])] if len(a) > 2 else []))
