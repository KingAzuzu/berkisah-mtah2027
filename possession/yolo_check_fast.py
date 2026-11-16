"""Human only. Fast version of yolo_check (same gate): every 50th frame, 640x640 crop around the ball, ffmpeg decode, live progress.

    .venv-vision\\Scripts\\pip install imageio-ffmpeg
    .venv-vision\\Scripts\\python -m possession.yolo_check_fast <tactical mp4> [model, default yolo11x.pt]

Crop centre: the ATD ball projected to pixels when seen, else the last seen projection (crop ~ a tracker's search window).
On ATD-seen frames this is mildly optimistic (the detector looks in the right place); a real filler would centre the crop on a
Kalman prediction. Gate unchanged: on ATD-seen frames, share with a detection within 0.5 m >= 0.80.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mtah_guard  # noqa: E402
from possession.yolo_check import PREP, to_pitch  # noqa: E402

STEP, C = 50, 640


def main(mp4, model="yolo11x.pt"):
    mtah_guard.refuse_under_claude()
    import cv2
    import imageio_ffmpeg
    from ultralytics import YOLO
    d = np.load(PREP)
    keep = d["idx"] % STEP == 0
    idx, Hs, ball, size = d["idx"][keep], d["H"][keep], d["ball"][keep], d["pitch"]
    want = dict(zip(idx.tolist(), range(len(idx))))
    cap = cv2.VideoCapture(mp4)
    W, Hh, total = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    cap.release()
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-threads", "0", "-i", mp4, "-vf", f"select=not(mod(n\\,{STEP}))",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * Hh * 3 * 4)
    net = YOLO(model)
    near = np.full(len(idx), np.inf)
    ndet = np.zeros(len(idx), int)
    far, last = 0, np.array([W / 2, Hh / 2])
    crops, meta = [], []
    t0 = time.time()

    def flush():
        nonlocal far
        if not crops:
            return
        for (k, ox, oy), r in zip(meta, net.predict(crops, imgsz=C, conf=0.05, classes=[32], verbose=False)):
            b = r.boxes.xywh.cpu().numpy()
            ndet[k] = len(b)
            if len(b) and not np.isnan(ball[k, 0]):
                dist = np.linalg.norm(to_pitch(Hs[k], b[:, :2] + [ox, oy], size) - ball[k], axis=1)
                near[k] = dist.min()
                far += int((dist > 5).sum())
        crops.clear(); meta.clear()

    j = 0
    while True:
        buf = proc.stdout.read(W * Hh * 3)
        if len(buf) < W * Hh * 3:
            break
        n = j * STEP
        j += 1
        if n not in want:
            continue
        k = want[n]
        if not np.isnan(ball[k, 0]):
            p = Hs[k] @ np.r_[(ball[k] + size / 2) / size, 1.0]
            if p[2] != 0 and np.all(np.isfinite(p)):
                last = p[:2] / p[2]
        ox = int(np.clip(last[0] - C / 2, 0, W - C)); oy = int(np.clip(last[1] - C / 2, 0, Hh - C))
        img = np.frombuffer(buf, np.uint8).reshape(Hh, W, 3)
        crops.append(np.ascontiguousarray(img[oy:oy + C, ox:ox + C])); meta.append((k, ox, oy))
        if len(crops) == 16:
            flush()
        if j % 200 == 0:
            el = time.time() - t0
            print(f"{n}/{total} frames {100 * n / total:5.1f}%  elapsed {el / 60:4.1f} min  ETA {el / max(n, 1) * (total - n) / 60:4.1f} min", flush=True)
    flush()
    proc.wait()
    seen = ~np.isnan(ball[:, 0])
    e = near[seen]
    print(f"sampled {len(idx)}; ATD-seen {int(seen.sum())}; model {model} crop {C}; {(time.time() - t0) / 60:.1f} min")
    print(f"ATD-seen frames: detection within 0.5 m {np.mean(e <= 0.5):.3f}, 1 m {np.mean(e <= 1):.3f}, 2 m {np.mean(e <= 2):.3f}, "
          f"any detection {np.mean(ndet[seen] > 0):.3f}; median error of matched (<=2 m) "
          f"{np.median(e[e <= 2]) if np.any(e <= 2) else float('nan'):.2f} m; far (>5 m) detections per frame {far / max(seen.sum(), 1):.3f}")
    print(f"ATD-missing frames: any detection {np.mean(ndet[~seen] > 0):.3f} ({int((~seen).sum())} frames)")
    print("GATE (within 0.5 m >= 0.80):", "PASS" if np.mean(e <= 0.5) >= 0.80 else "FAIL")


if __name__ == "__main__":
    main(*sys.argv[1:3])
