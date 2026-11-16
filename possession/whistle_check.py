"""Label-free check (human only): is there a referee-whistle signal in the tactical video audio that lines up with our D transitions?

    python -m possession.whistle_check <tactical mp4> <submission csv>      # own terminal; prints aggregates only

Audio decoded with imageio-ffmpeg to 16 kHz mono; STFT window 1024, hop 640 = one value per video frame (25 fps).
Whistle frame: in 2.0-4.5 kHz, peak bin power / median bin power >= TH dB (tonal); event = run >= 5 frames (0.2 s), gaps <= 2 merged.
Fixed beforehand: TH 15 dB is the decision value (12 and 18 printed for robustness only).
Signal gate: share of whistle onsets within +-2 s of a v10 transition (alive->D or D->alive) >= 3x the same share for onsets
shifted by 30..90 s (circular). If FAIL: no usable audio signal, idea closed. Also printed: median lag whistle onset - nearest D onset
(events within 5 s), which says whether our D onsets are systematically late.
"""
import subprocess
import sys
import numpy as np

SR, HOP, WIN, FPS = 16000, 640, 1024, 25


def audio(path):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16).astype(np.float32) / 32768


def tonal_db(x, n_frames):
    f = np.fft.rfftfreq(WIN, 1 / SR)
    band = (f >= 2000) & (f <= 4500)
    w = np.hanning(WIN).astype(np.float32)
    x = np.pad(x, (WIN // 2, WIN))
    out = np.zeros(n_frames, np.float32)
    for s in range(0, n_frames, 20000):
        idx = np.arange(s, min(s + 20000, n_frames))
        seg = np.stack([x[i * HOP:i * HOP + WIN] for i in idx]) * w
        p = np.abs(np.fft.rfft(seg, axis=1))[:, band] ** 2 + 1e-12
        out[idx] = 10 * np.log10(p.max(1) / np.median(p, 1))
    return out


def events(db, th):
    on = db >= th
    ch = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    st, en = list(ch[::2]), list(ch[1::2])
    i = 0
    while i < len(st) - 1:
        if st[i + 1] - en[i] <= 2:
            en[i] = en[i + 1]; del st[i + 1], en[i + 1]
        else:
            i += 1
    return np.array([s for s, e in zip(st, en) if e - s >= 5])


def near(t, ref, w):
    if len(t) == 0 or len(ref) == 0:
        return np.zeros(len(t), bool)
    j = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
    return np.minimum(np.abs(ref[j] - t), np.abs(ref[j - 1] - t)) <= w


def main(mp4, csv):
    from mtah_guard import refuse_under_claude
    refuse_under_claude()
    txt = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=1, dtype=str)
    alive = (txt == "H") | (txt == "A")
    dead = txt == "D"
    d_on = np.flatnonzero(dead[1:] & alive[:-1]) + 1
    d_off = np.flatnonzero(alive[1:] & dead[:-1]) + 1
    trans = np.sort(np.r_[d_on, d_off])
    n = len(txt)
    db = tonal_db(audio(mp4), n)
    print(f"frames {n}; transitions D-on {len(d_on)}, D-off {len(d_off)}; tonal dB p50 {np.median(db):.1f} p99 {np.percentile(db, 99):.1f}")
    for th in (15, 12, 18):
        ev = events(db, th)
        real = near(ev, trans, 2 * FPS).mean() if len(ev) else 0.0
        ctrl = np.mean([near((ev + k * FPS) % n, trans, 2 * FPS).mean() for k in range(30, 91, 10)]) if len(ev) else 0.0
        enr = real / max(ctrl, 1e-9)
        lags = []
        for t in ev:
            if len(d_on):
                k = d_on[np.argmin(np.abs(d_on - t))]
                if abs(k - t) <= 5 * FPS:
                    lags.append((t - k) / FPS)
        cover = near(d_on, ev, 2 * FPS).mean() if len(d_on) else 0.0
        tag = "DECISION" if th == 15 else "info"
        print(f"[{tag}] TH {th} dB: events {len(ev)}, near transition {real:.3f}, shifted control {ctrl:.3f}, enrichment {enr:.2f} "
              f"-> {'PASS' if enr >= 3 else 'FAIL'}; D-onsets with a whistle within 2 s {cover:.3f}; "
              f"lag whistle - D-onset median {np.median(lags) if lags else float('nan'):+.2f} s (n {len(lags)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
