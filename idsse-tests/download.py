"""Download the 7 open IDSSE matches (meta, events, raw positions) from the pysport/idsse-data mirror of the figshare release.

Writes to <name>.part and resumes it with an HTTP Range request after a dropped connection; the file only gets
its real name once its size matches Content-Length, so an interrupted run never leaves a truncated file behind.
"""
import time
import urllib.request

import idsse

BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
TRIES = 20


def fetch(url, dest):
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])
    for attempt in range(TRIES):
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(min(30, 2 ** attempt))
    if part.stat().st_size != total:
        raise SystemExit(f"{dest.name}: got {part.stat().st_size} of {total} bytes after {TRIES} tries")
    part.replace(dest)


idsse.DATA.mkdir(exist_ok=True)
for mid in idsse.MATCHES:
    for kind in ("DFL_02_01_matchinformation", "DFL_03_02_events_raw", "DFL_04_03_positions_raw_observed"):
        p = idsse._path(kind, mid)
        if not p.exists():
            print("downloading", p.name)
            fetch(BASE + p.name, p)
