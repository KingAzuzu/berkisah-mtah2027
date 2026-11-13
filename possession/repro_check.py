"""Determinism / reproducibility check for the v2 labelling path on one degraded IDSSE tuning half.

    python -m possession.repro_check [--match J03WMX] [--half 0] [--threads 1] [--method v2|v3|v4|v6|v8|v10|v10c1] [--idsse-data PATH]

Prints SHA-256 of the label bytes (int8 codes H 0 / A 1 / D 2) and of the letters string. Only IDSSE
tuning matches are accepted (the locked ones are refused). The half is passed to the pipeline as
ball / xy / team only, exactly the inputs the competition tracking provides. Thread env vars are
forced before numpy is imported. Compare the hashes across thread counts and across clones.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--match", default="J03WMX")
ap.add_argument("--half", type=int, default=0)
ap.add_argument("--method", choices=("v2", "v3", "v4", "v6", "v8", "v10", "v10c1"), default="v2")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--idsse-data", default=os.environ.get("IDSSE_DATA_DIR"),
                help="folder holding the IDSSE xml files and cache/ (default: <repo>/idsse-tests/data)")
args = ap.parse_args()

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(args.threads)  # forced, before numpy / sklearn load

import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from possession import pipeline, run_idsse  # noqa: E402
from possession.data import LOCKED, idsse, pm  # noqa: E402

if args.match in LOCKED:
    sys.exit("refusing: locked match")
if args.idsse_data:
    idsse.DATA = Path(args.idsse_data)

full = run_idsse.load(args.match)[args.half]
half = {k: full[k] for k in ("ball", "xy", "team")}  # no truth
if args.method in ("v3", "v4", "v6", "v8", "v10", "v10c1"):
    codes = getattr(pipeline, f"{args.method}_predictor")()[1](half)
else:
    codes = pipeline.predict(half, pipeline.load_params(), pipeline.load_models())
letters = "".join(np.array(pm.LABELS)[codes])
print(f"python {sys.version.split()[0]} numpy {np.__version__} sklearn {sklearn.__version__} threads {args.threads}")
print(f"match {args.match} half {args.half} frames {len(codes)}  counts H {np.sum(codes == 0)} "
      f"A {np.sum(codes == 1)} D {np.sum(codes == 2)}")
print("sha256 codes   ", hashlib.sha256(np.ascontiguousarray(codes, np.int8).tobytes()).hexdigest())
print("sha256 letters ", hashlib.sha256(letters.encode()).hexdigest())
