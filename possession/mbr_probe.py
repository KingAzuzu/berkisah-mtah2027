"""Timing and sanity probe for mbr.py on one degraded IDSSE half (tuning match only)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import mbr, safety
from .data import TUNE
from .run_idsse import load
from .score import score

params = json.loads(Path(__file__).with_name("params.json").read_text())
p = {**safety.PARAMS, **params}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
m = TUNE[0]
halves = load(m)
h = halves[0]
t0 = time.time()
e = safety.emissions(safety.features(h), p)
print("T", len(e), "features+emissions %.1fs" % (time.time() - t0))
t0 = time.time()
al, be, lz = mbr.forward_backward(e / temp, p["SWITCH_TEAM"], p["SWITCH_DEAD"])
print("fwd/bwd %.1fs" % (time.time() - t0))
# check: Viterbi score <= logZ and marginals sum to 1
g = mbr.marginals(al, be, lz)
print("marginal sums min/max", g.sum(1).min(), g.sum(1).max())
t0 = time.time()
z = mbr.ffbs(al, p["SWITCH_TEAM"], p["SWITCH_DEAD"], n, 0)
print("ffbs n=%d %.1fs" % (n, time.time() - t0), "spells/sample", np.mean([len(mbr.spells_of(s)) for s in z]))
z2 = mbr.ffbs(al, p["SWITCH_TEAM"], p["SWITCH_DEAD"], n, 0)
print("deterministic", bool((z == z2).all()))
v = safety.viterbi(e / temp, p["SWITCH_TEAM"], p["SWITCH_DEAD"])
print("viterbi-vs-sample frame agreement", np.mean([(s == v).mean() for s in z]))
mbr.check_fast_overlap([v, z[0], z[1], g.argmax(1).astype(np.int8)])
print("fast overlap == pm.spell_scores: ok")
info = {}
t0 = time.time()
out = mbr.decode_variants(e, p["SWITCH_TEAM"], p["SWITCH_DEAD"], temp, n, (0.5, 1.0, 2.0), (0.0, 0.3), info=info)
print("decode_variants %.1fs" % (time.time() - t0), info)
for k, path in out.items():
    r = score([h], [path]) if False else None
