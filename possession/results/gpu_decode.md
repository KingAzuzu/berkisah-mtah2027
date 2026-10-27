# GPU-batched decoders for tuning (status: PARTIAL, large GPU runs paused)

Files: `possession/gpu_decode.py`, `possession/gpu_tune.py`, `tests/test_gpu_decode.py`,
`possession/logs/gpu_decode_exact.json`. The final submission stays CPU; this is for tuning only.

## What was built

- `gpu_decode.viterbi(e, st, sd, lengths=None, dtype)`: same model as `safety.viterbi` (cost SWITCH_TEAM for
  H<->A, SWITCH_DEAD for alive<->D, first-index tie-break). Forward pass on GPU (3 kernels per frame,
  vectorised over cells), back-pointers computed vectorised afterwards, backtrack on CPU in numpy over all cells.
- `gpu_decode.posterior_argmax(e, st, sd, temp, lengths, dtype)`: log-space forward-backward on the same HMM as
  `mbr.forward_backward` (emissions / temp, costs unscaled), `exp(alpha+beta-logz)`, first-index argmax.
- Batch dimension = cells; evidence either shared `(T,3)` or per-cell `(B,T,3)`; padded halves via `lengths`.
- `gpu_tune.evaluate_grid_gpu`: mirrors `lomo.evaluate_grid` (same table, LOMO selection, report), decoding
  all cells of a half in GPU batches, scoring with `score.score`. Written, NOT yet run (see below).

## Exactness (measured, `python -m tests.test_gpu_decode --exact --cells 8`)

Match: first tuning match (`TUNE[0]`), both degraded halves (T = 70,708 and 75,259), 8 random cells per half
(SCALE, V0, VS, SWITCH_TEAM, SWITCH_DEAD, TEMP varied), compared to `safety.viterbi` and to
`mbr.forward_backward` + `mbr.marginals` + `np.argmax`. Run while another process held the GPU (97% util).

| half | dtype | Viterbi frames differing | posterior-argmax frames differing |
|---|---|---|---|
| 0 | float64 | 0 of 8 cells x 70,708 | 0 |
| 1 | float64 | 0 of 8 cells x 75,259 | 0 |
| 0 | float32 | 114 (8/8 cells), max rel. score diff 4.6e-07 | 837, largest top-2 posterior gap at a differing frame 0.025 |
| 1 | float32 | 100 (8/8 cells), max rel. score diff 2.6e-07 | 1,125, largest gap 0.163 |

float64 is exact on every frame, so no near-tie explanation was needed. float32 is NOT exact (differences are
rounding, not exact ties, and the argmax gaps are not tiny); use float32 only for screening, never for final choices.
Small tests (`--small`: 6,000-frame crop, 4 cells, plus a padded two-half batch with unequal lengths) pass.
pytest is not installed in the venv (not installing); the `test_*` functions are plain and pytest-compatible.

## CPU baseline (measured, half 0, T = 70,708, one process, other jobs running)

- `safety.viterbi`: 0.151 s per cell.
- `mbr.forward_backward`: 0.350 s per cell.
- Linear extrapolation (not measured): N = 16 / 64 / 256 cells -> Viterbi 2.4 / 9.7 / 38.7 s; forward-backward 5.6 / 22.4 / 89.6 s.

## Not yet run (paused on the coordinator's request: the GPU is shared with Ball Radar training)

- GPU timings for N = 16, 64, 256 (`python -m tests.test_gpu_decode --bench 16 64 256 --note "..."`,
  writes `logs/gpu_decode_bench.json`). An attempt was started and stopped before producing any number.
- Reference reproduction (`python -m possession.gpu_tune --reference`, target LOMO spellF1 0.3017 from
  `logs/reference_safety.json`) and the 96-cell GPU vs CPU wall-time comparison (`--grid-bench`).

## Unverified

- Any GPU speedup number; whether `gpu_tune` reproduces 0.3017; float32 speed.
- Exactness was checked on one match, 8 cells per half, not the full grid space.
- Large-B memory behaviour (B = 256, float64 needs roughly 2 GB) is untested.
