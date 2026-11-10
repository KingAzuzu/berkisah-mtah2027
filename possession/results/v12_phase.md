# v12 idea A: phase-shift TTA on v8 (IDSSE TUNE, LOMO, seeds 20-24, v8 decoder)

Fold nets retrained with the v8 recipe (`v12_phase.py`), each run at input phases k = 0..4 frames (features recomputed on the shifted half),
outputs upsampled to 25 fps, shifted back, averaged. `v12_eval.py`, `logs/v12/eval_v8.json`.
Gate fixed before running: TTA vs merged truth mean >= 0.7663 AND beats phase 0 on >= 4/5 matches.

| | vs merged | per match (MX, N1, OH, OY, PY) | vs DFL |
|---|---|---|---|
| phase 0 (retrained, = v8) | 0.7563 | .763 .719 .786 .734 .779 | 0.6517 |
| TTA 5 phases | 0.7545 | .763 .721 .783 .734 .773 | 0.6502 |

Gate FAIL (1/5 wins, -0.0018). Sanity: phase 0 reproduces the v8 per-match scores exactly, so no alignment bug in the baseline.
Boundary errors are around 1 s, not the 0.2 s block size, so sub-block phase averaging cannot fix them. Not adopted, not wired. v6 side not run.
