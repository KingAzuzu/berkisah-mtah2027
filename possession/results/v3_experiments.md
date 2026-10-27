# v3 candidate experiments on top of frozen v2

IDSSE tuning matches only, seed-0 degraded halves (`run_idsse.load`); the locked matches and competition data were never loaded.
Every number is from the commands below; raw output is in `possession/logs/v3_*.json|txt`.
Match order in every table: MX N1 OH OY PY (truth spells 511 500 402 456 421). Scoring only through `possession/score.py`. Single-match noise reference ~0.04.

Files (new): `v3_lib.py`, `v3_post.py` (snap, cleanup), `v3_candidates.py`, `v3_combine.py`, `degrade2.py`, `v3_robust.py`, `v3_robust_stats.py`, `params_v3_candidate.json`, `logs/v3_*`.

## Protocol

- v2 = WS2 rules evidence + players-only dead model (D column) + Viterbi. Held-out match m: v2 fold cell (from `integrate_s1_players_seed0.json`, chosen on the other 4), p_dead from the model trained on the other 4 matches (cached by `integrate_run.pdead_tables`).
- Candidate parameters are chosen per held-out match on the inner folds (the 4 other matches, p_dead from models that saw neither match, m's v2 cell). Grids list the no-op cell first, so ties go to "leave v2 alone".
- Check: `python -m possession.v3_candidates --cand base` reproduces v2 exactly: 0.4367 [0.3822, 0.4970], macroF1 0.8232, spells 304 340 296 329 313.
- Per-cell tables (outer score of every grid cell, descriptive only) are in the logs under `table`.

## Summary: each candidate vs v2 (LOMO, nested choice of the candidate parameters)

| run | spellF1 mean | min | max | per match spellF1 | macroF1 | per match macro | spells pred (gt 511 500 402 456 421) |
|---|---|---|---|---|---|---|---|
| v2 | 0.4367 | 0.3822 | 0.4970 | .448 .410 .446 .382 .497 | 0.8232 | .850 .787 .832 .793 .854 | 304 340 296 329 313 |
| (a) boundary snap | 0.4367 | 0.3822 | 0.4970 | identical to v2 | 0.8232 | identical | identical (no-op chosen on every fold) |
| (b) cleanup | 0.4368 | 0.3814 | 0.4970 | .448 .411 .447 .381 .497 | 0.8227 | .850 .787 .831 .793 .854 | 296 335 286 320 313 |
| (c) restarts, jitter 0.5 s, seed 0 | 0.4489 | 0.3965 | 0.5131 | .469 .404 .462 .396 .513 | 0.8306 | .860 .786 .837 .809 .862 | 311 335 296 339 311 |
| (c) restarts, jitter 0.5 s, seed 1 | 0.4426 | 0.3889 | 0.5106 | .462 .391 .460 .389 .511 | 0.8292 | .858 .785 .837 .806 .860 | 304 341 291 340 313 |
| (c) restarts, jitter 1.5 s, seed 0 | 0.4329 | 0.3776 | 0.4995 | .439 .398 .451 .378 .499 | 0.8245 | .848 .780 .834 .807 .854 | 297 323 294 328 315 |
| (c) restarts, jitter 1.5 s, seed 1 | 0.4365 | 0.3888 (OY) | 0.4951 | .451 .404 .443 .389 .495 | 0.8261 | (log) | 306 336 295 331 319 |

Differences vs v2 per match (MX N1 OH OY PY):
- (b): +.000 +.001 +.001 -.001 +.000 (mean +0.0001; fold choice L 25, K 0 on 4 folds, none on PY).
- (c) 0.5 s seed 0: +.021 -.007 +.016 +.014 +.016 (mean +0.0122); seed 1: +.014 -.019 +.014 +.007 +.014 (mean +0.0058).
- (c) 1.5 s seed 0: -.008 -.013 +.004 -.005 +.002 (mean -0.0038); seed 1: +.004 -.006 -.003 +.007 -.001 (mean -0.0002).

### (a) Boundary snap: null
Grid N {0, 10, 25, 50, 75} frames x R {0.75, 1.0, 1.5, 2.5} m; snap each direct H|A boundary to the nearest arrival frame (ball seen, new team's nearest player within R, not within R the frame before), inside the neighbouring runs. Command `-m possession.v3_candidates --cand a`.
The no-op was chosen on all 5 folds. Outer mean over cells: 0.4367 (no-op), 0.4366 (N 10 to 75, R 1.0 and N10 R0.75), down to 0.4329 (N 75, R 2.5); larger R is worse, no cell above v2. Only MX..OY are unchanged to 3 decimals everywhere; the PY match moves (0.497 to 0.478 at worst). The metric matches whole possessions at IoU 0.3 to 0.7, so a shift of tens of frames rarely changes a match, and the WS2 evidence already anchors team changes at arrival. Not worth anything.

### (b) Cleanup: null
Grid K {0, 25, 50, 100, 150, 250} x L {0, 10, 25, 50, 75}. Best cell (K 0, L 25) 0.4370, i.e. +0.0003 (below any reasonable margin, 2 matches +.001, one -.001). Merging H-D-H gaps hurts monotonically: K 25 0.4353, K 100 0.4319, K 250 0.4224 (L 0); K 100 with L 75 0.4032. Predicted spell counts are already below truth, so removing fragments cannot help much and merging removes real possessions. Command `-m possession.v3_candidates --cand b`.

### (c) Restart evidence: small, only when the tags are timed well
Simulated tags, jitter 0.5 s and 1.5 s, drop 15 %, seeds 0 and 1, throw-in team from `restarts.throw_team` (tag counts 432 / 421 / 428 / 419 of 494 events; throw-in wrong team 2 / 8 / 2 / 11, none 14 / 5 / 14 / 7). Adjustment `restarts.adjust` on the v2 evidence (after the dead damping), grid PRE {75, 200, 400} x W_D {0, 1, 2, 5} x POST {25, 75} x W_T {0, 1, 3} (W_D 1 added because v2's D column is on a different scale from the WS2 base used in restarts.md). Command `-m possession.v3_candidates --cand c --sigma S --seed K`.
- Fold choice at 0.5 s is PRE 200, W_D 1, POST 75, W_T 1 on all 5 folds (both seeds); at 1.5 s it varies by fold (W_D 0 on one fold, POST 25).
- Best fixed cell by pooled inner score: PRE 200, W_D 1, POST 75, W_T 1 for 0.5 s (outer mean 0.4489 / 0.4426); at 1.5 s the cells were (POST 25 seed 0) and (POST 75 seed 1) with outer means 0.4359 / 0.4422 (-0.0008 / +0.0055 vs v2).
- Verdict: +0.012 and +0.006 at 0.5 s, 4 of 5 matches positive (N1 negative in both seeds, -.007 and -.019); about 0 at 1.5 s. Gains are below the 0.04 single-match noise, N1 loses, and the sign depends on the tag timing and seed. Not strong enough to adopt without knowing the real tag timing.

## Combination

By the rule "improves LOMO by a margin that holds on most matches": (a) no change, (b) +0.0001 (no), (c) +0.012 / +0.006 at 0.5 s but -0.004 / 0.000 at 1.5 s (fails across jitters, margin under noise). **No candidate qualifies, so the recommended v3 equals v2.** `params_v3_candidate.json` = v2 params with `snap` and `cleanup` null and an optional, not adopted restarts block.

For completeness, c (fixed inner-best cell per config) followed by b (K 0, L 25), outer models, v2 fold cells (`-m possession.v3_combine`; c alone in this fixed-cell form differs slightly from the nested LOMO above at 1.5 s):

| run | spellF1 mean | min | max | per match | macroF1 | spells pred |
|---|---|---|---|---|---|---|
| c 0.5 s seed 0 (fixed cell) | 0.4489 | 0.3965 | 0.5131 | .469 .404 .462 .396 .513 | 0.8306 | 311 335 296 339 311 |
| c + b, 0.5 s seed 0 | 0.4507 | 0.3959 | 0.5160 | .470 .403 .469 .396 .516 | 0.8302 | 302 329 286 330 303 |
| c 1.5 s seed 0 (fixed cell) | 0.4359 | 0.3832 | 0.4948 | .446 .405 .451 .383 .495 | 0.8259 | 306 338 294 332 313 |
| c + b, 1.5 s seed 0 | 0.4370 | 0.3840 | 0.4964 | .445 .404 .455 .384 .496 | 0.8253 | 296 330 286 320 305 |
| c 0.5 s seed 1 | 0.4426 | 0.3889 | 0.5106 | .462 .391 .460 .389 .511 | 0.8292 | 304 341 291 340 313 |
| c + b, 0.5 s seed 1 | 0.4439 | 0.3888 | 0.5114 | .462 .390 .467 .389 .511 | 0.8287 | 296 333 280 330 304 |
| c 1.5 s seed 1 (fixed cell) | 0.4422 | 0.4036 | 0.4959 | .451 .404 .456 .404 .496 | 0.8271 | 306 336 293 332 317 |
| c + b, 1.5 s seed 1 | 0.4435 | 0.4031 | 0.4967 | .455 .403 .460 .403 .497 | 0.8267 | 293 333 283 323 311 |

Adding b to c gains +0.001 to +0.002, the same order as b alone: nothing.

## (d) Robustness of v2 under a more realistic degrade (`degrade2.py`)

Perturbations applied on top of the seed-0 degraded halves, each separately, seeded:
- ball noise: Gaussian sigma 0.3 m on observed ball x, y.
- dead-ball missing: extra ball loss on truth-D frames (gap chain, mean 2 s) until the missing rate over dead frames is 70 % (measured after: 0.698 to 0.704 per match; before 0.359 to 0.383; all-frame missing rate 0.406 to 0.408 becomes 0.542 to 0.571).
- wrong team: 2 % of tracklets (maximal runs of an occupied slot) moved to a free slot of the other team (about 45 runs per match; `v3_robust_stats` counts 91 to 92 slots touched, source plus target).

Modes: mismatch = clean-trained fold models and clean v2 fold cells scored on the perturbed halves; matched = p_dead models retrained inside folds on the perturbed halves (cells stay the clean fold cells, not re-tuned). Commands `-m possession.v3_robust --cond ... --mode mismatch|matched`.

| condition | mode | spellF1 mean | min | max | per match | macroF1 | spells pred |
|---|---|---|---|---|---|---|---|
| clean (v2) | mismatch | 0.4367 | 0.3822 | 0.4970 | .448 .410 .446 .382 .497 | 0.8232 | 304 340 296 329 313 |
| ball noise 0.3 m | mismatch | 0.4226 | 0.3776 | 0.4755 | .413 .396 .451 .378 .476 | 0.8184 | 320 341 319 331 315 |
| ball noise 0.3 m | matched | 0.4226 | 0.3776 | 0.4755 | identical | 0.8184 | identical |
| dead missing 70 % | mismatch | 0.4342 | 0.3792 | 0.4950 | .444 .411 .442 .379 .495 | 0.8218 | 302 341 295 331 312 |
| dead missing 70 % | matched | 0.4342 | 0.3792 | 0.4950 | identical | 0.8218 | identical |
| wrong team 2 % | mismatch | 0.4173 | 0.3762 | 0.4670 | .429 .388 .427 .376 .467 | 0.8150 | 324 338 306 352 325 |
| wrong team 2 % | matched | 0.4111 | 0.3718 | 0.4581 | .424 .385 .417 .372 .458 | 0.8141 | 325 337 318 353 323 |

Changes vs clean: ball noise -0.014, dead missing -0.003, wrong team -0.019 (mismatch) / -0.026 (matched). All are within the 0.04 single-match noise note; all 5 matches fall under wrong team (-.006 to -.030 mismatch), and macroF1 moves at most 0.008. Matched equals mismatch for the two ball perturbations because the dead model is players-only (features do not read the ball); only the rules evidence reads it. Retraining on wrong-team data does not help (0.4111 < 0.4173): the loss comes from the rules team evidence.
v2 + b (K 0, L 25), mismatch (tag v2b25): clean 0.4370, ball noise 0.4231, dead missing 0.4345, wrong team 0.4168. Same as v2 within 0.0005.

## Recommended combination

v2 unchanged. Neither (a) nor (b) is worth adding; (c) is optional and only when the real tag timing is about 0.5 s (cell PRE 200, W_D 1, POST 75, W_T 1, recorded as not adopted in `params_v3_candidate.json`). v2 stays the frozen submission.

## Unverified

- Restart tags are simulated from DFL events (instants, 15 % dropped, Gaussian jitter, true teams on non-throw-in tags, no false positives); real Smart Tagging timing, intervals and error are unknown. Throw-in team rule accuracy was measured only on IDSSE.
- (c) was not run under the (d) perturbations; no candidate was adopted, so robustness was measured for v2 and v2 + b only.
- Nested choice covers the candidate parameters only; v2's own cells are the logged fold cells. Fold-chosen c at 1.5 s differs from the fixed-cell numbers (table above), a sign that the cell is unstable there.
- (a) uses the seen ball only; with the real, sparser ATD ball (59 % seen, already simulated) or a real 0.6 m arrival effect it may behave differently. IDSSE truth `BallPossession` flip timing is a DFL convention.
- degrade2 is an assumption model (white ball noise, extra dead-ball loss placed with a gap chain using IDSSE truth, wrong-team assignment at run level); wrong-team runs merge adjacent tracklets of a slot, so true tracklet counts may be higher. Matched mode did not re-tune the decoder cell.
- Single seed-0 degraded data for all of (a), (b), (d); the seed spread of v2 was 0.013 in integration.md.
- Locked matches not evaluated; competition data and `bar_ath.py` not touched; nothing committed.
