# WS4: dead-ball classifier

Files: `possession/dead.py` (features, model factory, emissions), `possession/dead_experiment.py` (LOMO + nested
tuning + final model), `possession/dead_ablate.py` (players-only ablation), `possession/models/dead.joblib`,
logs `possession/logs/dead.json`, `dead_ablate.json`, `dead_run.txt`.

## Commands (from D:\barca, venv python)
    python -m possession.dead_experiment     # 336 s, joblib n_jobs=4, OMP threads 4
    python -m possession.dead_ablate

## What was built
- 241 per-frame features from the degraded half only: player speed (mean, median, q75, q90, shares <1, <2, >4 m/s),
  counts near touchlines, corner arcs, goal areas, penalty areas, centre circle (all direction-invariant, |x| |y|),
  team centroids/spreads/"whole team on one side" and the sign product of the two centroids (kickoff), ball seen,
  ball speed (short sighting gaps only), ball out of bounds, ball distance to line/corner/centre, nearest-player
  distances, frames since / until the ball was seen. 1, 3, 10 s windows, past and future, plus speed steps.
- HistGradientBoostingClassifier (200 iter, lr 0.08, 15 leaves, random_state=0), trained on every 5th frame.
- Emissions (`dead.emissions`): team evidence exactly as `safety.emissions`; D log-odds = clip(alpha * logit(p_dead)),
  the speed/out terms are zeroed. Decoded with `safety.viterbi`.
- Tuning is nested: for held-out match m, (alpha, SWITCH_TEAM, SWITCH_DEAD) chosen on the other 4 matches using
  p_dead from models that saw neither m nor that match (trained on 3). Grid alpha {0.5,1,1.5,2} x team {10,20,30}
  x dead {5,10,20,40}. Held-out score uses the model trained on the 4 others.

## Frame-level classifier (LOMO, threshold 0.5)
| held-out | AUC | D F1 | D share | AUC of -speed alone | AUC, players-only features | D F1, players-only |
|---|---|---|---|---|---|---|
| J03WMX | 0.9994 | 0.9859 | 0.415 | 0.931 | 0.9856 | 0.9226 |
| J03WN1 | 0.9992 | 0.9863 | 0.517 | 0.928 | 0.9780 | 0.9199 |
| J03WOH | 0.9994 | 0.9881 | 0.476 | 0.942 | 0.9863 | 0.9352 |
| J03WOY | 0.9993 | 0.9868 | 0.465 | 0.925 | 0.9793 | 0.9050 |
| J03WPY | 0.9994 | 0.9855 | 0.443 | 0.937 | 0.9883 | 0.9397 |

## LOMO spellF1 vs reference (reference from `logs/reference_safety.json`)
| match | ref spell | WS4 spell | ref macro | WS4 macro | spells pred/gt ref | spells pred/gt WS4 | chosen alpha/team/dead |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.3002 | 0.3651 | 0.7784 | 0.8681 | 359/511 | 280/511 | 2.0/20/40 |
| J03WN1 | 0.2804 | 0.3489 | 0.7402 | 0.8309 | 366/500 | 299/500 | 1.5/30/5 |
| J03WOH | 0.3203 | 0.4185 | 0.7628 | 0.8478 | 351/402 | 289/402 | 1.5/20/5 |
| J03WOY | 0.2814 | 0.3531 | 0.7326 | 0.8426 | 357/456 | 294/456 | 1.5/20/5 |
| J03WPY | 0.3259 | 0.4239 | 0.7728 | 0.8699 | 351/421 | 281/421 | 1.5/30/5 |
| mean | 0.3017 | **0.3819** [0.3489, 0.4239] | 0.7573 | 0.8519 | 1784/2290 | 1443/2290 | |

All 5 matches improve (+0.065 to +0.102 spellF1, above the ~0.04 noise level), macroF1 +0.095.
Fixed-cell sensitivity on the held-out predictions (mean spellF1, context only, in `dead.json`): best
(1.5, 20, 5) 0.386, worst (0.5, 30, 40) 0.366; the result is not knife-edge in the grid.

## Top permutation importances (fold J03WMX, frame AUC drop, base AUC 0.9994)
b_speed_f3 0.0025, b_speed_f1 0.0019, b_speed_p3 0.0017, b_speed_p1 0.0016, b_speed 0.0005, b_speed_p10 0.00015,
b_speed_f10 0.00013, spd_lt2_p1 0.00008, b_line_d_p1 0.00007, spd_med_f10 0.00007. Ball speed dominates.

## Fewer spells than truth
Predicted spells fell (1784 -> 1443 of 2290 truth). The classifier makes dead segments clean, but possession
changes inside alive play are still the nearest-player evidence from `safety`; WS1/WS3 address that.

## Unverified / caveats
- Ball speed dominates, AUC 0.999. Players-only features still give AUC 0.978-0.988. In IDSSE the ball is very
  likely stationary or frozen while dead (a property of the DFL data); I did not check this. A video-derived ball
  track on the real target data may behave differently when dead (jitter, occlusion, ball seen on 59% of frames),
  so the IDSSE AUC may overstate performance there. `degrade.py` only masks the ball, it does not add noise to
  the ball position, so ball speed is not stress-tested. A robustness test (position noise on the ball) is not done.
- Nested tuning leaves no label leak into chosen parameters, but the GBM hyper-parameters were fixed by hand once,
  not searched; the locked matches were never loaded.
- Pitch dimension constants are IDSSE's; the target pitch size and coordinate frame are not checked here.
- Permutation importance is on one fold only. Single degrade seed (0).
- `dead.joblib` (all 5 tuning matches, 241 features, sklearn 1.9.1, numpy 2.5.4, joblib 1.6.0) is saved but no
  separate CPU-determinism rerun test was run; HGB with fixed random_state and no early stopping should be
  deterministic (unverified across thread counts).
- Alpha and switch costs were not frozen (choices varied by fold: alpha 1.5-2.0, team 20-30, dead 5-40); the best
  fixed cell on all folds is (1.5, 20, 5), selected with test folds included, so treat 0.386 as optimistic.
