# Matched degrade profile and v2.1 decision

Code: `possession/degrade_matched.py` (profile), `matched_check.py` (sim vs target), `matched_run.py` (LOMO),
`freeze_v2_1.py`, `test_v2_1.py`. The old assumed profile (`degrade.py`) is unchanged. Seed 0. TUNE matches only.

## Constants
Ball: alternating seen/missing runs, lognormal. Seen median 13.8 s sigma 1.15. Missing median 9 s sigma 0.75, plus a
12% short-gap component (median 0.4 s). Renewal seen share 0.80, then whole dead-ball (truth D) episodes hidden w.p. 0.5
and outside-pitch episodes w.p. 0.6 (uses IDSSE truth, simulation only; this is an ASSUMPTION chosen so that the ball
speed median and outside-pitch share match the real aggregates). 24 candidates, the one with most crowded missing frames wins.
Jitter sd 0.01 m. Players: per player alternating tracklets (lognormal median 35 s sigma 1.75) and gaps (median 3.5 s, sigma 1.4),
assigned to free slots of 23 per team. Two tuning iterations only (scope cut).

## Simulator vs target (5 TUNE matches pooled)
| stat | sim | target |
|---|---|---|
| ball seen | 0.569 | 0.593 |
| seen run mean / p25 / p50 / p75 / p90 (s) | 23.4 / 5.7 / 12.4 / 25.6 / 52.4 | 25.3 / 5.5 / 13.8 / 30.3 / 61.7 |
| missing run mean / p10 / p25 / p50 / p75 / p90 / p99 (s) | 17.9 / 1.36 / 5.1 / 11.4 / 23.0 / 40.7 / 100 | 17.2 / 0.47 / 6.1 / 10.7 / 21.1 / 33.8 / 104.8 |
| tracklet mean / p10 / p25 / p50 / p75 / p90 (s) | 114 / 3.7 / 10.7 / 34 / 106 / 282 | 107 / 2.6 / 7.9 / 35.5 / 145 / 285 |
| objects per frame | 20.2 (p10 19) | 20.49 (p10 18) |
| ball step median m/frame; speed median / p99 m/s | 0.161; 4.0 / 24.2 | 0.203; 5.1 / 24.6 |
| share > 40 m/s | 0.0007 | 0.0023 |
| seen > 1 m outside pitch | 0.0297 | 0.0268 |
| nearly stationary speed median / p90 | 0.43 / 0.80 | 0.32 / 1.75 |

Misses beyond 30%: missing-run p10, tracklet p10/p25/p75, >40 m/s share, stationary p90. Slots used 46 vs 43.
Run counts are higher than real because 5 matches are pooled.

## LOMO under the matched profile (spellF1 per match J03WMX, N1, OH, OY, PY)
| candidate | spellF1 mean [min, max] | macroF1 mean | per match |
|---|---|---|---|
| (a) v1 safety | 0.2961 [0.2581, 0.3276] | 0.7509 | .292 .315 .328 .258 .288 |
| (b) v2 frozen (dead model from old-profile data, other 4 matches) | 0.4300 [0.4131, 0.4628] | 0.8138 | .414 .413 .463 .422 .438 |
| (c) players-only retrained on matched | 0.4280 [0.4126, 0.4525] | 0.8137 | .416 .413 .452 .421 .437 |
| (c) with-ball retrained on matched | 0.4891 [0.4683, 0.5192] | 0.8547 | .468 .493 .519 .476 .489 |

Alpha, switch costs, rule params fixed at v2 values (step (d) retune skipped by scope cut). For (b) the frozen all-5 model
was not used, because it saw the held-out match; the same recipe was refit on the other 4 old-profile matches.

## Decision
Rule: adopt if mean spellF1 beats (b) by >= 0.02 and improves on >= 4 of 5. With-ball retrained: +0.059, improves 5 of 5.
Adopted as v2.1: `params_v2_1.json`, `models/dead_v2_1.joblib` (trained on all 5 TUNE matches, matched profile), `--method v2_1`.

Caveats: the with-ball gain depends on the ball-missing structure in the simulator, in particular that the real tracker loses
dead-ball and outside-pitch ball (inferred from the outside share 2.7% and median ball speed 5.1 m/s, not measured directly).
Switch costs and alpha were not re-tuned. Not run on BAR-ATH; locked matches untouched.
