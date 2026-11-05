# v5 study: Smart Tagging team prior in the Viterbi emissions (result: FAIL, nothing frozen)

Code: `possession/tagfuse.py` (codes table, simulator, fusion bias), `possession/v5_run.py` (calibration, grid, robustness).
Logs: `possession/logs/v5/{run.log,calibration.json,grid.json}`. IDSSE TUNE matches only, CPU Viterbi, no TCN inference
(v4 cB seeds 10-14 LOMO folds with hook, reproduced: mean 0.6639 [0.6137, 0.7249]). Command: `python -m possession.v5_run`.

## Design
- Real label-free aggregates (human ran `tag_check` against v4 labels) give per code: n, alive frames, D share, agreement with v4.
  DEFENDING and defensive-transition codes duplicate the attacking intervals with the defender as team, so they are dropped.
  Ten codes used: build up, progression, attacking transition, long ball, ball in final third, creating chances, ball in the box,
  players in the box, set pieces, goals.
- Simulator (from DFL truth spells): interval length lognormal (sigma 0.4) around alive / n / (1 - D share) frames (build up 501,
  progression 472, transitions 388, long ball 330, ...); counts scale with half length; anchor = alive spell (drawn in proportion to
  its length) or the start of a dead run (set pieces, goals); start jitter 1.5 s; team = anchor spell's team. Noise knobs per code,
  one used at a time: `p_flip` (wrong team) or, when the spilling structure alone is already too noisy, `q_trunc` (clean interval,
  no jitter, cut at spell end). Calibrated by bisection (8 draws x 5 matches) so simulated agreement with v4 LOMO predictions
  matches the real table. This uses label-free statistics only.
- Fusion: inside an interval, e[:, team] += w_c and e[:, other] -= w_c (D column untouched) before the Viterbi;
  w_c = k * logit(agreement_c), one global k, grid {0, 0.5, 1, 2}. Mean over 3 simulation seeds.
- Rule fixed beforehand: adopt iff LOMO mean spellF1 >= v4 + 0.02 AND beats v4 on >= 4/5 matches.

## Calibration (simulated vs real agreement with v4)
| code | real | simulated | p_flip | q_trunc |
|---|---|---|---|---|
| build up | 0.943 | 0.943 | 0 | 0.97 |
| progression | 0.880 | 0.880 | 0 | 0.67 |
| attacking transition | 0.683 | 0.682 | 0.20 | 0 |
| long ball | 0.681 | 0.680 | 0.17 | 0 |
| ball in final third | 0.685 | 0.685 | 0.15 | 0 |
| creating chances | 0.664 | 0.664 | 0.20 | 0 |
| ball in the box | 0.630 | 0.631 | 0.29 | 0 |
| players in the box | 0.591 | 0.590 | 0.32 | 0 |
| set pieces | 0.606 | 0.606 | 0 | 0.32 |
| goals | 0.786 | 0.687 | 0 | 1.0 (cap) |

All within +-0.03 except goals (4 instances; max reachable 0.687 with this structure; low weight, kept as is).

## Result (LOMO spellF1, v4 per-fold predictions as reference)
| k | mean | [min, max] | wins vs v4 |
|---|---|---|---|
| 0 (= v4) | 0.6639 | [0.6137, 0.7249] | - |
| 0.5 | 0.6586 | [0.6115, 0.7085] | 1/5 |
| 1 | 0.6174 | [0.5708, 0.6602] | 0/5 |
| 2 | 0.5514 | [0.5204, 0.5776] | 0/5 |

Best k = 0.5: -0.005 vs v4, far from the +0.02 gate. **FAIL: v5 not adopted, no params_v5.json, no `--method v5`,
no `load_phases`, no `v5` branch in run_all.ps1.**

Robustness (simulated reliability 0.1 worse, k = 0.5): 0.6368 [0.5915, 0.6896], 0/5 wins (not a gate; worse than v4).

## Reading
- Tags simulated with calibrated noise do not help at any k: the real agreement values (0.59-0.69 for most codes) are too low
  to add information beyond v4, and long intervals spanning possession changes bias the Viterbi toward the wrong team.
- Only build up and progression (88 instances) are reliable by this measure; they carry w = 2.8 and 2.0 per unit k. A variant
  using only those two codes was not run (it would be chosen after seeing the results, so it would need the locked check).
- Caveats, unverified: agreement with v4 is not accuracy; the simulator's noise is independent of v4's errors, while real tag
  errors may correlate with the tracking-derived evidence (optimistic or pessimistic, unknown). A code subset or a different
  interval model was not explored, per the fixed rule.
