# Integration: WS2 rules + WS4 dead-ball classifier + decoder, leave-one-match-out

IDSSE tuning matches only (`data.TUNE`, degraded halves). The locked matches were never loaded. Competition data never touched.
All numbers come from the commands below; raw output is in `possession/logs/integrate_*.json|txt`.
Match order in every table: MX N1 OH OY PY (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY). Truth spells: 511 500 402 456 421.

Files (new): `possession/pipeline.py`, `integrate_run.py`, `integrate_check.py`, `params_v2.json`, `models/dead_v2.joblib`,
`logs/integrate_*`, this file.

## Protocol (nested, all fitted parts inside folds)

- Team evidence = WS2 `rules.evidence` with fixed params R 1.0, BV 0.5, KW 20, W_HOLD 0.3, W_DIFF 0.6 (WS2's final choices; not re-tuned).
  The dead damping on the team columns and the D column are recomputed from the D source being tested.
- D column = `alpha * logit(p_dead)` from a WS4 HistGradientBoosting model (same hyper-parameters, every 5th frame, `random_state=0`).
  Variants: **players** (160 of 241 features, all ball-derived ones removed: prefixes `b_`, `d_nearest`, `n_near`, `d_gap`, `d_min_team`) and **ball** (all 241).
- For held-out match m: cell (ALPHA, TEMP, SWITCH_TEAM, SWITCH_DEAD) chosen on the other 4 matches using p_dead from models trained on the 3 matches that are neither m nor that match; score on m uses the model trained on the other 4.
- Scoring only through `possession/score.py`. Per-match noise reference ~0.04.

## Step 0: rerun WS2 final

`python -m possession.rules_run --stage 6 --name rules_final` (26 s) -> LOMO spellF1 0.3985 [0.3764, 0.4366], macroF1 0.7790,
spells pred/gt [[320,511],[347,500],[317,402],[344,456],[334,421]]. Identical to the WS2 write-up; `rules_final.json` is byte-identical to the previous file (`cmp`). No difference.
Reference safety net re-scored on the same code path (`integrate_run --cond seed0 --ref`): 0.3017 [0.2804, 0.3259], macroF1 0.7573.

## Steps 1 and 2: seed 0 results

Commands: `python -m possession.integrate_run --cond seed0 --runs s1_rules s1_players s1_ball s2_rules s2_players s2_ball`.
Grids: s1 (Viterbi) ALPHA {0.25,0.5,1,1.5,2} x SWITCH_TEAM {5,10,20,40} x SWITCH_DEAD {2.5,5,10,20,40,80}; s2 (posterior argmax of `mbr.forward_backward`, no sampling) ALPHA {0.25,0.5,1} x TEMP {1,2,3} x SWITCH_TEAM {5,10,20,40} x SWITCH_DEAD {5,10,20}.
"rules" rows use the safety-net D column (no WS4), with SWITCH_DEAD/TEMP tuned the same nested way (rules only: no ALPHA).

| run | LOMO spellF1 | min | max | spellF1 per match | macroF1 | macro per match | spells pred (gt 511 500 402 456 421) |
|---|---|---|---|---|---|---|---|
| reference safety net | 0.3017 | 0.2804 | 0.3259 | .3002 .2804 .3203 .2814 .3259 | 0.7573 | .778 .740 .763 .733 .773 | 359 366 351 357 351 |
| WS2 rules (step 0) | 0.3985 | 0.3764 | 0.4366 | .4173 .3764 .3811 .3810 .4366 | 0.7790 | (WS2 table) | 320 347 317 344 334 |
| rules, SWITCH_DEAD retuned (s1_rules) | 0.4033 | 0.3748 | 0.4596 | .4170 .3831 .3820 .3748 .4596 | 0.7828 | .817 .756 .782 .756 .803 | 311 329 286 323 310 |
| **Step 1 players** (s1_players) | **0.4367** | 0.3822 | 0.4970 | .4476 .4105 .4464 .3822 .4970 | 0.8232 | .850 .787 .832 .793 .854 | 304 340 296 329 313 |
| Step 1 ball (s1_ball) | 0.4986 | 0.4497 | 0.5624 | .5005 .4755 .5048 .4497 .5624 | 0.8738 | .895 .851 .873 .855 .895 | 329 342 304 332 303 |
| Step 2 rules + argmax (s2_rules) | 0.4024 | 0.3727 | 0.4541 | .4216 .3788 .3848 .3727 .4541 | 0.7836 | .817 .756 .784 .758 .803 | 305 330 307 334 326 |
| Step 2 players + argmax (s2_players) | 0.4302 | 0.3831 | 0.4875 | .4411 .4104 .4288 .3831 .4875 | 0.8235 | .850 .788 .832 .794 .854 | 317 349 321 347 333 |
| Step 2 ball + argmax (s2_ball) | 0.4986 | 0.4518 | 0.5561 | .4996 .4796 .5057 .4518 .5561 | 0.8736 | .895 .851 .873 .857 .892 | 344 364 317 339 327 |

Findings:
- Step 1 (WS4 D column on WS2 evidence), players-only: +0.033 over rules alone (0.4033 -> 0.4367), +0.135 over the reference. Four of five matches improve by 0.03 to 0.06 (MX +.030, N1 +.027, OH +.064, PY +.037, OY +.007); only OH clears the 0.04 single-match note, so the mean gain is consistent but modest per match.
- Ball features add +0.062 more (0.4986), but that is the variant at risk (see step 3).
- Fold-chosen cells (players): ALPHA 0.25-0.5, SWITCH_TEAM 10-20, SWITCH_DEAD 5. ALPHA is interior of its grid; SWITCH_DEAD 5 is interior (2.5 was tried).
- Step 2 is negative: posterior argmax does not stack on the rules evidence. Rules: 0.4033 -> 0.4024; players: 0.4367 -> 0.4302; ball: 0.4986 -> 0.4986. WS3's +0.02 does not survive once the evidence is better. Decoder kept as Viterbi.
- Predicted spells stay below truth (about 300-340 vs 402-511), so fragmentation is not the limit.

## Step 3: robustness (best combination = Step 1 players, Viterbi; ball and rules rows for comparison)

Commands: `integrate_run --cond seedN --ref --runs s1_rules s1_players s1_ball` for N = 1, 2 (halves from `run_idsse.load(mid, seed=N)`, cached under `idsse-tests/data/cache/`), and `integrate_run --cond noise0.3|noise1.0 --ref --runs s1_rules s1_players s1_ball --mismatch s1_players s1_ball`.
Noise = Gaussian sigma added to observed ball x, y (seeded). "matched" = features, models and cells all fitted on that condition inside folds. "mismatch" = models trained on clean seed-0 data (fold models) and cells from the clean run, scored on noisy halves (the deployment worry).

| condition / method | LOMO spellF1 | min | max | per match | macroF1 | per match macro |
|---|---|---|---|---|---|---|
| seed1 reference | 0.3012 | 0.2731 | 0.3280 | .2883 .2898 .3266 .2731 .3280 | 0.7570 | .781 .730 .764 .739 .770 |
| seed1 rules | 0.4075 | 0.3963 | 0.4234 | .4234 .3981 .4021 .3963 .4177 | 0.7837 | .815 .762 .784 .770 .787 |
| seed1 players | 0.4365 | 0.4172 | 0.4586 | .4390 .4172 .4444 .4230 .4586 | 0.8219 | .850 .787 .825 .813 .834 |
| seed1 ball | 0.5116 | 0.4927 | 0.5273 | .5069 .5049 .5273 .4927 .5260 | 0.8758 | .895 .863 .871 .870 .879 |
| seed2 reference | 0.3051 | 0.2801 | 0.3393 | .3123 .2897 .3042 .2801 .3393 | 0.7610 | .786 .730 .766 .748 .776 |
| seed2 rules | 0.4046 | 0.3754 | 0.4435 | .3937 .3956 .4151 .3754 .4435 | 0.7821 | .806 .763 .791 .755 .795 |
| seed2 players | 0.4239 | 0.3980 | 0.4633 | .4097 .4209 .4278 .3980 .4633 | 0.8207 | .839 .790 .835 .804 .835 |
| seed2 ball | 0.4918 | 0.4481 | 0.5365 | .4736 .4979 .5029 .4481 .5365 | 0.8729 | .886 .859 .876 .863 .880 |
| noise 0.3 reference | 0.3060 | 0.2768 | 0.3338 | .3055 .2894 .3247 .2768 .3338 | 0.7634 | .785 .747 .769 .735 .781 |
| noise 0.3 rules | 0.4044 | 0.3641 | 0.4385 | .4096 .3933 .4167 .3641 .4385 | 0.7884 | .813 .769 .791 .764 .805 |
| noise 0.3 players, matched | 0.4270 | 0.3812 | 0.4724 | .4301 .4086 .4428 .3812 .4724 | 0.8210 | .841 .794 .829 .795 .846 |
| noise 0.3 players, mismatch | 0.4283 | 0.3857 | 0.4657 | .4336 .4086 .4480 .3857 .4657 | 0.8216 | .841 .794 .832 .794 .847 |
| noise 0.3 ball, matched | 0.4717 | 0.4282 | 0.5040 | .4674 .4677 .4909 .4282 .5040 | 0.8594 | .874 .846 .859 .842 .876 |
| noise 0.3 ball, mismatch | 0.2919 | 0.2655 | 0.3211 | .3211 .2655 .2966 .2660 .3104 | 0.4878 | .521 .463 .470 .468 .517 |
| noise 1.0 reference | 0.3069 | 0.2757 | 0.3362 | .3111 .2884 .3232 .2757 .3362 | 0.7716 | .794 .745 .780 .747 .791 |
| noise 1.0 rules | 0.3503 | 0.3201 | 0.3834 | .3627 .3201 .3404 .3447 .3834 | 0.7786 | .804 .758 .773 .755 .803 |
| noise 1.0 players, matched | 0.3633 | 0.3214 | 0.4189 | .3794 .3214 .3529 .3440 .4189 | 0.8012 | .827 .775 .798 .781 .825 |
| noise 1.0 players, mismatch | 0.3690 | 0.3371 | 0.4103 | .3856 .3371 .3588 .3534 .4103 | 0.7992 | .822 .770 .801 .779 .824 |
| noise 1.0 ball, matched | 0.4128 | 0.3951 | 0.4603 | .4096 .3951 .4011 .3979 .4603 | 0.8375 | .858 .826 .825 .823 .855 |
| noise 1.0 ball, mismatch | 0.2614 | 0.2382 | 0.2842 | .2842 .2483 .2382 .2587 .2776 | 0.4770 | .510 .458 .454 .456 .507 |

Spell counts for these runs are in `logs/integrate_<run>_<cond>.json` (`held_out[m].n_pred`; truth counts as above).

Findings:
- Players-only gain over the reference holds on every seed and noise level: seeds 0/1/2 0.4367 / 0.4365 / 0.4239 vs 0.3017 / 0.3012 / 0.3051; at sigma 1.0, 0.3633 (matched) and 0.3690 (mismatch) vs 0.3069, and it stays above rules-only (0.3503) at both noise levels. Seed spread of the mean is 0.013 (inside the noise note).
- The ball-feature model is not usable if the real ball track is noisy and the model was trained on clean tracking: mismatch at sigma 0.3 m gives 0.2919 (below the reference 0.3060) and macroF1 0.49; at 1.0 m, 0.2614. Retrained on noisy data it recovers (0.4717, 0.4128) but is then still dropping fast. Ball speed in IDSSE is clean, so the 0.4986 headline is not trustworthy for a video ball. Players-only is the primary variant.
- Under sigma 1.0 noise all methods fall (players 0.4367 -> 0.3633 matched); the rules evidence is the part that suffers (rules 0.4033 -> 0.3503), since it reads ball distances and speed changes.

## Step 4: freeze

Command: `python -m possession.integrate_run --freeze s1_players`, then `python -m possession.integrate_check`.
- Method: WS2 rules team evidence (fixed params above) + WS4 players-only D column (alpha * logit p_dead) + `safety.viterbi`. Posterior argmax and the HSMM are not used.
- `possession/params_v2.json`: R 1.0, BV 0.5, KW 20, W_HOLD 0.3, W_DIFF 0.6, ALPHA 0.5, TEMP 1.0, SWITCH_TEAM 20, SWITCH_DEAD 5, DECODER viterbi.
  Chosen as the **best fixed cell on the inner folds**: mean spellF1 pooled over the 20 (m, k) inner folds (models that saw neither match), not the mean of the fold-chosen values. Its outer-fold mean is 0.4377 (context only; the honest figure is the LOMO 0.4367).
- `possession/models/dead_v2.joblib`: players-only HGB trained on all 5 tuning matches (160 features, every 5th frame, sklearn/numpy/joblib versions stored inside).
- `possession/pipeline.py`: `predict(half, params, models)`, `load_params()`, `load_models()`.
- Checks (`logs/integrate_check.txt`): `pipeline.predict` with the J03WMX fold model and its fold cell gives spellF1 0.447607 and macroF1 0.849958, equal to the logged LOMO values; two runs of the frozen model on one half give identical codes.

## Summary vs reference (seed 0, LOMO, 5 tuning matches)

| step | spellF1 mean [min, max] | macroF1 | vs reference |
|---|---|---|---|
| reference safety net | 0.3017 [0.2804, 0.3259] | 0.7573 | |
| 0 WS2 rules (rerun) | 0.3985 [0.3764, 0.4366] | 0.7790 | +0.097 |
| rules, SWITCH_DEAD retuned | 0.4033 [0.3748, 0.4596] | 0.7828 | +0.102 |
| 1 + WS4 D column, players-only (frozen) | 0.4367 [0.3822, 0.4970] | 0.8232 | +0.135 |
| 1 + WS4 D column, with ball | 0.4986 [0.4497, 0.5624] | 0.8738 | +0.197 |
| 2 players + posterior argmax | 0.4302 [0.3831, 0.4875] | 0.8235 | +0.129 (below step 1) |
| 2 ball + posterior argmax | 0.4986 [0.4518, 0.5561] | 0.8736 | same as step 1 ball |

## Unverified

- Everything is IDSSE degraded by our own `degrade.py`; its ball-gap and slot model are assumptions. Real ATD tracking may differ (identity swaps, wrong team, duplicated players, ball position noise were not modelled except the Gaussian test).
- The Gaussian noise test is only ball position noise, applied to IDSSE's clean ball; players-only features still use the ball-independent player tracks, and the rules evidence still reads the ball.
- WS2 rule parameters were fixed from WS2's selection, which looked at all 5 matches, so the integrated LOMO is mildly optimistic. Grid ranges were widened after a first look (alpha 0.25 added, SWITCH_DEAD 2.5 and 80, SWITCH_TEAM 40); step-1 cells chosen interior except none at the s2 edges TEMP 1 and SWITCH_DEAD 5 (s2 TEMP below 1 was not tried).
- Step 2 negative result is for this grid; sampling-based MBR was not rerun (WS3 showed it equals argmax).
- Seeds 1 and 2 retrain and re-tune inside each condition; the frozen seed-0 model/params were not scored on seeds 1 and 2.
- Players-only AUC on real video-derived tracking is unknown: IDSSE pitch size, coordinate frame and a unknown attack direction handling are as in WS4.
- Frozen model is scored only in-sample on tuning matches through the pipeline check (one fold check, not the frozen model). HGB determinism across thread counts and a byte-for-byte rerun on another machine are untested.
- Locked matches not evaluated (coordinator's gate). `bar_ath.py` not updated.
