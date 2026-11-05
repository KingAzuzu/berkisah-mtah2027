# v4 confirmation (fresh seeds 10-14, LOMO IDSSE TUNE, matched2)

Purpose: confirm the post-hoc lead of tcn_v4.md ("no SkillCorner + boundary + hook 0.6"). Seeds 10-14 were never used before. Both configs: no SkillCorner, boundary loss (0.5), EMA 0.995, domain randomisation, mirror aug + TTA, hook 0.6, 1500 steps, 5 seeds, fused 50/50 with v2.1 evidence. Code: possession/v4_run.py (configs cA, cB), v4c_queue.sh, v4c_eval.py; numbers in logs/v4c_eval.json.

Rule fixed before running: pass iff mean spellF1 >= 0.661 and better than v3 (0.6411, 3-seed ensemble) on >= 4/5 matches; if both pass choose B unless A beats B by >= 0.01; if neither, keep v3.

| config | spellF1 mean [min,max] | wins vs v3 | macro | per match MX N1 OH OY PY | pass |
|---|---|---|---|---|---|
| v3 (ref) | 0.6411 [0.5973, 0.6813] | - | 0.9139 | see tcn.md | - |
| A: IDSSE + Metrica + mirror | 0.6664 [0.629, 0.712] | 4/5 | 0.9248 | .655 .629 .712 .640 .695 | yes |
| B: IDSSE only + mirror (licence-safe) | 0.6639 [0.614, 0.725] | 5/5 | 0.9229 | .652 .614 .725 .653 .675 | yes |

## Decision
Both pass. A - B = 0.0025 < 0.01, so B is chosen (no Metrica data, no licence question). The earlier post-hoc number (0.6672 with Metrica) replicates on fresh seeds (0.6664), and the no-Metrica variant is within noise of it. Gain over v3 for B is +0.0228; on one match below ~0.04 is noise, so the evidence is the 5/5 consistency and the mean, not any single match (B min 0.614 vs v3 min 0.597).

## Frozen v4 (config B)
Final 5 models trained on all 5 TUNE matches (canonical matched2 + 6 DR variants per half), seeds 10-14, 1500 steps, no Metrica, no SkillCorner: possession/models/tcn_v4_s{10..14}.pt, params_v4.json, `pipeline.v4_predictor()`, `bar_ath.py --method v4`, `run_all.ps1 v4`. Freeze code: possession/tcn_v4.py.
CPU determinism (repro_check --method v4, J03WMX half 0, IDSSE TUNE): identical codes SHA at 1 and 4 threads, 679130d5d6334c542fb6aa0b5b0c358e9efb907aea18cf6e9bbfb59949672907.
Not done: nothing run on the competition video; locked matches J03WQQ/J03WR9 not evaluated. Unverified: byte-identical rerun on the competition tracking and across machines (human runs `.\run_all.ps1 v4`); the frozen models are not scored on IDSSE (trained on all TUNE).
