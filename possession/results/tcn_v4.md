# v4 candidate: TCN + extra data + domain randomisation + boundary head + EMA (LOMO IDSSE TUNE, matched2)

## Data and licences
- SkillCorner open data (github.com/SkillCorner/opendata, MIT licence, (c) SkillCorner; credit SkillCorner): 20 A-League matches, broadcast tracking 10 Hz, partial view. Raw ~1.8 GB in idsse-tests/data/external/skillcorner (gitignored, git-lfs). Loader possession/scdata.py.
- Metrica sample games 1-3 as in results/tcn_extdata.md (no formal licence, acknowledge Metrica Sports). IDSSE CC BY 4.0.
- SkillCorner labels: H/A = team of the last non-null `possession.group`, held through nulls (DFL-like), runs < 2 samples (0.2 s) dropped. D is NOT derivable: possession is null 36-54% of frames (flight and dead alike), ball-out is not flagged (ball detected outside pitch 1-3.5% of seen frames only). So D is masked: SkillCorner windows get the team loss only (2-way softmax over H/A) and the boundary loss on H/A changes.
- Resampled 10 -> 25 fps (interpolate between two observed samples, else NaN) then the same 5 Hz features. No extra matched2 degrade: broadcast tracking is already partial (ball seen 0.52-0.65, mean 0.59; 7.5-10.9 players per frame).
- Sanity vs IDSSE: SC mean H/A spell 18-26 s (mean ~23 s) vs IDSSE 5.5-7.7 s (6.8) and Metrica ~10 s; SC has no D class. SC labels are far smoother (the group flag fires only on control, not on every pass reception), a systematic mismatch with DFL truth.

## Recipe (all folds: held-out IDSSE match never in training)
Training sources and fixed sampling mass: IDSSE other 4 (0.60), Metrica (0.15), SkillCorner (0.25) (renormalised when one is removed). Domain randomisation: 6 extra degrade variants per IDSSE half with matched2 constants drawn uniformly (ball seen 0.48-0.70, missing-run median 2.5-6 s, tracklet median 20-60 s, gap 2-6 s, jitter 0.5-4 cm, P_DEAD 0.45-0.80; possession/v4_prep.py), plus the canonical variant; a variant is drawn per window (pool of 7, not fresh per epoch). Mirror augmentation (p 0.5) and mirror TTA. Loss: CE (team-only for SC) + 0.5 x BCE boundary head (Gaussian-smoothed H/A/D change target, sigma 2 steps = 0.4 s) + EMA 0.995. 1500 steps, bs 64, crop 300, same net. Ensemble mean log-probs, 50/50 with v2.1 evidence, same Viterbi costs. "+hook": Viterbi switch costs scaled by (1 - 0.6 x ensemble boundary prob), constant fixed beforehand, not tuned. Hard-negative upweighting: not done (optional, skipped).
Code: possession/v4_lib.py, v4_run.py, v4_queue.sh, v4_eval.py, v4_status.py; logs possession/logs/v4_eval.json.

## Results (spellF1 mean [min,max] over 5 matches; wins = matches better than v3 ensemble; per match MX N1 OH OY PY)
| config | spellF1 | wins vs v3 | macro | per match |
|---|---|---|---|---|
| v3 (3 seeds, ref) | 0.6411 [0.5973, 0.6813] | - | 0.9139 | see tcn.md |
| full x3 | 0.6411 [0.609, 0.693] | 4/5 | 0.9208 | .625 .609 .693 .618 .661 |
| full x3 +hook | 0.6544 [0.610, 0.698] | 4/5 | 0.9216 | .646 .610 .698 .642 .676 |
| full x5 | 0.6391 [0.601, 0.692] | 3/5 | 0.9203 | .618 .601 .692 .615 .669 |
| full x5 +hook | 0.6536 [0.612, 0.699] | 4/5 | 0.9216 | .640 .612 .699 .641 .676 |
| no Metrica x3 | 0.6376 [0.583, 0.683] | 2/5 | 0.9204 | .618 .583 .683 .637 .667 |
| no Metrica x3 +hook | 0.6519 [0.601, 0.699] | 4/5 | 0.9222 | .636 .600 .699 .637 .687 |
| no SkillCorner x3 | 0.6598 [0.617, 0.708] | 5/5 | 0.9245 | .641 .617 .708 .648 .684 |
| no SkillCorner x3 +hook | 0.6698 [0.637, 0.715] | 4/5 | 0.9259 | .658 .637 .714 .644 .696 |
| no SkillCorner x5 | 0.6578 [0.618, 0.701] | 5/5 | 0.9238 | .641 .618 .701 .646 .685 |
| no SkillCorner x5 +hook | 0.6672 [0.637, 0.709] | 5/5 | 0.9251 | .644 .637 .709 .652 .694 |
| no boundary loss x3 | 0.6322 [0.604, 0.670] | 1/5 | 0.9195 | .614 .604 .670 .620 .653 |

## Decision (fixed rule: mean >= 0.661 and better than v3 on >= 4/5)
The declared candidate (full, 5 seeds, with hook) gets 0.6536: NO WINNER. The licence-safe variant (no Metrica) is 0.6519 at 3 seeds, also below the bar. Findings: SkillCorner data hurts (about -0.013 to -0.016 with hook; its labels are ~3x smoother than DFL truth and lack D), the boundary loss helps (+0.008 vs no-boundary at 3 seeds, +0.021 with the hook), the hook adds +0.009 to +0.013 on top of full/nomet.
Post-hoc lead, not frozen: "no SkillCorner" (IDSSE + Metrica, DR, mirror, boundary, EMA, 5 seeds, hook) reaches 0.6672 [0.637, 0.709], 5/5 wins, numerically above the bar. It was picked after seeing the ablations (1 of 4 configs x 2 decoders), so it is not a clean winner; a no-SkillCorner no-Metrica run was not done, so its licence-safe status is untested. Freezing it needs the coordinator's call. Not wired into bar_ath, requirements or run_all.
Unverified: v3 reference uses the stored 3-seed ensemble; effects below ~0.01 are within seed noise (SD of seed-match-means ~0.003-0.006).
