# Candidate B: TCN (LOMO on TUNE, matched2 degrade, IDSSE only)
Net: 7 dilated residual blocks, 64 ch, 175,747 params, RF +-7.4 s at 5 Hz; 26 per-frame pooled features; 1500 steps, bs 64 x 60 s windows,
team-swap augmentation, seed 0, bf16 on GPU; inference on CPU fp32, verified bit-identical twice. Viterbi with params_v2_1 switch costs, TEMP 1.
One configuration run, nothing tuned on held-out matches (no early-stop split; fixed steps).

| | spellF1 mean [min,max] | macroF1 mean | wins vs R |
|---|---|---|---|
| R (v2.1) | 0.5562 [0.5413, 0.5864] | - | - |
| TCN alone | 0.6169 [0.5648, 0.6505] | 0.9028 | 5/5 |
| TCN + v2.1 fusion 50/50 | 0.6267 [0.5891, 0.6525] | 0.9089 | 5/5 |

Per match (R / alone / fusion): MX .558/.605/.608; N1 .541/.565/.589; OH .586/.651/.653; OY .547/.627/.636; PY .548/.637/.649.
Rule (>= R+0.02 and >= 4/5): both are winners. Not wired into bar_ath/requirements (torch needs human approval).
Code: possession/tcn_lib.py, tcn_run.py; log possession/logs/tcn_lomo_1500.json. No weights saved (per-fold models only).

## Seed check and ensemble (LOMO on TUNE, matched2, fusion 50/50 + Viterbi, 1500 steps)
| | spellF1 mean [min,max] | macroF1 |
|---|---|---|
| seed 0 | 0.6267 [0.5891, 0.6525] | 0.9089 |
| seed 1 | 0.6209 [0.5692, 0.6586] | 0.9062 |
| seed 2 | 0.6250 [0.5966, 0.6495] | 0.9099 |
| 3-seed ensemble (mean log-probs) | 0.6411 [0.5973, 0.6813] | 0.9139 |
Ensemble > mean single seed (0.6242) by 0.017, so v3 uses the ensemble. Final models: all 5 TUNE matches, seeds 0-2, `models/tcn_v3_s{0,1,2}.pt` (719 KB each), `params_v3.json`. Code `tcn_v3.py`, `tcn_v3_seeds.py`; log `logs/tcn_v3_seeds.json`.
