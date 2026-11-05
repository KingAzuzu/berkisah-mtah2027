# v7 ceiling diagnostic (nothing frozen, not wired in)

v6 recipe (cB, SHORT_W 3, 1500 steps, no DR when clean), LOMO on TUNE, seeds 20+21 averaged, TCN-only decode (fusion 1.0, Viterbi 20/5/hook 0.6). Organiser scorer. IDSSE TUNE only.
Clean = undegraded IDSSE train and test (cache tcn_feats_clean.pkl). Degraded = existing v6 e2_s20/s21 log-probs, same decoder.
Code: possession/v7_ceiling.py; scores: possession/logs/v7/ceil_scores.json.

| match | clean spellF1 | clean macroF1 | degr spellF1 | degr macroF1 | spells pred/true clean | pred/true degr |
|---|---|---|---|---|---|---|
| J03WMX | .6941 | .9463 | .6568 | .9361 | 437/511 | 397/511 |
| J03WN1 | .6896 | .9306 | .6212 | .8954 | 424/500 | 417/500 |
| J03WOH | .7530 | .9476 | .7149 | .9360 | 381/402 | 364/402 |
| J03WOY | .7528 | .9406 | .7128 | .9280 | 442/456 | 402/456 |
| J03WPY | .7312 | .9481 | .6786 | .9317 | 387/421 | 363/421 |
| mean | .7241 | .9426 | .6769 | .9254 | | |

Caveats: 2 seeds only; TCN-only (degraded 0.6769 is below the 0.6860 fused 5-seed v6); the DR features were used in degraded training but not clean.
Clean ceiling +0.047 spellF1 over degraded: even undegraded, the recipe is ~0.72, far from 0.85.
