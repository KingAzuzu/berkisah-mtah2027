# v10 B: hedge, MBR on v8, 10-seed v8 (IDSSE TUNE LOMO, spellF1; per-match order MX N1 OH OY PY)
Scripts: v10_hedge.py, v10_mbr.py, v10_seeds_eval.py. Scores: logs/v10/{hedge,mbr,seeds}.json. Merged truth = merge(truth, 2 s).
Decoder everywhere: 0.5 TCN logp + 0.5 eR, viterbi_scaled(20, 5, 1-0.6b). Nothing wired in.

## 1. Hedge (decision aid, no gate)
| candidate | vs DFL | vs merged | regret DFL | regret merged | max regret |
|---|---|---|---|---|---|
| v6 | 0.6860 | 0.7099 | 0.0000 | 0.0465 | 0.0465 |
| v8 | 0.6517 | 0.7563 | 0.0343 | 0.0000 | 0.0343 |
| v6+merge K1 | 0.6854 | 0.7140 | 0.0006 | 0.0424 | 0.0424 |
| v6+merge K2 | 0.6548 | 0.7310 | 0.0312 | 0.0253 | 0.0312 |
| avg(v6,v8) logp | 0.6686 | 0.7449 | 0.0174 | 0.0114 | 0.0174 |
| avg + merge K1 | 0.6686 | 0.7449 | 0.0174 | 0.0114 | 0.0174 |
Minimax regret: log-prob average of v6 and v8 nets (merge K1 is a no-op on it, identical numbers). It costs 0.017 vs DFL-best and 0.011 vs merged-best.
Per-match avg: vs DFL .655 .615 .718 .653 .702; vs merged .752 .716 .770 .721 .766. Regret differences < 0.04 are within one-match noise.

## 2. MBR on v8 (gate: mean >= 0.7663 AND >= 4/5 wins vs v8)
Settings stated beforehand: A temperature 1, B temperature 2 (S=30 FFBS samples, candidates = v8 Viterbi, posterior argmax, samples; costs scaled per frame by 1-0.6b).
Fixed-setting means vs merged: A 0.7520, B 0.7545 (v8 0.7563). Nested LOMO (setting picked on the other 4 matches; always B):
0.771 0.723 0.774 0.735 0.770, mean 0.7545, wins vs v8 3/5. **FAIL.** MBR picked Viterbi/argmax (index 1) in most halves; no gain.
Vs DFL: A 0.6497, B 0.6439 (v8 0.6517).

## 3. 10-seed v8 (gate: mean >= 0.7613 AND >= 4/5 wins vs 5-seed)
Seeds 25-29 trained sequentially on GPU, deterministic recipe.
| | vs merged | per match | vs DFL |
|---|---|---|---|
| 5 seeds (20-24) | 0.7563 | .763 .719 .786 .734 .779 | 0.6517 |
| 10 seeds (20-29) | 0.7495 | .758 .711 .771 .737 .771 | 0.6476 |
Wins 1/5. **FAIL.** More seeds slightly lower (-0.007, within noise); seed averaging is saturated.

## Caveats
Single IDSSE tune set, 5 matches; the true competition convention is unknown. The hedge is a decision aid chosen after seeing both truths. MBR sampled posterior uses the HMM Gibbs model with the fixed costs, which may be miscalibrated for fused evidence.
