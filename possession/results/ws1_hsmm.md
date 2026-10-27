# WS1: explicit-duration HSMM decoder

Result: **no gain over the reference.** HSMM LOMO spellF1 0.2877 [0.2578, 0.3227] vs reference 0.3017
[0.2804, 0.3259]; macroF1 0.7589 vs 0.7573. Every number below comes from the commands listed.

## Built
- `possession/hsmm.py`: `fit(truths)` (per-state smoothed duration pmf in 5-frame blocks, geometric tail past the
  pooled 99.5th percentile = 404 blocks, segment transition log-probs without self-transitions, all from training
  truth) and `decode(e, model, wd, wt)` (semi-Markov Viterbi at 5 fps, vectorised over segment length, first
  segment censored, then each boundary refined at 25 fps within +/-5 frames on evidence alone).
- `possession/hsmm_experiment.py`: own fold loop with nested selection. For held-out m, each grid cell is scored
  on every other match k using a model fitted on the 3 matches that are neither m nor k; the best cell is then
  applied to m with a model fitted on the other 4. Held-out truth never enters fitting or selection.
- `possession/hsmm_report.py`: histograms, runtime, determinism.
- Emissions: `safety.emissions(safety.features(half), {safety.PARAMS + params.json})`. Evidence scale is not a separate
  parameter, since only the ratios evidence : wd : wt change the argmax, so the "temperature" is folded into
  wd/wt. Grid: wd {1,2,3,4,6,8,12} x wt {0,1,2}.

## Commands (from D:\barca)
    D:\barca\idsse-tests\.venv\Scripts\python -m possession.hsmm_experiment   # writes logs/hsmm.json (111 s, 4 workers)
    D:\barca\idsse-tests\.venv\Scripts\python -m possession.hsmm_experiment --time
    D:\barca\idsse-tests\.venv\Scripts\python -m possession.hsmm_report       # writes logs/hsmm_report.json
(An earlier grid, wd up to 2, put its best cell at the edge, so the grid was extended once; first run gave 0.2813.
The final grid's best cells are interior, wd 3-4.)

## LOMO per match (params chosen in nested folds)
| match | HSMM spellF1 | ref spellF1 | HSMM macroF1 | ref macroF1 | spells pred/gt HSMM | ref pred/gt | chosen wd,wt |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.2887 | 0.3002 | 0.7784 | 0.7784 | 301/511 | 359/511 | 4,2 |
| J03WN1 | 0.2662 | 0.2804 | 0.7325 | 0.7402 | 364/500 | 366/500 | 3,2 |
| J03WOH | 0.3031 | 0.3203 | 0.7665 | 0.7628 | 337/402 | 351/402 | 3,2 |
| J03WOY | 0.2578 | 0.2814 | 0.7391 | 0.7326 | 298/456 | 357/456 | 4,1 |
| J03WPY | 0.3227 | 0.3259 | 0.7781 | 0.7728 | 350/421 | 351/421 | 3,2 |
| mean | 0.2877 | 0.3017 | 0.7589 | 0.7573 | | | |

HSMM is lower on all 5 matches, by 0.003 to 0.024 (mean -0.014): within the ~0.04 single-match noise, but
consistently signed. Fixed-cell outer-fold means (not an honest selection, for shape only): wd=3,wt=2 0.2939;
wd=4 ~0.290; wd=6 0.284; wd=1 0.264; wd=12 0.250. The HSMM never reaches the reference's spell count: stronger
duration weight cuts flicker but also loses short possessions (predicted spells fall from 2808 at wd=1 to 1739 at
wd=3 over the 5 outer folds, against 2290 true spells).

## Duration histogram (aggregate over the 5 tuning matches, runs per state, frames; reference = safety Viterbi)
Bins: <1s, 1-2s, 2-5s, 5-10s, 10-20s, 20-40s, >40s.
| state | source | counts | total |
|---|---|---|---|
| H | truth | 162 220 335 201 129 84 21 | 1152 |
| H | reference | 45 85 201 263 199 103 21 | 917 |
| H | HSMM | 37 157 165 186 168 106 22 | 841 |
| A | truth | 144 198 378 213 132 59 14 | 1138 |
| A | reference | 118 109 200 198 163 63 16 | 867 |
| A | HSMM | 28 132 207 174 172 75 21 | 809 |
| D | truth | 3 4 42 77 140 165 91 | 522 |
| D | reference | 0 3 57 95 120 133 90 | 498 |
| D | HSMM | 1 8 79 100 126 138 87 | 539 |

The HSMM matches the truth shape better in the 1-2 s bin (157/132 vs 85/109 for the reference) but under-produces
2-5 s team runs (165/207 vs 335/378 truth) and over-produces 10-20 s ones. Matching the histogram does not help
spellF1: the evidence rarely pins down which short possessions exist, so a prior that permits them adds false ones
as often as true ones.

## Runtime / determinism
Decode of a full half (cell wd=3, wt=2): 70,708 frames 0.30 s, 75,259 frames 0.32 s, CPU single thread. Two runs
identical (checked). Far below the 60 s limit.

## Unverified
- The comparison reference histogram uses `safety.viterbi` on params.json directly (fixed params), not in folds.
- The "predicted spells" totals in the cell table are possession_metrics spell counts summed over matches; the true
  total was not printed by this experiment (per-match gt counts are in the table above, sum 2290).
- No test of other duration families (log-normal mixture), smoothing width, block size, or a per-segment cost.
- Only one degrade seed (0); results on other seeds are unknown. No locked matches were touched.
