# WS3: minimum Bayes risk decoding

Files: `possession/mbr.py` (module), `possession/mbr_experiment.py` (LOMO entry point),
`possession/mbr_probe.py` (timing / sanity probe). Logs: `possession/logs/mbr_reference.json`,
`mbr_stage1.json`, `mbr_stage2.json`, `mbr_ablation.json`. IDSSE tuning matches only (degraded, `data.TUNE`);
locked matches and competition data never loaded.

## What was built
- Same 3-state HMM as `safety.viterbi`: path score = sum e_t/TEMP minus switch costs. Log-space forward and
  backward, marginals, forward-filtering backward-sampling (S paths, `np.random.default_rng(0)`).
- Candidates: `safety.viterbi` at multiples of (ST, SD), posterior argmax, every sample (leave-one-out).
  Pick the one with highest mean spellF1 vs the S samples. Spell extraction and greedy matching are
  `possession_metrics.possession_spells` / `match_spells` with one valid run per half; only the pairwise
  overlap enumeration is a sweep instead of the O(n*m) loop. `mbr_probe` asserts it equals
  `pm.spell_scores` on 4 paths (passed).
- Keep-threshold: spells whose mean posterior marginal is below KEEP are set to D.
- Sanity (probe, first match first half, T=70,708): marginals sum to 1 (1 +/- 2e-9), sampling is
  deterministic across two calls, samples have ~195 spells per half (Viterbi path agrees with a sample on 94% of frames).

## Commands (from D:\barca, `idsse-tests\.venv\Scripts\python -m ...`)
- `possession.mbr_probe 20 1.0`
- reference: `possession.lomo`-style run, logged as `mbr_reference`
- `possession.mbr_experiment --stage 1` (TEMP 0.5/1/2/3, S 50, ST 10, SD 20, MULT one/wide/low, METHOD viterbi/argmax/mbr, KEEP 0)
- `possession.mbr_experiment --stage 2` (TEMP 0.5..3, ST 5/10, SD 10/20, MULT wide, METHOD mbr, KEEP 0/.2/.35/.5)
- `possession.mbr_experiment --name mbr_ablation --grid <json>` (TEMP 3, ST 5, SD 20, S 50/200, all METHOD, all KEEP)

## LOMO results (parameters chosen on the other 4 matches; per-match order MX, N1, OH, OY, PY)

| config | LOMO spellF1 [min, max] | macroF1 | spellF1 per match | spells pred per match (gt 511/500/402/456/421) |
|---|---|---|---|---|
| reference safety net | 0.3017 [0.2804, 0.3259] | 0.7573 | 0.3002 0.2804 0.3203 0.2814 0.3259 | 359 366 351 357 351 |
| stage 1 (KEEP 0, selects posterior argmax) | 0.3121 [0.2783, 0.3404] | 0.7624 | 0.3216 0.2881 0.3322 0.2783 0.3404 | 417 408 395 290 404 |
| stage 2 (MBR + keep, as selected by LOMO) | 0.3225 [0.2932, 0.3467] | 0.7651 | 0.3254 0.2932 0.3302 0.3167 0.3467 | 422 403 383 423 405 |
| ablation grid (best cell: TEMP 3, ST 5, SD 20, KEEP 0.5) | 0.3235 [0.2932, 0.3467] | 0.7668 | 0.3304 0.2932 0.3302 0.3167 0.3467 | 409 403 383 423 405 |

Stage 2 per-match macroF1: 0.7854 0.7502 0.7629 0.7434 0.7834. Reference per-match macroF1: 0.7784 0.7402 0.7628 0.7326 0.7728.

Fixed-cell comparison (ablation, TEMP 3, ST 5, SD 20, mean over 5 matches, no LOMO selection):
Viterbi 0.2979, posterior argmax 0.3214, MBR 0.3214 (identical to argmax at every KEEP), argmax/MBR with KEEP 0.35: 0.3215,
KEEP 0.5: 0.3235 (Viterbi at KEEP 0.5: 0.2974). S 50 and S 200 give identical numbers.

## Findings
- LOMO gain over the reference: +0.021 (stage 2) to +0.022 (ablation cell), with every one of the 5 matches at or above
  the reference (per-match gain 0.002 to 0.035). The gain over Viterbi is almost all the posterior argmax; the
  largest single move is match OY (0.2814 -> 0.3167), above the ~0.04 noise note only partly, so treat the mean gain as
  modest and the per-match spread as comparable to it.
- The sampling-based MBR step never chose anything other than the posterior argmax in the cells I compared (MBR spellF1
  equals argmax spellF1 in every grid cell I checked in stage 1 and the ablation). So WS3's useful ingredient is the
  marginal-posterior decode, not sample-based selection. I did not inspect `info["picked"]` across all halves
  (only the probe: argmax picked, util 0.905 vs 0.767 for Viterbi).
- Keep-threshold: no effect up to 0.35, +0.002 at 0.5 (below noise); not shown to help.
- Higher TEMP (flatter emissions) helps, 3 was the top of the grid; the optimum may lie beyond it (not tested).
- Spell counts rise from ~357 to ~405-423 per match vs gt ~458 (OY under-produces at 290 in stage 1 only).

## Runtime
`decode_variants` (forward/backward, FFBS, 5-9 Viterbi/candidate paths, MBR scoring) averaged 2.0 s per half at S=50
(160 calls in stage 2, 330 s total wall including scoring) and 10.4 s per half at S=200 (ablation mean, mixed S 50/200).
Forward-backward alone 0.2 s per ~70k-frame half; FFBS 20 samples 0.4 s. Single process, CPU, deterministic.

## Unverified
- The 2-step LOMO selection includes the KEEP and cost grid, so the +0.02 is an LOMO estimate, but only 5 matches and the
  grid top edge (TEMP 3) was selected; not retested beyond it.
- Why MBR equals argmax (sampled posterior too smooth, or utility favors the smoothest candidate) was not investigated.
- Not run on locked matches, not run on BAR-ATH, no byte-for-byte regeneration test. The stage-1/stage-2 logs hold the
  top-level LOMO table; the ablation was run for the fixed-cell comparison and is not an independent LOMO test.
- Posterior calibration of the temperature was not assessed apart from spellF1.
