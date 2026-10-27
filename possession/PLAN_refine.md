# Refinement plan: research items 1-5

Source: `reports/Riset algoritma possession sepak bola.md`. Planned 27 Oct 2026. Five workstreams run in
parallel, each in its own new files; integration happens afterwards, in one place.

## Reference to beat

`python -m possession.lomo` -> safety net with `params.json`:
**LOMO spellF1 0.3017 [0.2804, 0.3259], macroF1 0.7573, spells pred/gt ~357 / ~458.**
On one match a spellF1 gap below ~0.04 is noise; report all 5 per-match numbers, not only the mean.

## Rules for every workstream (from CLAUDE.md, non-negotiable)

- Never read, list or open anything under `D:/mtah-data`, `data/`, `evaluation/results/`, `evaluation/gt/`,
  or any `BAR-ATH_*`, `*_atd.*`, `*_homography.csv`, `*_smart_tagging.xml`, `*.mp4` file. Never run
  `possession/bar_ath.py`.
- IDSSE only. Tune on `data.TUNE` (5 matches). `J03WQQ`, `J03WR9` (`data.LOCKED`) are never loaded.
- Anything fitted on truth (duration histograms, classifiers, neural nets) is fitted inside
  leave-one-match-out folds: train on 4 tuning matches, predict the 5th. No exceptions.
- Score only with `possession/score.py` (wraps the organiser's `possession_metrics.py`).
- Inputs are the degraded halves from `run_idsse.load(mid)` (cached). `truth` is used only as a label.
- Deterministic: fixed seeds, CPU for anything that feeds the final labels. GPU only for training.
- Python: `D:\barca\idsse-tests\.venv\Scripts\python`, run from `D:\barca` as `-m possession.<module>`.
- Do not edit shared files: `data.py score.py degrade.py safety.py lomo.py run_idsse.py params.json
  baseline.py bar_ath.py`. Create new files only. Import from shared files freely.
- Do not add dependencies, except WS5's PyTorch (approved by the human for GPU training).
- Keep the CPU polite: at most 4 worker processes per workstream.

## Contracts

- Codes: `pm.H = 0`, `pm.A = 1`, `pm.D = 2`.
- **Evidence** = `(T, 3)` float array of per-frame log-scores for H, A, D, where 0 means "this frame says
  nothing" (see `safety.emissions`). Evidence modules expose `evidence(half, params, cache) -> (T, 3)`.
- **Decoder** = `decode(e, params) -> (T,) int8 codes` on an evidence array.
- Experiments use `lomo.evaluate_grid(predict, grid, name)` for grid-only params, or their own
  fold loop (train on 4, score the 5th with `score.score`) for fitted components; either way write
  `possession/logs/<name>.json` and print the per-match numbers.
- Each workstream writes `possession/results/<ws>.md`: what was built, the commands, the LOMO numbers
  vs the reference, per-match spread, what is unverified. Numbers come from commands, not estimates.

## WS1: explicit-duration decoder (HSMM) -> `possession/hsmm.py`

Constant switch costs mean geometric durations, which force a choice between flicker and losing short
possessions. Replace with an explicit-duration semi-Markov Viterbi over H/A/D.

- Durations: per-state log-pmf. Within each fold, from the 4 training matches' truth, smoothed
  histogram (or log-normal mixture) with a geometric tail beyond Dmax (99.5th percentile).
- Transitions between segments: H->A, A->H, H->D, A->D, D->H, D->A (no self-transitions), counted from
  training-fold truth; optionally a scalar weight on the transition and duration terms (grid).
- Emissions: `safety.emissions(safety.features(half), params)` with `params.json` values, times a
  temperature (grid). Prefix sums for segment emission scores.
- Speed: decode at 5 fps (sum emissions over 5-frame blocks) then refine each boundary at 25 fps within
  +/- 5 frames, or vectorise over d. Must run a full half in < 60 s on CPU.
- Report also the spell-duration histogram predicted vs truth (aggregate), before and after.

## WS2: possession rules from the literature -> `possession/rules.py`

New team evidence, decoded with the existing `safety.viterbi` (tune its switch costs jointly).

- First, on clean IDSSE (`data.load_idsse`), measure the DFL convention: around truth possession changes,
  where is the ball relative to the passer and receiver (aggregate numbers only). Does BallPossession stay
  with the passer while the ball is in flight? Write the finding into the results file.
- Gain validation (Link & Hoernig 2017; Vidal-Codina 2022; databallpy defaults 1.5 m / 5 m/s / 10 deg):
  a team gains the ball only at frames where the nearest player is within R and the ball shows a kick
  signature (speed change or direction change) or the ball stops being free.
- Hold the last validated team through ball flight and through ball gaps when the nearest team on both
  sides of the gap agrees; duel (players of both teams within ~1 m) = carry, no new evidence.
- Output evidence `(T, 3)` with the D column from `safety.emissions` unchanged, so the gain is isolated.

## WS3: minimum Bayes risk decoding -> `possession/mbr.py`

- Forward-backward on the same 3-state HMM as `safety.viterbi` (emissions / temperature, switch costs as
  log transition penalties), in log space. Forward-filtering backward-sampling: S paths, fixed seed.
- Candidates: Viterbi at several switch-cost settings, the posterior argmax, and each sample.
- Pick the candidate with the highest mean spellF1 against the samples, using
  `possession_metrics.spell_scores` (one valid run per half). Grid: temperature, S, cost settings.
- Also test a simple keep-threshold: drop predicted spells whose posterior mass is below a threshold.

## WS4: dead-ball classifier -> `possession/dead.py`

- Per-frame features from the degraded half only: players' speed (mean, quantiles, share below 1 and
  2 m/s), team centroids and spreads, players near touchlines / corner arcs / goal areas / centre circle,
  both teams in own halves (kickoff), ball seen / speed / out of bounds, time since ball last seen.
  Rolling windows of 1, 3, 10 s, past and future (look-ahead is allowed).
- `sklearn.ensemble.HistGradientBoostingClassifier` (already pinned), `random_state=0`, LOMO folds.
- Report frame AUC and F1 for D per held-out match, then replace the speed-based D log-odds in the
  emissions with `alpha * logit(p_dead)` and decode with `safety.viterbi` (grid alpha, switch costs).
- Save the final model trained on all 5 tuning matches to `possession/models/dead.joblib` with the
  feature list and versions, for CPU inference later.

## WS5: Ball Radar style ball and possessor imputation (GPU) -> `possession/ballradar/`

Reimplement from the paper (Kim et al., KDD 2023, arXiv 2306.08206). The public repo has no licence: do not
copy its code.

- Install PyTorch with CUDA into `idsse-tests\.venv`, exact versions pinned in
  `possession/ballradar/requirements.txt`.
- Model: per-frame set encoder over slots (permutation invariant, masked for empty slots, team
  embedding), plus the observed ball where seen (imputation, not pure inference), then a bidirectional
  LSTM over time windows. Heads: ball (x, y) and team in possession (H / A / dead).
- Training data: degraded IDSSE tuning matches with several `degrade` seeds as augmentation, plus random
  pitch flips (x and y), since attack direction is unknown on BAR-ATH. Folds: train on 4, predict the 5th.
  Cache predictions under `idsse-tests/data/cache/ballradar/` (gitignored).
- Report: ball error on frames where the ball is hidden, team-possession accuracy on alive frames, and
  spellF1 when its team probabilities replace the nearest-player evidence on hidden-ball frames, decoded
  with `safety.viterbi`.
- Final weights trained on all 5 tuning matches to `possession/models/ballradar.pt`. Inference must be
  CPU, `torch.use_deterministic_algorithms(True)`, fixed thread count, and give identical outputs on two
  runs (test it).

## Integration (after all five report)

Combine the evidence that helped (WS2, WS4, WS5) with the best decoder (WS1 or WS3), one change at a time,
each measured with LOMO. Then freeze, evaluate once on the locked matches, update `bar_ath.py`.
