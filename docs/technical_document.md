# Technical document: frame-level ball possession, BAR-ATH tactical video

Team: **TODO team name** (CSV and document must be named `<team_name>.csv` / `<team_name>.pdf` or `.md`)
Repository: **TODO public repo URL**
Method version: v3 (frozen), commit **TODO final commit hash**

## 1. Summary

The file is produced by a fully algorithmic pipeline with three parts: hand-designed possession rules on the provided tracking, a small temporal convolutional network (TCN) that reads the tracking, and a Viterbi decoder. The TCN (3-seed ensemble) and a gradient-boosted dead-ball classifier were trained **only on five public IDSSE matches**, which we degraded with our own simulator so that they look like the provided tracking. **No competition data (video, tracking, tags, ground truth) was used for training, tuning or parameter choice.** Nothing was labelled by hand on BAR-ATH. The public 5-minute ground-truth window was used only to check alignment (video frame versus tracking, measured latency -0.3 frames), file format and label meaning, never for tuning.

Results (leave-one-match-out, LOMO, on the 5 IDSSE tuning matches, degraded to look like the ATD input; scored with the organiser's `possession_metrics.py`; spread is [min, max] over the 5 matches; sources in section 8):

| method | profile | spellF1 mean [min, max] | macroF1 mean |
|---|---|---|---|
| v1 safety net (nearest player + Viterbi) | matched | 0.2961 [0.2581, 0.3276] | 0.7509 |
| v2 (rules + players-only dead-ball model) | matched | 0.4300 [0.4131, 0.4628] | 0.8138 |
| v2.1 (v2 + dead-ball model with ball features) | matched | 0.4891 [0.4683, 0.5192] | 0.8547 |
| v2.1 | matched2 | 0.5562 [0.5413, 0.5864] | not recorded |
| TCN alone, seed 0 | matched2 | 0.6169 [0.5648, 0.6505] | 0.9028 |
| v3 fusion, single seed 0 / 1 / 2 | matched2 | 0.6267 / 0.6209 / 0.6250 (min-max of one seed: 0.5891-0.6525 / 0.5692-0.6586 / 0.5966-0.6495) | 0.9089 / 0.9062 / 0.9099 |
| **v3 = 3-seed ensemble + v2.1 fusion** | matched2 | **0.6411 [0.5973, 0.6813]** | 0.9139 |

"matched" and "matched2" are two iterations of the same simulator (`possession/degrade_matched.py` and its second version); v1 and v2 were not rerun under matched2, so the rows are not directly comparable across profiles. Within matched2, v3 beats v2.1 on 5 of 5 matches. On one match, differences below about 0.04 spellF1 are noise, and the mean over 5 matches is the figure to read.

**Locked check.** Two IDSSE matches (`J03WQQ`, `J03WR9`) were never used for any choice and were evaluated once on the v2 method after it was frozen (spellF1 0.463 and 0.467, macroF1 0.819 and 0.837, against 0.314 and 0.317 for the safety net; `possession/logs/locked_v2.json`). **v3 has no locked check**: it was chosen and frozen on the five tuning matches only, and the LOMO numbers above are therefore somewhat optimistic for it.

BAR-ATH aggregates from the human's own run of v3 (no score is available): 199,319 rows, validator passed; graded frames H 54,091, A 36,836, D 67,930; 392 possessions, mean duration 9.3 s; ball seen on 0.593 of frames.

## 2. Approach and why

The task ranks whole possessions (spellF1), so the output must be a clean sequence of spells, not frame-wise guesses.

**v3 pipeline** (`possession/pipeline.py`, `possession/tcn_v3.py`, parameters in `possession/params_v3.json`):

1. **Rules and dead-ball evidence (v2.1)** (`possession/rules.py`, `possession/dead.py`). Team evidence from the nearest player of each team to the ball, kept only when the ball is seen, a player is within 1.0 m and it is not a duel; in other stretches the last validated team is held (strength 0.3 if unchanged, 0.6 if it changes). A HistGradientBoostingClassifier (200 iterations, learning rate 0.08, 15 leaves, `random_state=0`) gives p(dead) per frame, now with ball features (`models/dead_v2_1.joblib`); it adds `0.5 * logit(p_dead)` to the D column and damps the team columns.
2. **TCN** (`possession/tcn_lib.py`). A dilated residual temporal convolutional network, 7 blocks, 64 channels, 175,747 parameters, receptive field about +-7.4 s at 5 Hz. Input: 26 per-frame features that are invariant to player permutation (pooled team and ball statistics, so tracklet and slot identity changes do not matter). Output: log-probabilities for H / A / D. Three seeds (0, 1, 2), 1,500 training steps each, trained with team-swap augmentation; the per-frame log-probabilities are averaged over the seeds.
3. **Fusion and decoder.** Evidence = 0.5 * TCN ensemble log-prob + 0.5 * v2.1 evidence. Viterbi over H / A / D with switch cost 20 for H<->A and 5 for alive<->D (`TEMP` 1). The labels are overwritten with D on frames 110,808 to 113,741 (second half starts at 113,742, per the organiser notes). Look-ahead is used (allowed, FAQ item #15). Inference is CPU, float32, 1 thread, deterministic algorithms.

**Why this.** The error analysis of v2.1 (`possession/results/error_analysis.md`, LOMO, matched2, 5 matches) showed where score is lost:

- Of the true spells missed at IoU 0.3, 56% are assigned the wrong team, 35% are one same-team spell with poor overlap (boundary or merge), 7% are predicted D.
- Dead-ball boundaries are good: 89-91% of D starts and ends are within 1 s of truth (median 0.00 s).
- Oracle tests on the decoded output: giving the model the true D frames raises spellF1 only from 0.556 to 0.567; giving it the true team on frames where it says alive raises it to 0.726. So the headroom is in team assignment, not in dead-ball detection.
- Short spells are the hard part: of 724 true spells shorter than 2 s, 70% are missed; spells over 15 s are missed 4% of the time.

A learned model with a long temporal window can use the whole movement pattern around the ball, including when the ball is not seen, which the nearest-player rule cannot. That is why the TCN was tried, and why v3 uses it for team assignment while keeping the v2.1 evidence in the fusion.

**Simulator matched to the provided tracking.** The first simulator used assumptions taken from DATA.md. After the human computed aggregate statistics of the real ATD tracking (aggregates only, no rows were shared), we matched the simulator to them (`possession/results/matched_degrade.md`):

| statistic | simulator | real ATD |
|---|---|---|
| ball seen share | 0.569 | 0.593 |
| ball-missing run mean (s) | 17.9 | 17.2 |
| player tracklet mean (s) | 114 | 107 |
| ball jitter | sd 0.01 m | about 1-2 cm |
| objects per frame | 20.2 | 20.49 |

Several statistics still miss the target by more than 30% (missing-run p10, tracklet p10/p25/p75, share of ball speeds above 40 m/s, stationary-ball speed p90; see the file). The simulator hides whole dead-ball and outside-pitch episodes from the ball with some probability so that the ball speed median and outside-pitch share match; that is an assumption inferred from aggregates, not measured directly.

**Label-free check of that assumption** (`possession/ball_loss_check.py`, run on the provided tracking by the human, aggregates only): P(ball missing | v2 says D) = 0.702 versus P(ball missing | v2 says alive) = 0.101 on the real tracking, against 0.636 and 0.256 in the simulator. The real tracker does lose the ball mostly when it is dead, which is what the ball-feature dead-ball model relies on. (v2's D comes from a players-only model, so the check does not read the ball it is testing.)

## 3. Assets used and dropped

Used: the provided ATD tracking (ball position, team-labelled player slots) only; the organiser's scorer; the IDSSE public data (section 4).

Not used: the video, the homography, Smart Tagging tags, the broadcast feed, any event data of the competition game, Ball Radar, any detector or neural model trained on the match video.

Tried and dropped (all on IDSSE; spellF1 is LOMO mean over the 5 tuning matches unless noted):

| idea | result | source |
|---|---|---|
| Explicit-duration HSMM decoder | 0.2877 against 0.3017 reference, no gain | `ws1_hsmm.md` |
| Posterior argmax / minimum-Bayes-risk (MBR) decoding | +0.02 on the safety net, negative on better evidence (0.4367 -> 0.4302), Viterbi kept | `ws3_mbr.md`, `integration.md` |
| Context stack (second-stage HistGB over windows of first-stage output) | 0.511-0.520 against 0.5562, wins 0/5 matches in most variants (best fused variant 0.546, 1/5) | `error_analysis.md` |
| Camera centre as ball estimate | median distance 23.6 m to the ATD ball, fails the 10 m bar fixed beforehand | HANDOVER |
| Restart (Smart Tagging) evidence | +0.018 on the safety net under simulated tags; throw-in team rule 94.3% accurate on IDSSE; not in the final method | `restarts.md` |
| Extra training data (3 Metrica sample games) and pitch-mirror augmentation for the TCN | best config 0.6405 fused against baseline 0.6242 (+0.016), below the +0.02 bar fixed beforehand; no winner, **not used** | `tcn_extdata.md` |
| Other external data (PFF World Cup 2022) | not downloaded: no licence text, broadcast-derived tracking with a different distribution | `tcn_extdata.md` |
| Ball Radar (neural ball and holder inference) | not used | HANDOVER |

The extra-data experiment is why v3 contains no external data beyond IDSSE: the rule was set before the run and the best configuration did not meet it.

## 4. Training data

Learned components: the three TCN weight files `possession/models/tcn_v3_s{0,1,2}.pt` and the dead-ball classifier `possession/models/dead_v2_1.joblib`.

- Data: five matches of the IDSSE open dataset, `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, 25 fps tracking, with DFL `BallStatus` (dead / alive) and `BallPossession` (home / away) as labels.
- Source and licence: Bassek, M., Rein, R., Weber, H. et al., "An integrated dataset of spatiotemporal and event data in elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. Released by DFL / IDSSE under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). The labels are the dataset's own (produced by DFL operators). We added no hand labels. IDSSE use was approved by the organisers (FAQ issue #16).
- No external data in v3. Metrica Sports sample games (acknowledged: Metrica Sports, https://github.com/metrica-sports/sample-data) and SkillCorner data appear only in experiments that were not adopted (section 3); if that changes, this section will list them.
- The IDSSE input is degraded by our matched simulator `possession/degrade_matched.py` (seed 0): the ball alternates between seen and missing runs (lognormal), whole dead-ball and outside-pitch episodes are hidden with some probability, players appear in tracklets with gaps and are reassigned to team-fixed slots (46 slots used), ball jitter sd 0.01 m. Constants were set from the aggregate statistics of the provided tracking (section 2) in two tuning iterations. Simulator assumptions that the real file does not confirm are listed in `matched_degrade.md`.
- TCN training: the 5 tuning matches, 1,500 steps, batch 64 windows of 60 s, team-swap augmentation, seeds 0, 1, 2, GPU training with bf16. Training is not byte-reproducible; the committed weights are the artifact (FAQ issue #11). Model selection used LOMO on those five matches with a rule fixed beforehand; the ensemble was chosen because it beats the mean single seed by 0.017.
- Two further IDSSE matches (`J03WQQ`, `J03WR9`) were held back for a locked check (done once for v2 only, see section 1).
- None of the training or tuning data comes from the provided game.

## 5. Possession rules implemented

The rules below are inputs to v3 through the v2.1 evidence (50% of the fusion); the TCN supplies the rest of the team evidence.

- **Alive play with a close player**: possession goes to the team of the nearest player within 1.0 m of the ball. If both teams have a player within 1.0 m (a duel), this frame gives no evidence.
- **Loose ball / flight**: no evidence is generated; the last validated team keeps possession and it flips only when the opposing player arrives at the ball. Measured basis: during flight the truth equals the last close player's team in 77.8% of frames (`ws2_rules.md`).
- **Challenges / duels**: no team is credited; the held team stays until a clear arrival.
- **Dead ball**: D wherever the combined evidence says dead, because the tracking names a team even while the ball is dead. Possession of the team that will restart is not predicted.
- **Restarts**: no special rule. Frames 110,808 to 113,741 are forced to D.
- **Ungraded frames** are written D; the scorer ignores them where truth is X.
- **Decoding**: Viterbi with switch cost 20 (H<->A) and 5 (alive<->D) removes flicker.

## 6. Limitations

- No locked check for v3 (section 1). The LOMO figure is optimistic: the choice of v3 over v2.1, the fusion weight and the seed ensemble all looked at the same five matches.
- All tuning is on IDSSE degraded by our own simulator, and the simulator is matched only to aggregate statistics of the provided tracking. Real ATD tracking may have identity swaps, wrong team labels, duplicated players or ball behaviour that we did not model. There is no BAR-ATH score to compare with.
- The simulator's dead-ball hiding is an assumption supported by one label-free aggregate (0.702 versus 0.101), not by direct measurement.
- Team assignment remains the main loss (oracle gap 0.556 -> 0.726 for v2.1). Short possessions under 2 s are mostly missed, and predicted spell counts are below true counts (about 330-390 against 400-510 on IDSSE), so some possessions are merged. BAR-ATH produced 392 predicted possessions.
- Dead-ball features depend on IDSSE pitch size and coordinate conventions; their behaviour on video-derived tracking is only indirectly checked.
- Inference identity was verified on one machine type (Windows, same CPU, pinned wheels); another CPU architecture is untested.
- With more time: more training matches and augmentation (mirror was the cheapest candidate, +0.010 fused), restart-based team evidence, a locked check of the final method, cross-machine determinism.

## 7. How to reproduce

Python 3.12, CPU only, no GPU needed. The match data is not in the repository; the weights (`possession/models/tcn_v3_s{0,1,2}.pt`, `possession/models/dead_v2_1.joblib`) and parameters (`possession/params_v3.json`, `possession/params_v2_1.json`) are committed. `requirements-submission.txt` pins every dependency; torch is `torch==2.11.0` from the PyTorch CPU index (`+cpu` build).

    git clone --branch possession-dev TODO-repo-URL
    cd <clone>
    py -3.12 -m venv venv
    venv\Scripts\python -m pip install -r requirements-submission.txt
    $env:MTAH_DATA_DIR = "D:\mtah-data"
    venv\Scripts\python -m possession.bar_ath --method v3 --out <team_name>.csv
    venv\Scripts\python evaluation\validate_submission.py <team_name>.csv

Expected: 199,319 rows, validator passes. Torch is imported only for `--method v3`; seeds are fixed and inference uses `torch.set_num_threads(1)` and `torch.use_deterministic_algorithms(True)`.

Determinism evidence (`possession/results/reproducibility.md`):

- v3 predictor on IDSSE `J03WMX` half 0: label SHA-256 `95ed2024...e93691f3c` at 1, 4 and 16 threads and on a second 1-thread run; the same hash in a fresh `git clone` with a fresh Python 3.12 venv (CPU torch 2.11.0) at 1 and 16 threads. (The v2 path gave `4dd0d8de...bacfa` at 1, 4, 16 threads and in a fresh clone.)
- Full CSV on the competition data: the human ran v3 twice; the files are identical (hash compare True), and the validator passed.

## 8. Sources of numbers

`possession/results/matched_degrade.md` (simulator versus target, v1/v2/v2.1), `error_analysis.md` (loss causes, oracles, context stack), `tcn.md` (TCN, seeds, ensemble), `tcn_extdata.md` (extra data and mirror), `reproducibility.md`, `integration.md`, `ws1_hsmm.md`, `ws2_rules.md`, `ws3_mbr.md`, `ws4_dead.md`, `restarts.md`; `possession/logs/locked_v2.json` (locked check of v2); `possession/params_v3.json`; `possession/ball_loss_check.py` (ball-missing check); `HANDOVER.md` (status of 1 Nov 2026). BAR-ATH aggregates (row count, label counts, possession count, ball-seen share, validator verdict, rerun hash comparison) were run by a team member in their own terminal and reported as aggregates only.

Rules compliance: no hand labels on BAR-ATH, no human-produced data about it, no detector trained on the match video, public 5-minute window used for alignment and correctness only, IDSSE cited above (CC BY 4.0).
