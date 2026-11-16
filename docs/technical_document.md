# Technical document: frame-level ball possession, BAR-ATH tactical video

Team: **Berkisah** (submission `Berkisah.csv`, document `Berkisah.md`)
Repository: **TODO public repo URL**
Method version: **v10** (frozen; v6 is the fallback), commit **TODO final commit hash**

## 1. Summary

The file is produced by a fully algorithmic pipeline with three parts: hand-designed possession rules plus a dead-ball classifier on the provided tracking, an ensemble of ten small temporal convolutional networks (TCN) that read the tracking, and a Viterbi decoder. All learned parts were trained **only on five public IDSSE matches**, degraded with our own simulator so that they look like the provided tracking. **No competition data (video, tracking, tags, ground truth) was used for training, tuning or parameter choice.** Nothing was labelled by hand on BAR-ATH. The public 5-minute ground-truth window was used only to check alignment, file format and label meaning, and once to diagnose a failure case (section 6); it never set a parameter or chose a method.

**v10 is a hedge between two readings of "controlled possession".** The DFL labels we train on flip the owner on very short touches (19% of IDSSE spells are X-Y-X runs shorter than 2 s), while CHALLENGE.md says that a touch without control is not a change. We therefore trained the same recipe twice: v6 on the DFL labels and v8 on touch-merged labels (an opponent spell shorter than 2 s between two spells of the same team is merged into them). v10 averages the log-probabilities of all ten nets (5 v6 + 5 v8). It is the minimax-regret choice over the two truths, chosen by a rule fixed before looking at the public window.

Results (leave-one-match-out, LOMO, on the 5 IDSSE tuning matches degraded to look like the ATD input, organiser's `possession_metrics.py`; "merged" = DFL truth with the 2 s touch merge):

| method | spellF1 vs DFL truth | spellF1 vs merged truth | note |
|---|---|---|---|
| v2.1 rules + dead-ball model | 0.5562 | | first matched2 reference |
| v3 = 3-seed TCN + v2.1 fusion | 0.6411 | | |
| v4 = 5-seed TCN (boundary head, EMA, domain randomisation, mirror + TTA) | 0.6639 | | |
| v6 = v4 + loss x3 near short spells, seeds 20-24 | **0.6860** [0.650, 0.736] | 0.7099 | best under DFL |
| v8 = v6 recipe on touch-merged labels | 0.6517 | **0.7563** [0.719, 0.786] | best under merged |
| **v10 = v6 + v8 nets, one ensemble** | **0.6686** | **0.7449** | max regret 0.017 (v6 0.047, v8 0.034) |

v10 per match vs DFL 0.655, 0.615, 0.718, 0.653, 0.702; vs merged 0.752, 0.716, 0.770, 0.721, 0.766. On one match, differences below about 0.04 spellF1 are noise.

**Locked check** (run once after v10 was frozen, `possession/locked_eval_v10.py`, `possession/logs/locked_v10.json`; the two held-back IDSSE matches `J03WQQ`, `J03WR9` were never used for any choice):

| | vs DFL (QQ, R9, mean) | vs merged (QQ, R9, mean) | macroF1 vs DFL |
|---|---|---|---|
| v6 | .726 .703 **0.714** | .733 .712 **0.722** | 0.940 |
| v8 | .705 .663 **0.684** | .745 .749 **0.747** | 0.937 |
| **v10** | .716 .676 **0.696** | .740 .739 **0.739** | 0.938 |

Same ordering as on the tuning matches; v10's worst-case regret 0.019 (tuning 0.017). No sign of overfitting to the tuning matches.

**BAR-ATH aggregates** (human run in their own terminal, aggregates only): 199,319 rows, validator passed, rerun byte-identical; graded H 53,223, A 37,302, D 68,332; 347 possessions, mean 10.4 s (v6 402 / 9.0 s, v8 326 / 11.1 s); D share of graded frames 0.43, in line with typical effective playing time. Public 5-minute window (correctness only, 17 possessions, about +-0.06 per possession): spellF1 0.570, macroF1 0.732, team accuracy while both alive 0.92.

## 2. Approach and why

The task ranks whole possessions (spellF1), so the output must be a clean sequence of spells, not frame-wise guesses.

**v10 pipeline** (`possession/pipeline.py` `v10_predictor()`, `possession/tcn_v10.py`, `possession/v4_lib.py`, `possession/params_v10.json`):

1. **Rules and dead-ball evidence (v2.1)** (`possession/rules.py`, `possession/dead.py`). Team evidence from the nearest player of each team to the ball, kept only when the ball is seen, a player is within 1.0 m and it is not a duel; in other stretches the last validated team is held (strength 0.3 if unchanged, 0.6 if it changes). A HistGradientBoostingClassifier (200 iterations, learning rate 0.08, 15 leaves, `random_state=0`, `models/dead_v2_1.joblib`) gives p(dead) per frame from player and ball features; it adds `0.5 * logit(p_dead)` to the D column and damps the team columns.
2. **TCN ensemble** (`possession/tcn_lib.py`). Dilated residual temporal convolutional network, 7 blocks, 64 channels, 175,747 parameters, receptive field about +-7.4 s at 5 Hz. Input: 26 per-frame features invariant to player permutation (pooled team and ball statistics, so slot identity changes do not matter). Output: log-probabilities for H / A / D plus a boundary probability. Recipe (v6): boundary head (BCE weight 0.5, Gaussian target sigma 0.4 s); EMA 0.995; domain randomisation (6 extra degraded variants per IDSSE half with simulator constants drawn at random: ball seen 0.48-0.70, missing-run median 2.5-6 s, tracklet median 20-60 s, jitter 0.5-4 cm, dead-ball hiding probability 0.45-0.80); team-swap and pitch-mirror augmentation with mirror test-time augmentation; class and boundary losses x3 within +-1 s of a true H<->A change next to a run shorter than 3 s; 1,500 steps; seeds 20-24. v8 nets: identical recipe and seeds, labels touch-merged (`possession/v8_convention.py`, K = 2 s). v10 averages the 10 nets' log-probabilities and boundary probabilities with equal weight.
3. **Fusion and decoder.** Evidence = 0.5 * ensemble log-prob + 0.5 * v2.1 evidence. Viterbi over H / A / D with switch cost 20 (H<->A) and 5 (alive<->D), both scaled by (1 - 0.6 * ensemble boundary probability). Frames 110,808 to 113,741 are forced to D (second half starts at 113,742). Look-ahead is used (FAQ #15). Inference: CPU, float32, 1 thread, deterministic algorithms.

**Why this.** The error analysis (`possession/results/error_analysis.md`) showed that of the true spells missed, 56% get the wrong team and 35% are one same-team spell with poor overlap; dead-ball boundaries are good (89-91% within 1 s); giving the decoder the true team on alive frames would lift spellF1 from 0.556 to 0.726, giving it the true D only to 0.567. So the headroom is in team assignment, which a learned model with a long temporal window does better than a nearest-player rule, especially while the ball is not seen. The convention question (short touches) then moved spellF1 by 0.03-0.05 depending on the truth, more than any modelling change after v6; hence the hedge.

**Simulator matched to the provided tracking** (`possession/degrade_matched2.py`, `possession/results/matched_degrade.md`). Constants were set from aggregate statistics of the real ATD tracking computed by a team member (aggregates only): ball seen 0.569 sim vs 0.593 real, ball-missing run mean 17.9 vs 17.2 s, tracklet mean 114 vs 107 s, ball jitter about 1-2 cm, 20.2 vs 20.49 objects per frame. Label-free check of the dead-ball hiding assumption (`possession/ball_loss_check.py`): P(ball missing | dead) 0.702 vs P(ball missing | alive) 0.101 on the real tracking (simulator 0.636 / 0.256).

## 3. Assets used and dropped

**Used by the submitted method:** the provided ATD tracking (ball position, team-labelled player slots) only; the IDSSE public data for training (section 4).

**Not used by the submitted method:** the video, its audio, the homography, Smart Tagging, the broadcast feed, any event data of the competition game, any detector.

**Used only in label-free diagnostics** (run by a team member, aggregates only; none feeds the submission): Smart Tagging intervals (`tag_truth_check`, `dead_check`), the homography (camera-centre test, `yolo_prep`), the tactical video (off-the-shelf COCO YOLO11x ball detection, `yolo_check_fast`) and its audio (`whistle_check`).

Tried and dropped (IDSSE LOMO unless noted; every gate was fixed before the run):

| idea | result | source |
|---|---|---|
| Explicit-duration HSMM decoder | no gain (0.2877 vs 0.3017) | `ws1_hsmm.md` |
| MBR / posterior decoding | negative on good evidence; on v8 0.7545 vs 0.7563 | `ws3_mbr.md`, `v10_hedge_mbr_seeds.md` |
| Context stack (second-stage HistGB) | 0.511-0.546 vs 0.5562 | `error_analysis.md` |
| Camera centre as ball estimate | median 23.6 m to the ATD ball, fails 10 m bar | HANDOVER |
| Smart Tagging team prior (v5) / restart evidence | 0.6586 vs 0.6639, 1/5 wins | `v5_tags.md`, `restarts.md` |
| Metrica / SkillCorner extra training data | +0.016 (below +0.02 bar); SkillCorner labels smoother than DFL, no D | `tcn_extdata.md`, `tcn_v4.md`, `v11.md` |
| Decoder retune, wide nested grid (v6 E1, v7) | 0.6790 / 0.6887 vs gates 0.6839 / 3 of 5 wins | `v6.md`, `v7_decoder.md` |
| Short-spell loss x5 / x8 | 0.6855 / 0.6813 vs v6 0.6860 | `v7_short.md` |
| Ceiling of the recipe on clean (undegraded) IDSSE | 0.724 vs DFL: 0.85 is out of reach with this recipe | `v7_ceiling.md` |
| 3000 steps, longer context, boundary refiner, 10 seeds | 0.7493 / 0.7347 / 0.7553 / 0.7495 vs v8 0.7563 | `v9.md`, `v10_refine.md` |
| Cross-fitted distillation (teacher on clean features) | below v8 | `v11.md` |
| Phase-shift test-time augmentation | 0.7545 vs 0.7563 | `v12_phase.md` |
| GBM fusion / auxiliary ball-position head | 0.7460 / 0.7520 vs 0.7563 | `v13.md` |
| Stronger augmentation (noise, time stretch, dropout) | 0.7406, 0/5 wins | `v14.md` |
| Ball Radar (per-player set model) as extra team evidence | +0.004 on both truths, gate +0.01 | `possession/v15_br_probe.py` |
| Control-aware touch merge (merge X-Y-X only without clear control) | 0.7053 vs 0.7067 (avg of truths) | `possession/v16_ctrl_merge.py` |
| Video ball fill, simulated (frozen v10) | perfect fill +0.03; 1 m error -0.02 to -0.04; 2 m -0.14 | `possession/v17_fill_sim.py` |
| Off-the-shelf YOLO11x ball detector on the tactical video (label-free, vs seen ATD ball) | 21.7% of detections within 0.5 m (gate 80%), 0.59 false detections per frame | `possession/yolo_check_fast.py` |
| Transductive self-training on the unlabeled held-out match | 0.7554 vs 0.7563 | `possession/v18_selftrain.py` |
| Referee whistle from the video audio | no enrichment near D transitions (1.13, gate 3) | `possession/whistle_check.py` |
| Team-label repair / jersey classifier | real tracking: one team > 11 players in 18% of frames, 83% swap pattern, not at track jumps; simulated cost about 0.01 unless swaps sit at the ball | `possession/team_label_check.py`, `v19_team_noise_sim.py`, `team_swap_diag.py` |
| Players-only dead-ball model when the ball is seen (v20) | 0.7009 vs 0.7067, fails the no-harm gate | `possession/v20_dead_mix.py` |
| v10 retrained on 7 IDSSE matches (v10c1) | fails its sanity rule (in-sample -0.010 to -0.037) | `possession/v10c1_sanity.py` |

## 4. Training data

Learned components: `possession/models/tcn_v6_s{20..24}.pt`, `possession/models/tcn_v8_s{20..24}.pt` and the dead-ball classifier `possession/models/dead_v2_1.joblib`.

- Data: five matches of the IDSSE open dataset, `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, 25 fps tracking, with DFL `BallStatus` (dead / alive) and `BallPossession` (home / away) as labels; for the v8 nets the same labels with the 2 s touch merge (an algorithm, no human decision).
- Source and licence: Bassek, M., Rein, R., Weber, H. et al., "An integrated dataset of spatiotemporal and event data in elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. DFL / IDSSE, **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). The labels are the dataset's own; we added no hand labels. IDSSE use was approved by the organisers (FAQ #16).
- No other external data in the submitted weights. Metrica Sports sample data and SkillCorner open data (MIT, (c) SkillCorner) were used only in experiments that were not adopted.
- Input degradation: matched simulator (seed 0) plus 6 domain-randomised variants per half (`possession/v4_prep.py`).
- Training: 1,500 steps, batch 64 windows, GPU with bf16; not byte-reproducible, so the committed weights are the artifact (FAQ #11).
- Two further IDSSE matches (`J03WQQ`, `J03WR9`) were held back and evaluated once for v10 after freezing (section 1).
- None of the training or tuning data comes from the provided game.

## 5. Possession rules implemented

The rules are inputs to v10 through the v2.1 evidence (50% of the fusion); the TCN ensemble supplies the rest.

- **Alive play with a close player**: the team of the nearest player within 1.0 m of the ball. If both teams have a player within 1.0 m (a duel), no evidence.
- **Loose ball / flight**: no evidence; the last validated team keeps possession until an opponent arrives at the ball (during flight the truth equals the last close player's team in 77.8% of IDSSE frames, `ws2_rules.md`).
- **Short touches**: handled by the v6/v8 hedge (section 1), not by a hard rule.
- **Dead ball**: D wherever the combined evidence says dead, because the tracking names a team even while the ball is dead.
- **Restarts**: no special rule. Frames 110,808 to 113,741 are forced to D.
- **Decoding**: Viterbi with switch costs 20 (H<->A) and 5 (alive<->D), scaled down where a boundary is expected.

## 6. Limitations

- **Dead ball with a visible, slow ball.** In the public window, v10's D recall is 0.33 (precision 0.98): two stoppages of 20 s and 15 s with the ATD ball seen throughout and moving slowly are labelled alive. The cause is the dead-ball model's ball features, which learned from the simulator that a dead ball is usually missing; a players-only model labels these stoppages D. On IDSSE the case barely exists (D recall with the ball seen 0.998), so the fix (v20) could not be validated and costs 0.006 where it can be measured. We did not change the method from 5 minutes of truth; over the whole match v10 labels 43% of graded frames D, a plausible amount.
- **Team labels in the provided tracking.** In 18% of frames one team has more than 11 labelled players and the other fewer (swap pattern). Our features pool per team, and the simulated cost is about 0.01 when swaps are random, up to 0.03-0.05 if they sit at the ball; a label-free check shows they are only mildly enriched near the ball. The simulator does not swap teams.
- **Convention.** If the organisers' truth follows DFL on short touches, v6 would score about 0.02 more; if it ignores them, v8 would. v10 limits the loss either way to about 0.02.
- All tuning is on IDSSE degraded by our simulator, matched only to aggregates of the provided tracking. The recipe's ceiling on clean IDSSE is 0.724 vs DFL.
- Short possessions under 2 s are mostly missed and predicted spell counts are below true counts on IDSSE.
- Inference identity was verified on one machine type (Windows, pinned wheels); another CPU architecture is untested.

## 7. How to reproduce

Python 3.12, CPU only. The match data is not in the repository; weights and parameters (`possession/params_v10.json`, `possession/params_v2_1.json`) are committed. `requirements-submission.txt` pins every dependency; torch is `torch==2.11.0` from the PyTorch CPU index.

    git clone --branch possession-dev TODO-repo-URL
    cd <clone>
    py -3.12 -m venv venv
    venv\Scripts\python -m pip install -r requirements-submission.txt
    $env:MTAH_DATA_DIR = "<folder that contains the tactical/ data folder>"
    venv\Scripts\python -m possession.bar_ath --method v10 --out Berkisah.csv
    venv\Scripts\python evaluation\validate_submission.py Berkisah.csv

Or `.\run_all.ps1 v10`, which also reruns and compares the hash. Expected: 199,319 rows, validator passes.

Determinism: `python -m possession.repro_check --method v10` on IDSSE `J03WMX` half 0 (CPU, 1 thread; numpy 2.5.4, scikit-learn 1.9.1) gives label SHA-256 `d93596b02321e548cf255cf9c55fd608bbd0c4fae3681984ca45e108adcacdd5` (letters `98bc42d0f5a99d865ab791b231dfdd6389ce555984fd89917c56b8d7b0e0cc83`), identical at 1 and 4 threads. On the competition data the human ran v10 twice; the files are byte-identical and the validator passed.

## 8. Sources of numbers

`possession/results/*.md` (one file per experiment, named in section 3), `possession/logs/locked_v10.json`, `possession/logs/v10/hedge.json`, `possession/logs/v15`-`v20/*.json`, `possession/params_v10.json`, and `HANDOVER.md` (dated log of every experiment, gate and decision). BAR-ATH aggregates (row and label counts, possessions, validator verdict, rerun comparison, public-window scores, label-free diagnostics) were run by a team member in their own terminal and reported as aggregates only.

Rules compliance: no hand labels on BAR-ATH, no human-produced data about it, no detector trained on the match video, the public 5-minute window used for alignment and correctness only (one failure-case diagnosis, no parameter or method chosen from it), IDSSE cited (CC BY 4.0).
