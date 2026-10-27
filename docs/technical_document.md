# Technical document: frame-level ball possession, BAR-ATH tactical video

Team: **TODO team name** (CSV and document must be named `<team_name>.csv` / `<team_name>.pdf` or `.md`)
Repository: **TODO public repo URL**
Method version: v2 (frozen), commit **TODO commit hash**

## 1. Summary

The file is produced by a fully algorithmic pipeline: hand-designed possession rules on the provided tracking, a small gradient-boosted dead-ball classifier trained only on public IDSSE matches, and a two-switch-cost Viterbi decoder. **No competition data (video, tracking, tags, ground truth) was used for training, tuning or parameter choice.** The public 5-minute ground-truth window was used only to check correctness (frame alignment, file format, label meaning), never for tuning. Nothing was labelled by hand on BAR-ATH.

Results (all on public IDSSE data, degraded to look like the ATD input; sources in section 8):

| evaluation | v2 spellF1 / macroF1 | reference spellF1 / macroF1 |
|---|---|---|
| leave-one-match-out, 5 IDSSE tuning matches (`possession/results/integration.md`) | 0.4367 / 0.8232 | 0.3017 / 0.7573 |
| locked check, 2 IDSSE matches never used for any choice (`possession/logs/locked_v2.json`) | J03WQQ 0.463 / 0.819, J03WR9 0.467 / 0.837 | 0.314 / 0.746, 0.317 / 0.764 |

The "reference" is our own safety net (nearest-player evidence plus Viterbi). Locked-match scores were obtained once, after the method was frozen. On one match, differences below about 0.04 spellF1 are noise (`integration.md`). BAR-ATH aggregate numbers: **TODO paste from the human's run: number of possessions, mean duration, ball-seen share, validator verdict. No score is available for BAR-ATH.**

## 2. Approach and why

The task ranks whole possessions (spellF1), so the output must be a clean sequence of spells, not frame-wise guesses. We use three parts, scored everywhere with the organiser's `possession_metrics.py`:

1. **Team evidence from rules** (`possession/rules.py`). Per frame, a score for H versus A from the nearest player of each team to the ball, kept only when the ball is seen, a player is within 1.0 m and it is not a duel; in unvalidated stretches the last validated team is held (strength 0.3 if the team does not change, 0.6 if it does, so the passer keeps the ball until the receiver arrives). This follows what we measured on DFL truth: the label flips when the receiver arrives rather than when the passer kicks, and the nearest player is a weak cue beyond about 1.5 m (`ws2_rules.md`, section 1).
2. **Dead-ball probability** (`possession/dead.py`). A HistGradientBoostingClassifier (200 iterations, learning rate 0.08, 15 leaves, `random_state=0`) gives p(dead) per frame from player-only features (160 of 241; every ball-derived feature removed). It adds `0.5 * logit(p_dead)` to the D column and damps the team columns.
3. **Decoder**: Viterbi over H / A / D with switch cost 20 for H<->A and 5 for alive<->D (`params_v2.json`), which removes flicker. The labels are overwritten with D on frames 110,808 to 113,741 (second half starts at 113,742, per the organiser notes). Look-ahead is used (allowed, FAQ item #15).

Why this and not more: each alternative below was tested on the same protocol and either did not help or was not robust. The whole pipeline is CPU, a few hundred lines of numpy plus one sklearn model.

## 3. Assets used and dropped

Used: the provided ATD tracking (ball position, team-labelled player slots) only; the organiser's scorer; the IDSSE public data (section 4).

Not used: the video, the homography, Smart Tagging tags, the broadcast feed, any event data of the competition game.

Tried and dropped (numbers are leave-one-match-out spellF1 on the 5 IDSSE tuning matches, reference 0.3017):

| idea | result | source |
|---|---|---|
| Explicit-duration HSMM decoder | 0.2877, no gain | `ws1_hsmm.md` |
| Posterior argmax / minimum-Bayes-risk decoding | +0.02 on the safety net (0.3225); on the better rules evidence it is negative (0.4367 -> 0.4302 with players-only), so Viterbi kept | `ws3_mbr.md`, `integration.md` step 2 |
| Dead-ball model with ball features | 0.4986 clean, but with noise on the ball position (0.3 m) and a model trained on clean data it falls to 0.2919, below the reference. Dropped because the real ball track is video-derived. | `integration.md` step 3 |
| Restart (Smart Tagging) evidence | +0.018 on the safety net under simulated tags (0.3017 -> about 0.320); throw-in team rule 94.3% accurate on IDSSE. Not part of v2. | `restarts.md` |
| Ball Radar (neural ball and holder inference) | not part of v2; not used in the submission | HANDOVER |

## 4. Training data

The only learned component is the dead-ball classifier `possession/models/dead_v2.joblib`.

- Data: five matches of the IDSSE open dataset, `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, 25 fps tracking with DFL `BallStatus` as the dead / alive label (every 5th frame used for training).
- Source and labelling: Bassek, M., Rein, R., Weber, H. et al., "An integrated dataset of spatiotemporal and event data in elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. Released by DFL / IDSSE under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). The labels are the dataset's own (produced by DFL operators). We added no hand labels.
- The IDSSE input is degraded by our own `possession/degrade.py` (ball hidden on 59.3% of frames' complement, i.e. seen on 59.3%, in runs that start more often when players crowd the ball; each player present on 20.5/22 of frames in on/off runs; tracks cut into tracklets and reassigned to 46 team-fixed slots; targets from DATA.md, run lengths and crowding strength are our assumptions) so that the classifier sees input like the competition tracking. The degrader's assumptions are not verified against the real file.
- Selection of parameters (ALPHA, switch costs) used nested leave-one-match-out on those five matches only. Two further IDSSE matches, `J03WQQ` and `J03WR9`, were held back and evaluated once after freezing.
- None of the training or tuning data comes from the provided game.

## 5. Possession rules implemented

- **Alive play with a close player**: possession goes to the team of the nearest player within 1.0 m of the ball. If both teams have a player within 1.0 m (a duel), this frame gives no evidence.
- **Loose ball / flight**: no evidence is generated; the last validated team keeps possession, and it flips only when the opposing player arrives at the ball. Measured basis: during flight the truth equals the last close player's team in 77.8% of frames (`ws2_rules.md`).
- **Challenges / duels**: no team is credited; the held team stays until a clear arrival.
- **Dead ball**: D wherever the classifier says dead, because the tracking names a team even while the ball is dead. Possession of the team that will restart is not predicted in v2.
- **Restarts**: no special rule in v2 (the throw-in and set-piece evidence was tested and left out, see section 3). Frames 110,808 to 113,741 are forced to D.
- **Ungraded frames** are written D; the scorer ignores them where truth is X.

## 6. Limitations

- All tuning is on IDSSE, degraded by our own simulator. Real ATD tracking may have identity swaps, wrong team labels, duplicated players and ball jitter that we did not model, except a Gaussian ball-noise test: at 1.0 m noise spellF1 falls from 0.4367 to 0.3633 (matched) (`integration.md`).
- The rules evidence reads ball distance and speed changes, so it degrades first when the ball track is poor.
- The players-only dead-ball features depend on IDSSE pitch size and coordinate conventions; their performance on the video-derived tracking is unknown (frame AUC 0.978 to 0.988 on IDSSE, `ws4_dead.md`).
- Predicted spell counts are below the true counts (about 300 to 340 versus 402 to 511), so some possessions are merged; we did not find a fix that survived the protocol.
- Rule constants came from WS2 selection that looked at all five tuning matches, so the leave-one-match-out figure is mildly optimistic (`integration.md`, Unverified).
- With more time: model the real tracking errors from the first aggregate statistics of the ATD, add restart-based team evidence, and test cross-machine determinism.

## 7. How to reproduce

Python 3.12, CPU only, no GPU or network needed. The match data is not in the repository; the model weights (`possession/models/dead_v2.joblib`) and parameters (`possession/params_v2.json`) are committed.

    git clone --branch possession-dev TODO-repo-URL
    cd <clone>
    py -3.12 -m venv venv
    venv\Scripts\python -m pip install -r requirements-submission.txt
    $env:MTAH_DATA_DIR = "D:\mtah-data"
    venv\Scripts\python -m possession.bar_ath --method v2 --out <team_name>.csv
    venv\Scripts\python evaluation\validate_submission.py <team_name>.csv

Expected: 199,319 rows, validator passes. Determinism evidence (`possession/results/reproducibility.md`): the v2 predictor on one IDSSE half gives the same SHA-256 (4dd0d8de...bacfa) at 1, 4 and 16 threads and in a fresh clone with a fresh pinned venv. A byte-for-byte rerun of the full CSV: **TODO the human to run twice and `cmp`, then record here**; seeds are fixed (`random_state=0`), there is no randomness at inference.

## 8. Sources of numbers

`possession/results/integration.md` (LOMO and robustness), `ws1_hsmm.md`, `ws2_rules.md`, `ws3_mbr.md`, `ws4_dead.md`, `restarts.md`, `reproducibility.md`; `possession/logs/locked_v2.json` (locked check); `possession/params_v2.json`.

Rules compliance: no hand labels on BAR-ATH, no human-produced data about it, no detector trained on the match video, public 5-minute window used for correctness only, IDSSE cited above.
