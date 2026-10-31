# TCN with extra training data (Metrica sample games) and pitch-mirror augmentation

## Licences / terms
- Metrica Sports sample data, https://github.com/metrica-sports/sample-data : no formal licence. README "Legal stuff": "Please be responsible with the use of this data." and "If you use it for anything public, please aknowledge the source." Acknowledge Metrica Sports.
- PFF FC World Cup 2022, https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data : no licence text on the page, only "If you intend to publish your findings on social media, please tag us @PFF_FC". Access is by a sign-up form; NOT downloaded (human step). Note it is broadcast-derived tracking (players outside the view are missing), a different distribution from IDSSE/Metrica.

## Data used
Metrica Sample Game 1, 2 (CSV tracking + events) and 3 (EPTS tracking + JSON events), 25 fps, 105x68 m; 6 halves, ~429k frames (~86k at 5 Hz). Raw files ~210 MB total (game 1 ~66 MB, 2 ~62 MB, 3 ~55 MB), in idsse-tests/data/external (gitignored, not committed). Loader: possession/extdata.py.

## Label derivation (events + tracking ball status; no hand labels)
- H/A: team of the last controlled event (PASS, SHOT, SET PIECE, RECOVERY, CARRY). BALL LOST, CHALLENGE (aerial/ground/tackle), duels never change it; only a controlled event by the other team does.
- D: before the first restart; from BALL OUT, FAULT RECEIVED, CARD, offside, hand ball, end half, challenge fouls, goals (from shot end) until the next controlled event; plus any ball-untracked run >= 2 s (tracking ball is missing for 75-96% of event-dead frames, so this is the tracker's ball status). Short ball gaps are linearly interpolated.
- Unverified: event frame timing (BALL OUT may be logged at last touch), the 2 s threshold. Not checked against any independent truth.

## Sanity stats (all frames)
| match | D share | mean H/A spell (s) |
|---|---|---|
| MET1 / MET2 / MET3 | 0.43 / 0.43 / 0.32 | 9.3 / 10.0 / 10.1 |
| IDSSE TUNE (5) | 0.42-0.52 (mean 0.47) | 5.5-7.7 (mean 6.8) |
Metrica-derived spells are ~1.5x longer and D share lower than DFL truth (coarser events miss short possessions and brief stoppages): the extra labels are systematically smoother than IDSSE truth.

## Protocol
Extra halves degraded with degrade_matched2 (seed 0, ids MET1-3), added to every LOMO fold's training set (IDSSE held-out match never in training), evaluation identical to tcn_run.py (1500 steps, same decode and 50/50 fusion with R). Mirror: flip x,y of the centre/ball features in the feature pool (labels unchanged; the team swap augmentation already exists) and average log-probs of original + mirrored input at test time. Seeds 0,1,2; code possession/tcn_ext_run.py, per-run logs possession/logs/tcn_ext_*.json. Pitch-flip interpretation: x,y flip with labels kept (a true symmetry); H/A swap was already in tcn_lib.

## Results (LOMO IDSSE TUNE, spellF1, mean over 5 matches; spread = SD over 3 seeds of the match-mean)
| config | alone | fusion with R | macro (alone) | per-match fusion (MX,N1,OH,OY,PY) |
|---|---|---|---|---|
| R (v2.1, m2) | 0.5562 | - | - | - |
| IDSSE only (rerun, seed 0 reproduces 0.6169/0.6267) | 0.6138 +- 0.0063 | 0.6242 +- 0.0030 | 0.903 | .613 .585 .653 .629 .641 |
| + mirror | 0.6279 +- 0.0035 | 0.6339 +- 0.0034 | 0.907 | .627 .595 .655 .642 .650 |
| + Metrica | 0.6244 +- 0.0016 | 0.6347 +- 0.0037 | 0.907 | .613 .603 .672 .622 .663 |
| + Metrica + mirror | 0.6344 +- 0.0013 | 0.6405 +- 0.0059 | 0.911 | .640 .616 .673 .616 .657 |

## Decision (rule fixed beforehand: mean >= 0.627 + 0.02 = 0.647 and better on >= 4/5 matches)
No winner. Best config (Metrica + mirror, fused) gives 0.6405: +0.016 over the rerun baseline fusion (0.6242), below the +0.02 bar (needs 0.6467), better than 0.6267 on 4/5 matches (not on J03WOY). Both ingredients help a little and add up (+0.010 mirror, +0.010 Metrica on fusion, alone +0.014 / +0.011); effects are at the level of seed spread x2-3, so modest but consistent in direction. Not wired into bar_ath or requirements. Mirror alone is cheap (no extra data) and is the safest candidate to adopt.
