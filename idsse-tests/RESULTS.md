# Results: IDSSE tests A and B

Run 2026-10-27 on the 7 open IDSSE matches (all 7 used, none dropped). Code, thresholds and decision rule are
those of `PREREGISTRATION.md`, unchanged. Raw numbers: `results.json`. Plots: `plots.png`.

## Verdicts

| ID | Measured | Threshold | Verdict |
|---|---|---|---|
| A0 sample | 57 corners (57 in event data, 0 dropped) | >= 50 | PASS (barely) |
| A1 validity | Spearman rho = 0.244 (p = 0.067, n = 57) | pass >= 0.6, weak 0.4-0.6 | **FAIL** |
| A2 a style exists | Kruskal-Wallis H = 15.5, p = 0.0037 (5 defending teams with >= 5 corners) | p < 0.05 | PASS (does not gate) |
| A3 robustness | rho = 0.110 on the primary degraded draw (20 draws: mean 0.107, sd 0.108, range -0.13 to 0.26) | >= 0.5 | **FAIL** |
| **Test A** | | A0 + A1 + A3 | **FAIL** |
| B0 sample | 2,351 open-play losses; base rate Y1 = 32.8%, Y2 = 5.9% | report | n/a |
| B1 primary | coef(S) = -0.0626 per player (odds ratio 0.94), bootstrap 95% CI [-0.118, -0.002] | coef < 0, CI excludes 0 | PASS (CI upper bound is -0.002) |
| B2 usefulness | LOMO AUC 0.9064 (S + ball_x) vs 0.9065 (ball_x), gain = -0.0001 | >= +0.03 | **FAIL** |
| B3 secondary (Y2) | coef(S) = -0.0013, CI [-0.075, +0.084] | as B1 | FAIL (does not gate) |
| **Test B** | | B1 + B2 | **FAIL** |

**Decision rule:** neither passes. Phase 2 falls back to a descriptive analysis and the idea is re-chosen when the
organisers publish their suggested problems on 26 Nov.

## Detail

Test A
- Corners per match: MX 10, N1 7, OH 6, OY 2, PY 11, QQ 15, R9 6. Reference stickiness: mean 0.896, sd 0.056.
  In the 3 s before a corner most players stand still, so the reference hardly varies. That makes any correlation hard to get.
- A1 components: speed ratio alone rho = 0.282, negative pair distance alone rho = 0.144. Exploratory log speed ratio
  (not preregistered): 0.255. No variant gets near 0.4.
- A3: the identity-free index loses almost everything under degradation (0.244 to 0.107 mean over 20 draws). The primary
  draw (seed 0, chosen before looking) gives 0.110. No draw reaches 0.5.
- A2: defending-team means of the index range from -1.77 to +0.79 (z units). Düsseldorf (club DFL-CLU-00000P) is 21 of the 45 corners in teams
  with >= 5 corners. See Caveats before reading anything into this.

Test B
- Losses per match: MX 408, N1 394, OH 301, OY 344, PY 324, QQ 302, R9 278. S: mean 1.51, sd 1.24, corr with ball_x = -0.22.
- 16.1% of losses already have the ball in the losing team's own final third at the loss frame (these count as Y1 = 1).
- ball_x alone gives AUC 0.91 for Y1, because Y1 is defined by ball x. S adds nothing out of sample.
- Exploratory (not preregistered): excluding losses already in the zone, coef(S) = -0.032, CI [-0.085, +0.033] (includes 0);
  AUC gain -0.0004. So even the B1 pass does not survive this reading of Y1.

## Deviations (all dated 2026-10-27, made before seeing outcomes unless stated)

1. **Data host and parser.** Files were downloaded from the `pysport/idsse-data` Hugging Face mirror that kloppy uses
   (same DFL XML as the figshare release), not figshare itself. Parsed with a small streaming parser (`idsse.py`), not kloppy,
   to avoid holding 7 x 140k frames of Python objects. Verified on a few frames: BallStatus 1 = alive, 0 = dead;
   BallPossession 1 = home, 2 = away (matches kloppy's mapping). 25 fps, frame N and timestamp agree.
2. **Corner kick frame (alignment).** The CornerKick `EventTime` is not the kick: the ball was 10-30 m from the flag at that
   timestamp in most corners. The kick frame is the dead-to-alive BallStatus switch nearest to EventTime within +-6 s
   (found for all 57; median 1.56 s after EventTime). 0.2 s after that frame the ball is a median 6.0 m from the flag,
   which is consistent with a ball just kicked. The 3 s window is the 75 frames (25 fps, full rate) before the kick frame.
3. **Roles.** Reference stickiness excludes goalkeepers (DFL role TW) as the prereg says "outfield". The identity-free
   measure has no roles, so it uses all players of each team in the box, goalkeepers included. "Nearest attacker" means
   any attacking outfield player on the pitch, not only those in the box.
4. **Stickiness definition.** "Share of window frames in which the same attacker is the nearest" was read as the share of
   frames in which the defender's modal nearest attacker is nearest. Defenders are those in the box (+2 m) at the kick frame.
5. **Speed ratio.** Computed as (mean speed of defenders in the box over the window) / (same for attackers), not a mean of
   per-frame ratios (unstable when attackers stand still). Speeds come from 5-frame (0.2 s) position differences per slot
   in both clean (A1) and degraded (A3) runs, not from the DFL `S` field, so A1 and A3 differ only by the degradation.
   After the ID shuffle these differences glitch at shuffle points, as intended.
6. **Pair distance.** Rectangular Hungarian matching (min(n_def, n_att) pairs) per frame, then mean over frames with both
   sides present.
7. **A3 draw and ball dropout.** One primary draw (seed 0) decides the verdict; 20 draws are reported as robustness. The ball
   is not used by the identity-free index, so the "drop the ball on 41% of frames" step was not applied: it has no effect by
   construction. Player dropout (10% per player-frame) and within-team slot shuffle every 2 s (absolute-frame blocks of 50)
   were applied. The reference stays on clean data.
8. **"Losing team's final third (last 35 m)"** was read as the 35 m nearest the losing team's own goal (x <= -17.5 in the
   losing team's attacking frame), since that is where the winner's attack arrives. A loss with the ball already there counts
   as Y1 = 1 (flagged above; sensitivity reported as exploratory). The possession window ends at the first frame where the winner
   is no longer in possession or the ball is dead, or at the end of the half.
9. **S definition.** Both counts use players with x < ball x in the frame where the losing team attacks +x: the losing team's
   outfield players there minus the winning team's outfield players there. Attack direction per half comes from each team's
   goalkeeper mean x. Goalkeepers excluded by role.
10. **Y2.** Shot time = DFL `CalculatedTimestamp` when present (tracking-synced), else `EventTime`; any `ShotAtGoal` by the winning
    team in [loss, loss + 15 s], same half.
11. **Models.** Unpenalised logistic regression, raw units. Bootstrap CI is the percentile interval over 1,000 resamples of the 7
    matches (with replacement). LOMO AUC is averaged over 7 folds.
12. **Possession flicker.** The possession-change rule was applied literally. 61 of the 2,351 changes are followed by a possession
    run shorter than 5 frames; they were not filtered.
13. **Where it ran.** On the remote host `sonushub` (Python 3.12.12, venv, Linux) at the user's request. The downloaded data
    (2.7 GB) lives at `sonushub:~/idsse-tests/data/`, not in the local `data/` folder.

## Caveats

- **Small n.** 57 corners total, 2 in one match. A1/A3 correlations have wide intervals (n = 57: rho = 0.24 is not
  distinguishable from 0.4 or from 0). A FAIL here means "not shown with this sample and these definitions", not "impossible".
- **Clustering.** Corners cluster within teams and matches. A2 groups by defending club across matches; Düsseldorf is in 5 of
  7 matches and contributes 21 corners, so A2 mostly compares one club with four others of 5-7 corners each. A2 is not independent
  evidence of a "style".
- **Test B has only 7 clusters.** The bootstrap CI for B1 barely excludes 0 (upper bound -0.002). With 7 matches that CI is fragile.
  B1 is a small effect (about 6% lower odds per extra player) that adds no predictive value (B2).
- **Reference variance is tiny** (sd 0.056 on 0.90), so A1 may be low because the reference, not the identity-free index, carries little signal.
- Y1 is partly determined by ball position, so AUCs near 0.9 are mostly ball_x.
- Single competition and season (Bundesliga and 2. Bundesliga 2022/23); five of 7 matches feature the same club.
- Nothing here uses More Than A Hack data.

Data: IDSSE, Bassek et al. 2025, CC BY 4.0.
