# Preregistration: two candidate Phase 2 ideas, tested on IDSSE

Written 2026-10-27, before any IDSSE data was downloaded or inspected. The hypotheses, metrics
and pass thresholds below are fixed. Any later deviation is recorded in RESULTS.md with a date
and a reason; thresholds are never changed after looking at outcomes.

## Data

IDSSE: 7 Bundesliga / 2. Bundesliga matches, DFL positional data (25 fps, player IDs, ball,
per-frame ball status and ball possession) plus DFL event data. CC BY 4.0, Bassek et al. 2025.
No More Than A Hack competition data is used anywhere in this test.

## Test A: set pieces read from tracking without player identities

**Question.** Can an identity-free measure, computable on Metrica automatic tracking (no player
IDs, slots that swap), recover how a team defends corners?

**Unit.** Every corner kick in the event data. Window: the 3 s before the kick frame.

**Reference measure (uses IDs, available only on clean data).** For each defending outfield
player inside the penalty area (+2 m margin) at the kick frame: the share of window frames in
which the same attacker is that defender's nearest attacker ("stickiness"). Corner-level value:
mean stickiness over those defenders. High = man-marking, low = zonal.

**Identity-free measure.** Uses only per-frame positions and per-frame speeds, with no
association across frames:
1. speed ratio = mean speed of defenders in the box / mean speed of attackers in the box,
   averaged over the window;
2. pair distance = mean distance of the optimal (Hungarian) defender-attacker matching in the
   box, recomputed every frame, averaged over the window.
Combined index = z(speed ratio) − z(pair distance), z-scored across all corners.

**Hypotheses and pass thresholds.**
- A0 (sample): at least 50 corners across the 7 matches. Below that, report as underpowered.
- A1 (validity): Spearman ρ between the identity-free index and reference stickiness, across
  corners. Pass ρ ≥ 0.6. Weak 0.4–0.6. Fail < 0.4.
- A2 (a style exists): among defending teams with at least 5 corners, Kruskal-Wallis on the
  identity-free index across teams. Pass p < 0.05.
- A3 (robustness): repeat A1 after degrading the data to look like automatic tracking: drop
  10% of player positions at random per frame, drop the ball on 41% of frames, and shuffle
  player IDs within each team every 2 s. The identity-free index is recomputed on degraded
  data; the reference stays on clean data. Pass ρ ≥ 0.5.

**Test A passes** if A0, A1 and A3 pass. A2 is reported but does not gate.

## Test B: rest defence at the moment of loss, then the transition

**Question.** Does the number of players a team keeps behind the ball when it loses possession
predict how fast the opponent attacks, beyond where on the pitch the ball was lost?

**Unit.** Every open-play possession change in the per-frame ball-possession channel: team X
to team Y with the ball alive on both sides of the change. Changes through a dead ball are
excluded. Loss moments come from the DFL possession channel, never from a shape model.

**Predictor.** Rest-defence surplus S at the loss frame = (outfield players of the losing team
between the ball and their own goal line) − (outfield players of the winning team between the
ball and the losing team's goal line). Counts use x-position only, attacking direction
normalised.

**Covariate.** Ball x-position at the loss, normalised so that larger = closer to the winning
team's own goal.

**Outcomes.**
- Y1 (primary): the winning team keeps possession and the ball enters the losing team's final
  third (last 35 m) within 10 s of the loss.
- Y2 (secondary): the winning team takes a shot within 15 s of the loss (event data).

**Hypotheses and pass thresholds.**
- B0 (sample): report the number of losses and the base rate of Y1 and Y2.
- B1 (primary): logistic regression Y1 ~ S + ball_x. The coefficient of S is negative and its
  95% CI from a match-level bootstrap (1,000 resamples of the 7 matches) excludes 0.
- B2 (usefulness): leave-one-match-out AUC of Y1 ~ S + ball_x is at least 0.03 higher than
  Y1 ~ ball_x alone, averaged over the 7 folds.
- B3 (secondary): same as B1 with Y2. Reported, does not gate.

**Test B passes** if B1 and B2 pass.

## Decision rule (fixed now)

- A passes, B passes: A is the Phase 2 headline (more novel), B becomes a supporting section.
- Only one passes: that one is the headline.
- Neither passes: report both honestly; Phase 2 falls back to a descriptive analysis and the
  idea is re-chosen when the organisers publish their suggested problems on 26 Nov.
