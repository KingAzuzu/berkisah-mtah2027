# WS2: possession rules from the literature

Files (all new): `possession/rules.py` (evidence + predict), `rules_run.py` (LOMO experiment entry point),
`rules_measure.py` (DFL-convention measurement), `rules_ablate_safety.py` (control), logs `possession/logs/rules_*.json|txt`.
Every number below is from a command that was run; logs hold the raw output.

## Commands (from D:\barca, `D:\barca\idsse-tests\.venv\Scripts\python -m ...`)

- `possession.rules_measure` -> logs/rules_measure.json
- `possession.rules_run --stage N` (N = 1..6; stage 6 with `--name rules_final`), 4 worker processes
- `possession.rules_ablate_safety` -> logs/rules_ablate_safety.json
- `possession.lomo` -> reference (logs/reference_safety.json)

## 1. DFL convention (clean IDSSE, 5 tuning matches pooled, aggregate only)

- 1771 direct H<->A changes on alive frames; 519 alive->dead changes.
- The label flips when the receiver arrives, not when the passer kicks. At the change frame the new team's
  nearest player is within 1.5 m in at least 90% of changes (q10/q50/q90 of first frame within 1.5 m = 0/0/0
  relative to the change). Median distance of the new team's nearest player to the ball: 3.2 m at -50 frames,
  1.5 m at -5, 0.6 m at 0. Old team: last within 1.5 m at median -7 frames (q10 -56, q90 -1).
- Ball in flight (alive, nearest player >= 3 m): 27% of alive frames (105,019 of 383,437). Truth equals the team of
  the last close (<1.5 m) player in 77.8% of them, the next close player's team in 72.2%. When last and next close
  team agree (76,721 frames), truth = that team 84.1%; when they differ (28,012 frames), truth = last team 60.5%
  (so it has already flipped to the receiver in about 40%). So possession mostly stays with the passer through
  flight, switching shortly before the receiver is at the ball.
- Nearest-team agreement with truth by ball-to-nearest-player distance: <1 m 91.8%, 1-1.5 m 82.4%, 1.5-3 m 68.4%,
  3-6 m 58.8%. Nearest player is a weak cue beyond about 1.5 m.

## 2. Method

Validated gain: ball seen, nearest player within R, not a duel (both teams within 1 m), and a kick signature nearby
(velocity-vector change >= BV m/s over K=5 frames, or direction change >= 25 deg, dilated +-KW frames) or ball slow
(< VSLOW) or signature uncomputable. Other frames hold the last validated team: strength W_HOLD if the validated
team on both sides is the same, W_DIFF if it changes. Validated frames put -W_VAL on the other team. Team columns
get safety's dead damping; D column is `safety.emissions` unchanged. Decoded with `safety.viterbi`.
Ball gaps up to 25 frames are interpolated only for the signature, not for distances.

## 3. LOMO results (clean grid-only params, chosen on 4 matches, scored on the 5th)

Final grid (stage 6, 36 cells): R=1.0, BV {0.5,1.0}, KW {10,20}, W_HOLD 0.3, W_DIFF {0.3,0.6,1.0}, SWITCH_TEAM {5,10,20}.
Other params fixed: VSLOW 1.5, W_VAL 3, GAP 25, ALLOW_UNK 1, DUEL_R 1.0, params.json for SWITCH_DEAD/SCALE/V0.

| match | ref spellF1 | WS2 spellF1 | ref macroF1 | WS2 macroF1 | spells pred/gt ref | spells pred/gt WS2 |
|---|---|---|---|---|---|---|
| J03WMX | 0.3002 | 0.4173 | 0.7784 | 0.8107 | 359/511 | 320/511 |
| J03WN1 | 0.2804 | 0.3764 | 0.7402 | 0.7490 | 366/500 | 347/500 |
| J03WOH | 0.3203 | 0.3811 | 0.7628 | 0.7835 | 351/402 | 317/402 |
| J03WOY | 0.2814 | 0.3810 | 0.7326 | 0.7590 | 357/456 | 344/456 |
| J03WPY | 0.3259 | 0.4366 | 0.7728 | 0.7928 | 351/421 | 334/421 |
| LOMO mean | 0.3017 [0.2804, 0.3259] | 0.3985 [0.3764, 0.4366] | 0.7573 | 0.7790 | | |

Gain +0.097 spellF1 on the mean; every match improves by 0.06 to 0.12 (well above the ~0.04 noise level).
Chosen thresholds were identical across folds except KW (10 or 20): R 1.0 m, BV 0.5 m/s, W_HOLD 0.3, W_DIFF 0.6,
SWITCH_TEAM 10. Earlier stages (s1-s5 logs) gave 0.3725, 0.3936, 0.3984, 0.3962, 0.3996 on their own grids.

## 4. Ablations (these change the story)

- Plain safety net with R_CLOSE 1.0 / 1.5 / 3.0: mean over matches 0.3006 / 0.3122 / 0.3017 (LOMO 0.3122). A smaller radius
  alone is worth about +0.01.
- Rules, R=1.0 BV=0.5 KW=20 (stage 5, means over the 5 matches, not LOMO): W_HOLD 0 and W_DIFF 0: 0.3090;
  W_HOLD 0.3 only: 0.3203; W_DIFF 0.6 only: 0.3929; both: 0.3997.
- Kick validation off (VSLOW 1e9 makes every frame valid): both hold terms on 0.3999 vs 0.3997 with validation.
  The kick-signature validation contributes nothing measurable here; the whole gain comes from holding the last
  team through unvalidated stretches, chiefly the W_DIFF term (hold the passer until the receiver arrives).
- R=0.75 is worse than 1.0 (0.370-0.373 vs 0.396-0.400); W_VAL 6 is worse than 1.5-3.
- Predicted spells remain below truth (317-347 vs 402-511) so fragmentation is not the remaining error.

## 5. Unverified

- Only the degraded IDSSE tuning matches (degrade seed 0) were scored; not tested on other degrade seeds, the
  locked matches, or BAR-ATH. The degrader's ball-gap and slot model are assumptions, so transfer is unknown.
- Kick validation may matter on real noisy tracking (IDSSE ball is clean where seen); not shown here.
- Stage grids were chosen sequentially looking at the mean over all five matches, so the best-cell numbers in
  s1-s5 are optimistic; the LOMO figure (selection on 4, scoring on the 5th) is the honest estimate, though the
  grid ranges themselves were informed by earlier all-match stages.
- Truth-based measurement in section 1 uses 1.5 m / 3 m cutoffs chosen by hand for reporting only.
