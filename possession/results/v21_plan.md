# v21 plan: fix the dead ball in the simulator, then rebuild on it (written 17 Nov 2026, before any v21 run)

## Why

Public-window correctness check (16 Nov): v10 D recall 0.33, the two missed stoppages had the ATD ball seen and slow-moving.
Five public competitor repos (BYTZ, Tailors, DIMACodm, Subject47, BecaData; read 17 Nov) found the cause independently:
DFL freezes the ball at exactly zero speed during dead play. matched2 hides a whole stoppage w.p. 0.63 and leaves the other 37%
as a frozen visible ball, so the dead model (ball features) and the TCN learn "visible still ball = dead, moving = alive". In the
provided tracking the visible dead ball is being carried to the restart spot. Every IDSSE gate since v2.1 measured D in the frozen
world, so v20 (players-only dead) was judged by a biased test.

## Steps and gates (fixed now)

All on IDSSE TUNE, leave-one-match-out, organiser scorer, truth = DFL. Locked matches untouched until a method is frozen.

0. `v21_simstats`: matched3 must reproduce the real aggregates we have (ball seen 0.593 +- 0.04, P(missing|D) 0.70 +- 0.05,
   P(missing|alive) 0.10 +- 0.03) and give a moving visible dead ball (mean speed > 1.5 m/s). Report clean / matched2 / matched3.
1. Symptom check: the v2.1-style dead model trained on matched2, evaluated on matched3, must show low D recall where the ball is
   seen (the real failure). If it does not, matched3 does not capture the failure and the plan stops here.
2. Reference REF = r2 nets (v6 recipe on matched2) + dead model trained on matched2 (ball features), evaluated on matched3.
3. Candidates, each vs the best so far, evaluated on matched3: adopt only if mean spellF1 >= best + 0.01 AND >= 4/5 match wins.
   A  r2 nets + dead model trained on matched3 (ball features)       (CPU only; production could keep the v10 nets)
   A' as A with the players-only dead model
   B  r3 nets (v6 recipe on matched3 + DR) + best dead model of A/A'
   C  r3x (f32 features)
   D  r3xl (f32, DIL_LONG +-27.6 s)
   Seeds: 3 per recipe for screening (20, 21, 22).
4. Decoder: switch costs (H<->A, alive<->D) re-chosen by nested LOMO on the winner's held-out outputs; adopt under the same rule.
5. Freeze the winner as v21 (5 seeds, all 5 TUNE matches), evaluate once on J03WQQ / J03WR9 degraded with matched3, wire into
   bar_ath / run_all / repro_check. The human runs it on BAR-ATH and reports aggregates; public window = correctness only.

matched2 scores are reported next to matched3 for every candidate but do not gate (matched2 contains the frozen-ball artefact).
