# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

We are a team in More Than A Hack 2027 (Metrica Sports, Barça Innovation Hub, MWCapital).
Phase 1: label every frame of the BAR-ATH tactical video `H` / `A` / `D` / `X`. Ranked on
`spellF1` (whole possessions matched by temporal IoU 0.3-0.7), tie-break `macroF1`.
Deadline 17 Nov 2026. Top 20 go to Phase 2 (26 Nov - 14 Jan, jury). Top 5 pitch on 2 Mar 2027.
Plan, state and owners: `HANDOVER.md`. Organiser docs: CHALLENGE, DATA, EVALUATION, RULES, FAQ.

## Hard rule: you never see the competition data

T&C §8 forbids sending the competition videos, datasets or any extract of them to a hosted LLM.
You are a hosted LLM.

- Never read, list, grep, open or print anything in the competition data folder (`MTAH_DATA_DIR`,
  default `D:/mtah-data`), or any file named `BAR-ATH_*`, `*_atd.txt`, `*_atd.xml`,
  `*_homography.csv`, `*_smart_tagging.xml`, `*.mp4`, or our submission CSVs in
  `evaluation/results/`.
- Never run a command that loads the competition data. Every function that loads it must call
  `mtah_guard.refuse_under_claude()` first; it exits when `CLAUDECODE` is set.
- Build and test only on IDSSE (public, CC BY 4.0) and Metrica's public sample data.
- The human runs anything on BAR-ATH in their own terminal and pastes back aggregate numbers only
  (scores, counts, one-line summaries). If they paste data rows, say so and do not use them.
- `.claude/hooks/guard_data.py` blocks the obvious paths. It is a safety net, not a reason to try.

## Competition rules that shape the code

- Labels come from an algorithm. No hand labels on BAR-ATH for any purpose, including choosing
  parameters (issue #10). Hand labels on other games are fine if documented (#8).
- No human-produced data about BAR-ATH from any source (event data, corrected tracking).
- The public 5-minute ground truth checks correctness only. Never tune on it.
- Look-ahead is allowed (#15).
- The CSV must regenerate byte for byte: commit weights, run inference on CPU, pin dependencies,
  fix seeds (#11). Intermediate outputs may be committed with the script that makes them (#14).
- A detector trained on the match video with the provided tracking as labels is not allowed (#2).
- External data about other games is fine; IDSSE is approved (#16). Cite it (CC BY 4.0).
- Tune on IDSSE only. Matches `J03WQQ` and `J03WR9` are the locked final check: never tune on them,
  evaluate on them once, after the method is frozen.

## Traps already known

- Frames 110,808-113,741 are graded and all `D` (second half kicks off at 113,742).
- The tracking names a team even while the ball is dead. Overwrite with `D` whenever the ball is dead.
- ATD `frame_id` = video frame + 1. Smart Tagging time x 25, rounded = video frame.
- Smart Tagging `Team` is `N/A` for every throw-in. Tag columns are Title Case, codes UPPER CASE.
- The broadcast feed is not synced to tactical. A ~164k-row file means you indexed broadcast.
- Exactly 199,319 rows. Run `evaluation/validate_submission.py` on the exact file before upload.

## How to work

- Score with the organiser's `evaluation/possession_metrics.py` everywhere, including on IDSSE.
  On IDSSE the truth is DFL `BallStatus` (0 = dead) and `BallPossession` (1 = home, 2 = away).
- Report spellF1 differences with match-level spread; on one match, below ~0.04 is noise.
- One purpose per change. Back every claim with a number from a command. Say what is unverified.
- Ask the human before adding a dependency or anything that sends data off the machine.

## Commands (run from the repo root; the venv is `idsse-tests/.venv`)

- Python: `idsse-tests/.venv/Scripts/python -m possession.<module>`. Everything is a module of `possession/`.
- Tests: `idsse-tests/.venv/Scripts/python tests/test_bar_ath.py` (one file per test, run directly; no pytest config).
- Score on IDSSE TUNE: `python -m possession.run_idsse --method ...`. CPU determinism: `python -m possession.repro_check --method v10`.
- Competition run (human only, own terminal): `.\run_all.ps1 v10` writes the submission CSV, runs the validator and a
  byte-identical rerun. It calls `python -m possession.bar_ath --method v10`, the inference entry point.
- GPU training scripts run in the background and log to `possession/logs/`. Run GPU jobs one at a time (a job was killed for RAM).
- The `.claude/hooks/guard_data.py` hook also blocks Bash commands whose text names the competition files or the results
  folder; put such text in files with Edit/Write and keep it out of commit messages.

## Architecture

- `possession/data.py`: the one half-match representation (`ball`, `xy`, `team`, `truth`), the IDSSE loader, `TUNE`/`LOCKED`.
  IDSSE is degraded to look like competition tracking (`degrade_matched2.py`, profile "matched2"; `v4_prep.py` adds 6
  domain-randomised variants); `m2_lib.data2()` and the feature pickles in the IDSSE `data/cache` hold them.
- Methods are versioned and frozen, never edited in place: `tcn_vN.py` + `params_vN.json` + `models/tcn_vN_s{seed}.pt`,
  wired through `pipeline.vN_predictor()` into the inference CLI, `repro_check.py` and `run_all.ps1`. Experiments
  (`vN_*.py`) write `possession/results/*.md` and logs, never frozen files.
- Submission method v10 (`tcn_v10.py`): dead-ball model `models/dead_v2_1.joblib` (rules + classifier evidence `eR`) fused
  0.5/0.5 with the mean log-probs of 10 TCN4 nets (5 v6 nets on DFL labels + 5 v8 nets on touch-merged labels,
  `v8_convention.merge`), decoded by `v4_lib.viterbi_scaled`. CPU, float32, 1 thread, deterministic. Fallback v6.
  v10c1 (same, retrained on 7 matches) failed its sanity rule and is not submitted.
- Evaluation protocol: leave-one-match-out on the 5 TUNE matches, gate fixed before each run (mean >= reference + 0.01 and
  >= 4/5 match wins), organiser scorer via `possession/score.py`. Locked matches were evaluated once for v10
  (`logs/locked_v10.json`); `locked_eval_*.py` refuse to rerun.
- Scripts that read competition data (inference CLI, `tag_truth_check`, `dead_check`, `gt_convention_check`, `compare_csv`)
  call `mtah_guard.refuse_under_claude()`; the human sets `MTAH_DATA_DIR` and pastes aggregates back.
- `HANDOVER.md` is the log of every experiment, gate and decision; append to it after each result.
- The user writes in Indonesian and wants short, dense answers and frugal token use.
