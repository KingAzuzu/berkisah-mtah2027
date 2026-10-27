# Restart evidence and throw-in team rule (HANDOVER idea 4)

Files: `possession/restarts.py` (rule, simulation, evidence adjustment), `restarts_throwin.py` (Part A),
`restarts_run.py` (Part B), `restarts_report.py` (tables), `smart_tagging.py` + `tests/test_smart_tagging.py` (Part C).
Logs: `possession/logs/restarts_*.json|txt`. All numbers below come from those commands, IDSSE tuning matches only
(`J03WQQ`, `J03WR9` never loaded), degraded inputs from `run_idsse.load`, single process.

Commands (from `D:\barca`, venv python): `-m possession.restarts_throwin`; `-m possession.restarts_run --sigma S --seed K`
for S in 0.5, 1.5 and K in 0, 1, 2 (each runs both bases); `-m possession.restarts_report`; `-m tests.test_smart_tagging`.

## Part A: throw-in team rule

Rule (fixed before looking at results): at the DFL ThrowIn `EventTime` frame (`idsse.to_frame`), take slots whose |y|
comes within 1.0 m of the touchline (34 m) at some frame within +-25 frames (1 s); if several, pick the one nearest the
ball (ball seen in the window) else the most outside; its team is the throwing team. Truth: DFL ThrowIn `Team`.

| match | throw-ins | correct | wrong | no candidate | accuracy |
|---|---|---|---|---|---|
| J03WMX | 52 | 48 | 0 | 4 | 0.923 |
| J03WN1 | 46 | 43 | 0 | 3 | 0.935 |
| J03WOH | 36 | 34 | 1 | 1 | 0.944 |
| J03WOY | 54 | 52 | 0 | 2 | 0.963 |
| J03WPY | 41 | 39 | 1 | 1 | 0.951 |
| all | 229 | 216 | 2 | 11 | 0.943 |

Bar from HANDOVER: accuracy >= 90%. Result: **94.3% overall (no candidate counted as a miss): PASS**, lowest
match 0.923. Among throw-ins where a candidate was found: 99.1%
(2 wrong). No candidate: 11/229 = 4.8%.

Sensitivity (descriptive; the headline rule above was not chosen from these; W12 alone would fall below the bar):

| variant | correct | wrong | none | accuracy |
|---|---|---|---|---|
| W12_m1.0 | 205 | 3 | 21 | 0.895 |
| W25_m0.5 | 210 | 2 | 17 | 0.917 |
| W25_m1.0 | 216 | 2 | 11 | 0.943 |
| W25_m2.0 | 211 | 9 | 9 | 0.921 |
| W50_m1.0 | 217 | 7 | 5 | 0.948 |
| W50_m2.0 | 208 | 19 | 2 | 0.908 |

Rule applied on a jittered frame (window max(25, 2 sigma) frames, one draw, seed 0):

| jitter | correct | wrong | none | accuracy |
|---|---|---|---|---|
| sigma 0.5 s | 214 | 2 | 13 | 0.934 |
| sigma 1.5 s | 212 | 15 | 2 | 0.926 |

Caveats: the throw frame here is the DFL event time; the real tag has its own timing error, covered only by the jitter
rows. A player of either team can stand at the line (an opponent retrieving the ball); that is the source of the wrong
cases. Nothing here says how the rule behaves on the competition tracking, which is unmeasured.

## Part B: restart evidence, LOMO

Simulation assumptions (explicit arguments of `restarts.simulate_tags`, `SIM_ASSUMPTIONS`):
- tags come from DFL EventTime of ThrowIn, CornerKick, GoalKick, FreeKick, KickOff (494 events in the 5 tuning matches);
  a tag is an instant (the real file has ~5 s intervals; only the start would be used);
- Gaussian timing jitter, sigma 0.5 s or 1.5 s, on the tag frame;
- each tag dropped with probability 0.15 independently (421-432 tags remain, depending on variant and seed);
- corner / goal kick / free kick / kick-off keep the true team; throw-in team removed, then filled by the Part A rule on the
  jittered frame (window max(25, 2 sigma frames)); no candidate -> no team boost for that tag;
- not modelled: false-positive tags, wrong teams on non-throw-in tags, interval-shaped tags, correlated errors.

Adjustment (`restarts.adjust`): D score += W_D on [t-PRE, t); tagged team score += W_T on [t, t+POST). Grid (grid params
only, `lomo.evaluate_grid`): PRE {75, 200, 400}, W_D {0, 2, 5}, POST {25, 75}, W_T {0, 1, 3}. Base evidence parameters
and switch costs stay fixed: reference safety = `params.json`; WS2 = `rules_final.json` `best_on_all`. Decoder
`safety.viterbi`. The base row is that base's own LOMO log (`reference_safety.json`, `rules_final.json`). The
W_D = W_T = 0 cell reproduces the base evaluated in-sample (safety 0.3017; rules 0.3997 on its best-on-all params, versus
0.3985 held-out in its own LOMO). Seeds 0, 1, 2 redraw jitter and drops. Columns: LOMO spellF1 mean / min / max, the 5
held-out per-match numbers, mean macroF1.


### base safety, jitter sigma 0.5 s

| variant | LOMO mean | min | max | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | macroF1 |
|---|---|---|---|---|---|---|---|---|---|
| base (reference_safety) | 0.3017 | 0.2804 | 0.3259 | 0.3002 | 0.2804 | 0.3203 | 0.2814 | 0.3259 | 0.7573 |
| + restarts, seed 0 | 0.3200 | 0.2908 | 0.3529 | 0.3152 | 0.2908 | 0.3274 | 0.3137 | 0.3529 | 0.7747 |
| + restarts, seed 1 | 0.3176 | 0.2868 | 0.3546 | 0.3162 | 0.2868 | 0.3312 | 0.2991 | 0.3546 | 0.7731 |
| + restarts, seed 2 | 0.3202 | 0.2897 | 0.3558 | 0.3126 | 0.2897 | 0.3363 | 0.3067 | 0.3558 | 0.7744 |

mean gain over base across seeds: +0.0176 (per seed +0.0184, +0.0159, +0.0186); per-match gain seed 0: J03WMX +0.0150, J03WN1 +0.0104, J03WOH +0.0071, J03WOY +0.0323, J03WPY +0.0270
tag stats seed 0: {'n_events': 494, 'n_tags': 432, 'throw_tags': 202, 'throw_wrong': 2, 'throw_none': 14}
chosen params per held-out match (seed 0): {'J03WMX': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WN1': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WOH': {'PRE': 200, 'W_D': 5.0, 'POST': 75, 'W_T': 3.0}, 'J03WOY': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WPY': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}}

in-sample grid cells (mean over 5 matches, seed 0; chosen on the same 5, so descriptive only):
- base (W_D 0, W_T 0): best 0.3017 at {"PRE": 75, "W_D": 0.0, "POST": 25, "W_T": 0.0}  per match [0.3002, 0.2804, 0.3203, 0.2814, 0.3259]
- D only (W_T 0): best 0.3100 at {"PRE": 200, "W_D": 2.0, "POST": 25, "W_T": 0.0}  per match [0.3024, 0.2766, 0.3366, 0.2977, 0.3369]
- team only (W_D 0): best 0.3145 at {"PRE": 75, "W_D": 0.0, "POST": 75, "W_T": 1.0}  per match [0.3127, 0.2976, 0.3277, 0.2976, 0.3368]
- both: best 0.3235 at {"PRE": 200, "W_D": 2.0, "POST": 75, "W_T": 1.0}  per match [0.3152, 0.2908, 0.3446, 0.3137, 0.3529]

### base safety, jitter sigma 1.5 s

| variant | LOMO mean | min | max | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | macroF1 |
|---|---|---|---|---|---|---|---|---|---|
| base (reference_safety) | 0.3017 | 0.2804 | 0.3259 | 0.3002 | 0.2804 | 0.3203 | 0.2814 | 0.3259 | 0.7573 |
| + restarts, seed 0 | 0.3206 | 0.2888 | 0.3505 | 0.3104 | 0.2888 | 0.3442 | 0.3093 | 0.3505 | 0.7736 |
| + restarts, seed 1 | 0.3189 | 0.2885 | 0.3503 | 0.3180 | 0.2885 | 0.3294 | 0.3085 | 0.3503 | 0.7718 |
| + restarts, seed 2 | 0.3179 | 0.2889 | 0.3494 | 0.3113 | 0.2889 | 0.3359 | 0.3037 | 0.3494 | 0.7728 |

mean gain over base across seeds: +0.0175 (per seed +0.0190, +0.0173, +0.0162); per-match gain seed 0: J03WMX +0.0102, J03WN1 +0.0084, J03WOH +0.0239, J03WOY +0.0279, J03WPY +0.0246
tag stats seed 0: {'n_events': 494, 'n_tags': 421, 'throw_tags': 199, 'throw_wrong': 8, 'throw_none': 5}
chosen params per held-out match (seed 0): {'J03WMX': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WN1': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WOH': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WOY': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WPY': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}}

in-sample grid cells (mean over 5 matches, seed 0; chosen on the same 5, so descriptive only):
- base (W_D 0, W_T 0): best 0.3017 at {"PRE": 75, "W_D": 0.0, "POST": 25, "W_T": 0.0}  per match [0.3002, 0.2804, 0.3203, 0.2814, 0.3259]
- D only (W_T 0): best 0.3103 at {"PRE": 200, "W_D": 2.0, "POST": 25, "W_T": 0.0}  per match [0.306, 0.2791, 0.3304, 0.3012, 0.3346]
- team only (W_D 0): best 0.3115 at {"PRE": 75, "W_D": 0.0, "POST": 75, "W_T": 1.0}  per match [0.3058, 0.2923, 0.3293, 0.2928, 0.3372]
- both: best 0.3206 at {"PRE": 200, "W_D": 2.0, "POST": 75, "W_T": 1.0}  per match [0.3104, 0.2888, 0.3442, 0.3093, 0.3505]

### base rules, jitter sigma 0.5 s

| variant | LOMO mean | min | max | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | macroF1 |
|---|---|---|---|---|---|---|---|---|---|
| base (rules_final) | 0.3985 | 0.3764 | 0.4366 | 0.4173 | 0.3764 | 0.3811 | 0.3810 | 0.4366 | 0.7790 |
| + restarts, seed 0 | 0.4179 | 0.3714 | 0.4620 | 0.4352 | 0.3714 | 0.4139 | 0.4069 | 0.4620 | 0.7957 |
| + restarts, seed 1 | 0.4057 | 0.3577 | 0.4602 | 0.4231 | 0.3577 | 0.3911 | 0.3966 | 0.4602 | 0.7892 |
| + restarts, seed 2 | 0.4149 | 0.3638 | 0.4670 | 0.4301 | 0.3638 | 0.4061 | 0.4074 | 0.4670 | 0.7936 |

mean gain over base across seeds: +0.0144 (per seed +0.0194, +0.0073, +0.0164); per-match gain seed 0: J03WMX +0.0179, J03WN1 -0.0050, J03WOH +0.0328, J03WOY +0.0259, J03WPY +0.0255
tag stats seed 0: {'n_events': 494, 'n_tags': 432, 'throw_tags': 202, 'throw_wrong': 2, 'throw_none': 14}
chosen params per held-out match (seed 0): {'J03WMX': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WN1': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WOH': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WOY': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WPY': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}}

in-sample grid cells (mean over 5 matches, seed 0; chosen on the same 5, so descriptive only):
- base (W_D 0, W_T 0): best 0.3997 at {"PRE": 75, "W_D": 0.0, "POST": 25, "W_T": 0.0}  per match [0.4173, 0.3764, 0.3839, 0.381, 0.4397]
- D only (W_T 0): best 0.4007 at {"PRE": 200, "W_D": 2.0, "POST": 25, "W_T": 0.0}  per match [0.4184, 0.3599, 0.3944, 0.39, 0.4411]
- team only (W_D 0): best 0.4117 at {"PRE": 75, "W_D": 0.0, "POST": 75, "W_T": 1.0}  per match [0.4293, 0.3822, 0.3972, 0.397, 0.4526]
- both: best 0.4179 at {"PRE": 200, "W_D": 2.0, "POST": 75, "W_T": 1.0}  per match [0.4352, 0.3714, 0.4139, 0.4069, 0.462]

### base rules, jitter sigma 1.5 s

| variant | LOMO mean | min | max | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | macroF1 |
|---|---|---|---|---|---|---|---|---|---|
| base (rules_final) | 0.3985 | 0.3764 | 0.4366 | 0.4173 | 0.3764 | 0.3811 | 0.3810 | 0.4366 | 0.7790 |
| + restarts, seed 0 | 0.3971 | 0.3581 | 0.4468 | 0.4197 | 0.3581 | 0.3810 | 0.3800 | 0.4468 | 0.7806 |
| + restarts, seed 1 | 0.4077 | 0.3657 | 0.4504 | 0.4273 | 0.3657 | 0.3994 | 0.3956 | 0.4504 | 0.7899 |
| + restarts, seed 2 | 0.4067 | 0.3633 | 0.4590 | 0.4149 | 0.3633 | 0.3994 | 0.3965 | 0.4590 | 0.7919 |

mean gain over base across seeds: +0.0053 (per seed -0.0013, +0.0092, +0.0082); per-match gain seed 0: J03WMX +0.0024, J03WN1 -0.0183, J03WOH -0.0001, J03WOY -0.0010, J03WPY +0.0102
tag stats seed 0: {'n_events': 494, 'n_tags': 421, 'throw_tags': 199, 'throw_wrong': 8, 'throw_none': 5}
chosen params per held-out match (seed 0): {'J03WMX': {'PRE': 200, 'W_D': 2.0, 'POST': 25, 'W_T': 1.0}, 'J03WN1': {'PRE': 200, 'W_D': 2.0, 'POST': 75, 'W_T': 1.0}, 'J03WOH': {'PRE': 75, 'W_D': 0.0, 'POST': 25, 'W_T': 1.0}, 'J03WOY': {'PRE': 75, 'W_D': 0.0, 'POST': 25, 'W_T': 1.0}, 'J03WPY': {'PRE': 75, 'W_D': 0.0, 'POST': 25, 'W_T': 1.0}}

in-sample grid cells (mean over 5 matches, seed 0; chosen on the same 5, so descriptive only):
- base (W_D 0, W_T 0): best 0.3997 at {"PRE": 75, "W_D": 0.0, "POST": 25, "W_T": 0.0}  per match [0.4173, 0.3764, 0.3839, 0.381, 0.4397]
- D only (W_T 0): best 0.3970 at {"PRE": 200, "W_D": 2.0, "POST": 25, "W_T": 0.0}  per match [0.4107, 0.3581, 0.3939, 0.3854, 0.4367]
- team only (W_D 0): best 0.4044 at {"PRE": 75, "W_D": 0.0, "POST": 25, "W_T": 1.0}  per match [0.4273, 0.3867, 0.381, 0.38, 0.4468]
- both: best 0.4042 at {"PRE": 200, "W_D": 2.0, "POST": 75, "W_T": 1.0}  per match [0.4272, 0.3581, 0.3939, 0.391, 0.4507]


## Part C: Smart Tagging loader

`possession/smart_tagging.py`: `load_restarts(path=None)` calls `mtah_guard.refuse_under_claude()` first, then
`games.require(["smart_tagging"])` (unless a path is given), parses SET PIECES instances with the standard library and
returns `{frame = round(start_s * 25), end_frame, type, team 'H'/'A'/None, half}`; `to_tags()` converts to the
`restarts.adjust` format. Case-insensitive on code, group names and values; N/A and unknown clubs give None.
Tests (`tests/test_smart_tagging.py`: synthetic XML in a temp file, never real data; refusal with `CLAUDECODE` set,
with and without a path), output of `-m tests.test_smart_tagging`:

```
test_load_with_path_outside_claude (__main__.T.test_load_with_path_outside_claude) ... ok
test_parse (__main__.T.test_parse) ... ok
test_refuses_under_claude (__main__.T.test_refuses_under_claude) ... ok
test_to_tags (__main__.T.test_to_tags) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.023s

OK
```

## Summary and unverified

- Throw-in rule: 94.3% on IDSSE vs the 90% bar: pass.
- On simulated tags, LOMO spellF1 gain over base: safety about +0.017 to +0.018 at both jitters (all 3 seeds positive,
  all 5 matches positive at seed 0); rules base +0.014 at sigma 0.5 (seeds +0.019, +0.007, +0.016) but +0.005 at sigma 1.5
  (seeds -0.001, +0.009, +0.008), with J03WN1 negative at both jitters in seed 0 (-0.005, -0.018). Per project convention a
  gap below ~0.04 on one match is noise; the pooled gains are small and seed dependent for the rules base.
- Unverified: Part B rests entirely on the simulation assumptions. Real Smart Tagging timing, drop rate, interval semantics
  (a ~5 s window, not the restart instant), false positives and team accuracy on the competition match are unknown and were
  not measured (not allowed here). The gains are not a prediction for that match.
- The grid mostly chose PRE 200 / POST 75 (POST 75 is the upper grid value); no wider grid was tried.
