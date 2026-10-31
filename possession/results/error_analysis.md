# Error analysis of R (v2.1 under matched2, LOMO, TUNE, seed 0)

Script `possession/error_analysis.py`, reference preds from `possession/m2_lib.py`. Spell matching at IoU 0.3 (loosest threshold) unless stated; pooled over 5 matches.

## Spell counts per match
| match | pred spells | true spells | true missed @0.3 | pred unmatched @0.3 |
|---|---|---|---|---|
| J03WMX | 387 | 511 | 208 | 84 |
| J03WN1 | 389 | 500 | 194 | 83 |
| J03WOH | 349 | 402 | 138 | 85 |
| J03WOY | 376 | 456 | 178 | 98 |
| J03WPY | 327 | 421 | 173 | 79 |

## True spells by length (pooled)
| length | n true | matched @0.3 | missed | miss rate |
|---|---|---|---|---|
| <2s | 724 | 215 | 509 | 0.70 |
| 2-5s | 713 | 430 | 283 | 0.40 |
| 5-15s | 599 | 509 | 90 | 0.15 |
| >15s | 254 | 245 | 9 | 0.04 |

Matched at 0.3 but not at 0.7 (boundary loss): 546 of 1399 (0.39).

## Why true spells are missed (pooled)
| cause | n | share of missed |
|---|---|---|
| wrong team | 499 | 0.56 |
| one same-team spell, IoU<0.3 (boundary/merge) | 308 | 0.35 |
| predicted D | 60 | 0.07 |
| fragmented (>=2 same-team pred spells overlap) | 16 | 0.02 |
| mixed | 8 | 0.01 |

## Predicted spells unmatched @0.3 (pooled)
429 of 1828 predicted spells unmatched; 183 of those are <2 s.

| truth under the spell | n | share |
|---|---|---|
| wrong team | 283 | 0.66 |
| right team, bad IoU | 117 | 0.27 |
| mostly truth D | 27 | 0.06 |
| mixed | 2 | 0.00 |

## Boundary offsets, predicted minus truth (nearest predicted transition of the same type), seconds

| boundary | n true | signed p10/p50/p90 | abs<=1 s | abs<=3 s | median abs |
|---|---|---|---|---|---|
| alive -> D (D start) | 519 | -0.33/0.00/0.20 | 0.89 | 0.93 | 0.00 |
| D -> alive (D end) | 512 | 0.00/0.00/0.12 | 0.91 | 0.96 | 0.00 |
| direct H<->A change | 0 | n/a | nan | nan | nan |

## Oracle tests (spellF1 of the same decoded preds)
| match | R | truth D, our team | truth team, our D |
|---|---|---|---|
| J03WMX | 0.558 | 0.553 | 0.748 |
| J03WN1 | 0.541 | 0.552 | 0.727 |
| J03WOH | 0.586 | 0.608 | 0.702 |
| J03WOY | 0.547 | 0.576 | 0.716 |
| J03WPY | 0.548 | 0.548 | 0.734 |
| mean | 0.556 | 0.567 | 0.726 |

Truth D, our team: frames where truth is D become D, others take our team (our D filled with our previous team). Truth team, our D: frames where we say D stay D, others take the true team (truth D filled with last true team).

## Context stack (A), LOMO under matched2 (`possession/ctx_stack.py`, logs/ctx_stack.json)
Reference R = 0.5562 [0.5413, 0.5864]. Stage-2 HistGB on stage-1 + windows (+-0.5/2/6 s), out-of-fold stage-1 p_dead.
| candidate | spellF1 mean [min,max] | wins vs R |
|---|---|---|
| stack, R switch costs | 0.511 [0.486, 0.552] | 0/5 |
| stack, switch 10/5 | 0.520 [0.494, 0.560] | 0/5 |
| stack, switch 5/2.5 | 0.518 [0.484, 0.576] | 0/5 |
| R evidence + 0.5 x stack log-prob | 0.546 [0.527, 0.594] | 1/5 |
| R evidence + 1.0 x stack log-prob | 0.540 [0.519, 0.583] | 0/5 |
Not adopted. Restart-geometry features not tried: D boundaries are within 1 s for ~90% of transitions and the truth-team oracle
shows the headroom is in team assignment, not in D.
