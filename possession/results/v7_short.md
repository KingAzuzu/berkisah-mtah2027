# v7 step 3: short-spell loss factor 5 and 8 (IDSSE TUNE, matched2, LOMO)

Scripts `possession/v7_short.py` (v6 recipe cB, seeds 20-24, only SHORT_W changed), `possession/v7_short_eval.py` (v6 decoder: fusion 0.5,
Viterbi 20/5, hook 0.6). Gate fixed before running: mean >= 0.6960 (v6 + 0.01) AND beats v6 on >= 4/5 matches.

| factor | mean [min, max] | per match (MX, N1, OH, OY, PY) | wins vs v6 | macroF1 | pred spells | gate |
|---|---|---|---|---|---|---|
| 3 (v6) | 0.6860 [0.6501, 0.7360] | .6704 .6501 .7360 .6780 .6956 | - | | 381 399 335 364 334 | - |
| 5 | 0.6855 [0.6503, 0.7219] | .6760 .6503 .7219 .6916 .6875 | 3/5 | 0.9248 | 399 411 351 385 347 | FAIL |
| 8 | 0.6813 [0.6455, 0.7147] | .6760 .6455 .7147 .6833 .6872 | 2/5 | 0.9234 | 405 424 358 394 352 | FAIL |

True spells: 511 500 402 456 421. A higher factor raises the predicted spell count as intended but spellF1 stays flat (x5) or drops (x8):
the extra spells are not the right ones. x3 stays. v6 unchanged.
