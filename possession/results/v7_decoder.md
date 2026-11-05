# v7 decoder retune on v6 nets (IDSSE TUNE, matched2, nested LOMO)

Script `possession/v7_decoder.py` (copy of `v6_e1.py`), saved v6 LOMO log-probs `logs/v6/e2_s{20..24}_*.npz`, organiser scorer, CPU.
Grid: SWITCH_TEAM {4,6,8,12,16,20} x SWITCH_DEAD {1,2,3,5,8} x hook {0.6,0.8,0.9,1.0} x fusion {0.5..0.9} = 600 points.
Gate fixed before running: nested mean >= 0.6960 (v6 + 0.01) AND beats v6 on >= 4/5 matches.

| | mean [min, max] | per match (MX, N1, OH, OY, PY) |
|---|---|---|
| v6 decoder (20, 5, 0.6, 0.5) | 0.6860 [0.6501, 0.7360] | .6704 .6501 .7360 .6780 .6956 |
| nested retune | 0.6887 [0.6417, 0.7266] | .6810 .6417 .7266 .6933 .7008 |

Wins 3/5, +0.0027 mean: **FAIL**. Nested picks (16,8,0.8,0.7), (16,3,0.6,0.8), (12,8,0.6,0.7), (12,5,0.6,0.5), (16,8,0.8,0.7): interior of the
grid, so the edge problem of v6 E1 is gone. Best in-sample point (16,8,0.8,0.7) 0.6949 (optimistic, not nested); the top 5 points are within
0.0005 of each other, so the surface is flat. Conclusion: the decoder is not where the remaining score is. v6 unchanged.
(JSON key `v4` in `logs/v7/dec.json` = the reference point, which is the v6 decoder.)
