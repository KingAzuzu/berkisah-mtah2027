# v8: does the label convention matter? (IDSSE TUNE, diagnostic, nothing frozen)

## Why
- CHALLENGE.md: a touch without control, a deflection that does not intercept, or an aerial duel won without control is NOT a change of
  possession. Competition truth = ball owner of Metrica's corrected tracking (mechanical).
- IDSSE truth (DFL BallPossession) flips on such touches: of 2,290 alive spells in TUNE, 724 (32%) are < 2 s, and 425 (19% of all) are
  short spells sitting directly between two spells of the other team (X Y X). Those are exactly the spells we miss most (70% miss rate < 2 s).
- Context only, not used to choose anything (public window = correctness check, CLAUDE.md): DATA.md documents the public window as
  17 possessions, mean 14.2 s. The first 300 s of the 10 IDSSE TUNE halves under DFL truth have 22-38 possessions (median 29.5), mean
  3.6-9.0 s. Confounder: the home team's style produces long possessions.

## Test (`possession/v8_convention.py`, saved v6 LOMO log-probs, v6 decoder)
merge K: an alive spell < K s between two spells of the other team is relabelled as the surrounding team. K = 2 s fixed a priori.

| K | v6 vs DFL | v6+merge vs DFL | v6 vs merged | v6+merge vs merged |
|---|---|---|---|---|
| 1 | 0.6860 | 0.6854 | 0.7253 | 0.7275 |
| **2** | 0.6860 | 0.6548 | 0.7099 | **0.7310** (5/5 wins vs v6) |
| 3 | 0.6860 | 0.6306 | 0.6808 | 0.7277 |

K = 2: if the competition truth ignores short touches, merging gains +0.021 (5/5); if it is DFL-like, merging loses -0.031 (5/5).
With merge, predicted spell counts match merged truth almost exactly (349/343, 377/360, 315/310, 338/338, 308/309).
The convention is a bet worth about +-0.02-0.03 in spellF1, same order as every model gain since v4. It cannot be settled on IDSSE.
