# Ball-fill upper bound (IDSSE TUNE, matched2, frozen v4)

Caveat: per-fold v4 models were not saved (only log-prob npz); frozen models trained on all 5 TUNE matches are used, so the result is IN-SAMPLE (absolute levels inflated; compare conditions only).
Ball seen: baseline 0.615; restored in (b) 0.057 of frames (alive and hidden); (c) 0.030. Restored = truth + same jitter; dead-ball hiding kept. Code: possession/ball_fill_bound.py.

| condition | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | mean |
|---|---|---|---|---|---|---|
| (a) baseline matched2 | 0.717 | 0.699 | 0.791 | 0.731 | 0.735 | 0.7346 |
| (b) restore all hidden-alive | 0.724 | 0.725 | 0.817 | 0.771 | 0.775 | 0.7622 |
| (c) restore ~50% (random runs, seed 0) | 0.714 | 0.714 | 0.807 | 0.753 | 0.762 | 0.7501 |

(b) - (a) = +0.0276; (c) - (a) = +0.0155; threshold 0.02: yes.
Single-match differences below ~0.04 are noise; one degrade seed (0).
