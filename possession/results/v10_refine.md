# v10 boundary refiner (diagnostic, not wired)
Refiner: HistGradientBoosting P(true change of same type at f), candidates within +-2 s (step 0.2 s) of each v8 change, features = frame_features,
multi-scale local diffs, TCN b/log-probs, eR, offset, change type. Strict LOMO (trained on other 4 matches' v8 fold preds, merged-truth targets).
Move if score(best) > score(stay) + margin; order kept valid. Design margin 0.05, variant 0.15 (declared before running).

| match | v8 merged | design | variant |
|---|---|---|---|
| J03WMX | 0.763 | 0.763 | 0.763 |
| J03WN1 | 0.719 | 0.714 | 0.717 |
| J03WOH | 0.786 | 0.786 | 0.786 |
| J03WOY | 0.734 | 0.728 | 0.732 |
| J03WPY | 0.779 | 0.773 | 0.778 |
| mean | 0.7563 | 0.7528 | 0.7553 |
| wins vs v8 | - | 0/5 | 0/5 |
| vs raw DFL (info) | 0.6517 | 0.6484 | 0.6507 |

Gate (mean >= 0.7663 and >= 4/5 wins): FAIL for both.
Caveats: single GBM config, no tuning (by design); local kinematic features did not locate boundaries better than the TCN boundary head;
oracle bound was only +0.017 at 1 s, so headroom is small. Refiner training used v8 fold predictions (not final-model predictions).
