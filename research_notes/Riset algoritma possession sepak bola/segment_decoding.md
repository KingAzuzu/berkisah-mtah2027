# Segment decoding: turning noisy per-frame H/A/D scores into segments optimised for spell-F1 (temporal IoU matching)

Scope note: about 17 tool calls. Web evidence was thin on several points (search snippets, a few paywalled or binary PDFs). Formulas and pseudo-code below are standard results or my own derivations, so they sit under "Inferences" and are unverified against a source unless stated. Nothing here touched the competition data.

## 1. Hidden semi-Markov / explicit-duration HMMs: efficient Viterbi, duration families, Python libraries and licences

### Takeaway
An explicit-duration Viterbi is O(T·D·K²) naive but easy to write in vectorised numpy, and for K=3 (with H<->A and X-free structure) it is cheap. Existing Python libraries are poor fits: hsmmlearn is GPL-3.0 and archived, edhsmm is archived and author-disclaimed, pyhsmm is MIT but unmaintained and Bayesian/Gibbs-oriented. Write your own ~60-line decoder.

### Cited Findings
- hsmmlearn is licensed GPL-3.0 because it incorporates code from R's hsmm package and wraps the same C++ library; it is for unsupervised learning of explicit-duration HSMMs with an hmmlearn-style API — [hsmmlearn docs/README mirror](https://github.imc.re/ahmedatg/hsmmlearn/blob/master/README.md); [GitHub](https://github.com/jvkersch/hsmmlearn)
- The hsmmlearn GitHub repo is archived and read-only (archived by owner 30 Jan 2023); 112 commits; Python 3.6+ — [GitHub](https://github.com/jvkersch/hsmmlearn)
- edhsmm: pure-Python explicit-duration HSMM (EM per Yu 2010, Viterbi per Benouareth et al. 2008; Gaussian and multinomial emissions; deps numpy, scikit-learn, scipy). Repo archived 26 May 2024; the author says they deleted the pip package and that correctness is not guaranteed — [PyPI](https://pypi.org/project/edhsmm/); [GitHub](https://github.com/poypoyan/edhsmm) (as summarised by search; not fetched directly)
- pyhsmm (mattjj): MIT licence (LICENSE-MIT), README says "not maintained anymore", last tested on Python 3.7, needs a C++11 compiler. Supports HMM/HSMM Bayesian inference by Gibbs sampling, HDP-HMM/HDP-HSMM weak-limit, optional duration truncation (`trunc`), custom duration distributions — [GitHub](https://github.com/mattjj/pyhsmm)
- HSMMs raise HMM cost from O(T·N²) to O(T·N²·D) because of the duration dimension and a cumulative emission product over the d most recent observations (2026 arXiv paper on a tensor formulation of HSMM Viterbi) — [arXiv 2609.16500](https://arxiv.org/pdf/2609.16500)
- Yu & Kobayashi (2003) give a forward-backward for explicit-duration HMMs with lower cost and memory; Datta, Hu & Ray (2008) recast HSMM Viterbi as longest path in a DAG; a RAIRO paper claims HMM-like complexity (linear in T, quadratic in states) for an HSMM decoder (not verified how durations are handled) — [Yu-Kobayashi abstract list](https://typeset.io/pdf/an-efficient-forward-backward-algorithm-for-an-explicit-3jo3p19a00.pdf); [RAIRO](https://www.rairo-ro.org/10.1051/ro/2014053)
- Yu (2010) "Hidden semi-Markov models", Artificial Intelligence, is the standard survey (explicit duration HMM, variable-duration, inference incl. Viterbi and a complexity section per the outline) — [PDF](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf) (body not machine-readable; only outline seen)

### Inferences
Standard explicit-duration Viterbi (my restatement; verify against Yu 2010 before citing). States s in {H,A,D}; duration pmf p_s(d), d=1..Dmax; per-frame log-score e_t(s) with prefix sums E_s(t)=sum_{u<=t} e_u(s).
```
delta[t,s] = max over d in 1..Dmax, s' != s :
     delta[t-d, s'] + logA[s',s] + log p_s(d) + (E_s(t) - E_s(t-d))
```
- Segment emission cost is O(1) via prefix sums, so total cost is O(T·Dmax·K) with the max over s' hoisted: precompute best_prev[t,s] = max_{s'!=s}(delta[t,s']+logA[s',s]) in O(T·K²). For K=3, Dmax=3000 frames (2 min), T=80k per half: ~7e8 cell-ops. Vectorising over d with numpy (one slice per t) gives roughly 80k numpy calls of length 3000 per state: tens of seconds to a few minutes per half. Deterministic, CPU-only.
- Speed-ups: (a) truncate Dmax to the 99.5th percentile of true duration, and handle the tail with a "geometric tail" self-loop; (b) decode at 5 fps (every 5th frame, or windowed means) then refine boundaries at 25 fps within +-1 s, cutting cost by ~25x; (c) for a geometric tail beyond Dmax, run-length-free recursion; (d) for log-concave durations (gamma shape >= 1, log-normal is not log-concave in general) the max over d can use monotone-queue/SMAWK-style tricks; I did not find a reference, treat as unverified; (e) lag-based trick: because log p(d) is additive and E_s is a prefix sum, delta[t-d]-E_s(t-d) + logp(d) is a max-plus convolution over d; for concave logp this is O(T log T) with divide and conquer. Unverified.
- Duration families: log-normal, gamma, or a nonparametric histogram (smoothed) learned from IDSSE spell lengths. Heavy tails with many 1-3 s spells suggest a mixture (short mode + log-normal body) or just a histogram with Dmax truncation. Constant switch penalty (current decoder) equals a geometric duration, which has its mode at d=1 and cannot suppress short spells while still allowing long ones; this is the likely reason it under-produces spells too (penalty set large to kill flicker).
- Cause of under-production: with a single constant penalty tuned for noise suppression, real 1-3 s possessions are absorbed. An HSMM with an explicit short-duration mode and a per-frame emission strong enough lets them survive. Another lever is an asymmetric penalty (D<->H/A cheap, H<->A moderate). Unverified until tested on IDSSE.
- Own implementation avoids GPL; hsmmlearn GPL-3.0 would taint the repo if vendored. Only pyhsmm (MIT) is licence-clean and it is unmaintained and Gibbs-based, not a clean Viterbi.

### Gaps
- No source found confirming a cumulative-sum recursion specific to log-normal/gamma durations, or concave max-plus convolution speed-ups for HSMM Viterbi.
- Could not read Yu (2010) body text; exact complexity statements not verified from the primary source.
- No benchmark of HSMM vs HMM on segment-F1 for sports/possession data found.

## 2. Temporal action segmentation: smoothing, boundary refinement, over-segmentation control, reported F1@{10,25,50}

### Takeaway
TAS literature fights over-segmentation (our problem is the opposite: under-segmentation) mainly with a training-time smoothing loss, multi-stage refinement and, in DiffAct, boundary-aware generative refinement. Gains on F1@50 from architecture changes are large (e.g. ASFormer 76.0 to DiffAct 83.7 on 50Salads), but these are learned models, not decoders; for our setting the transferable idea is multi-stage refinement and treating segment-level F1 as the objective.

### Cited Findings
- MS-TCN trains with a classification loss plus a proposed smoothing loss that penalises over-segmentation errors, and refines predictions stage by stage; abstract gives no numeric gains — [arXiv 1903.01945](https://arxiv.org/abs/1903.01945)
- Survey: Ding, Sener, Yao, "Temporal Action Segmentation: An Analysis of Modern Techniques" (arXiv Oct 2022, v5 Oct 2023; IEEE TPAMI 2024), covers frame representation, temporal modelling, benchmarks, supervision and metrics; abstract does not mention post-processing or over-segmentation — [arXiv 2210.10352](https://arxiv.org/abs/2210.10352)
- DiffAct (ICCV 2023) 50Salads F1@{10,25,50} = 90.1 / 89.2 / 83.7; Breakfast 80.3 / 75.9 / 64.6. ASFormer baseline 85.1 / 83.4 / 76.0 (50Salads) and 76.0 / 70.6 / 57.4 (Breakfast) — [EffiDiffAct comparison table](https://arxiv.org/pdf/2408.02024). Breakfast DiffAct average is listed as 73.6 on [Papers with Code](https://paperswithcode.com/paper/diffusion-action-segmentation) vs 75.1 in the EffiDiffAct table: discrepancy; sotaverified marks DiffAct claims unverified — [sotaverified](https://sotaverified.org/papers/diffusion-action-segmentation)
- A review reports DiffAct with an MS-TCN backbone gets F1@50 = 79.4 on 50Salads vs 70.1 for MS-TCN++ — [search summary via hyper.ai listing](https://beta.hyper.ai/en/sota/tasks/action-segmentation/benchmark/action-segmentation-on-50-salads-1)

### Inferences
- Common TAS post-processing (from my knowledge, not sourced here): merge segments shorter than a minimum length into neighbours; median/mode filtering; action-boundary refinement via boundary-probability heads (ASRF) with boundary-aware voting inside class segments. These raise F1@k and edit score mostly by removing short false segments. We have the opposite failure, so apply the lesson as "match the decoder's segment-count prior to the true count".
- Practical diagnostic for us: compare predicted vs true segment count and duration histograms on IDSSE; if predicted count is ~57% of true (260 vs 460), recall at all IoU thresholds is capped and F1 cannot exceed about 2·P·R/(P+R) with R<=0.57·(matching rate).
- TAS benchmarks have ~10-30 segments per video with minute-scale video; our ~460 segments over ~110 minutes with 1-3 s spells is a different regime. F1@10/25/50 gains do not transfer numerically.

### Gaps
- Did not retrieve the ASRF/boundary-refinement paper numbers or MS-TCN's own ablation (smoothing loss on/off) numbers.
- No source on post-processing-only gains (no retraining) in TAS.

## 3. Minimum Bayes risk / expected-F1 decoding for IoU or span metrics

### Takeaway
MBR decoding has direct precedent for F-measure: "Algorithms for Minimum Risk Chunking" computes expected F-measure of chunkings efficiently. For IoU-matched spell-F1 no closed-form exists that I found; the practical route is sample-based MBR (FFBS samples, pick the candidate with highest average spell-F1 against samples) using the organiser's metric code.

### Cited Findings
- "Algorithms for Minimum Risk Chunking" frames chunking as precision/recall optimisation, contrasts E-measure (risk) with F-measure (utility), and contributes an algorithm for efficiently evaluating expected F-measure — [CORE PDF (could not fetch, 403; search snippet only)](https://core.ac.uk/download/161444453.pdf)
- Segmental MBR (Goel, Kumar, Byrne) segments N-best lists/lattices to shrink the MBR search space; reports WER gains on Switchboard — [ISCA 2001](https://www.isca-archive.org/eurospeech_2001/goel01_eurospeech.html); [Kumar & Byrne 2002](https://www.isca-archive.org/icslp_2002/kumar02_icslp.html)
- MBR in general picks the output with minimum expected loss; for intractable output spaces it uses Monte-Carlo pseudo-references drawn from the model; epsilon sampling is among the sampling strategies studied — [Epsilon Sampling Rocks](https://arxiv.org/pdf/2305.09860)
- MBR applied to error span detection in MT evaluation, arguing MAP is not always the right objective — [arXiv 2512.07540](https://arxiv.org/pdf/2512.07540v1)
- Arun, Haddow, Koehn use a Gibbs sampler for exact-distribution MBR in MT — [WMT 2010](https://aclanthology.org/www.mt-archive.info/WMT-2010-Arun.pdf)

### Inferences
MBR recipe for our metric (my design, not tested):
```
1. HMM/HSMM forward pass (log-space, scaled).     O(T K^2) (HMM)
2. Draw S samples by FFBS: sample z_T ~ alpha_T, then z_t | z_{t+1} ∝ alpha_t(z_t) A[z_t,z_{t+1}]. O(T K) each, seed fixed.
3. Candidates C = {Viterbi path, posterior-argmax path, each sample, Viterbi at several penalty levels}.
4. Score each c in C: U(c) = (1/S) sum_s spellF1(c, sample_s) using the organiser's possession_metrics (or a fast numpy port).
5. Output argmax U.
```
- Cost: spell-F1 between two paths is O(#spells) after run-length encoding with greedy IoU matching, so S=100 and |C|=100 costs 1e4 metric calls on ~500 spells: seconds to minutes. Deterministic given a fixed RNG seed (use numpy Generator with explicit seed; required by rule #11).
- Pitfall: FFBS samples from an HMM with constant penalties are far noisier at boundaries than the true process; samples overstate boundary jitter, so MBR will favour smoother-than-truth candidates unless posterior is calibrated (temperature-tune the emission log-scores on IDSSE, not BAR-ATH).
- Better: MBR over boundary-jitter-tolerant IoU 0.3-0.7 thresholds is more forgiving than frame accuracy, so MBR gain should come mainly from choosing which short spells to keep. Per-spell inclusion rule from posterior: keep a candidate spell if its posterior probability of being matched at IoU>=0.5 exceeds ~F*/2 (the F1-optimal threshold property: optimal decision threshold of F is half the achievable F*). This is a known result for F-measure optimisation (Lipton et al. 2014 style), not retrieved here; unverified. With F around 0.3 this implies a low keep-threshold of roughly 0.15, which supports producing more segments, consistent with the under-production symptom.
- Do not tune the keep-threshold on BAR-ATH (rule #10); calibrate on IDSSE.

### Gaps
- No paper found on MBR specifically for temporal-IoU matched segment F1.
- Could not read the expected-F-measure chunking paper body (403); formulas not verified.

## 4. Semi-Markov CRFs / segmental models trainable on small data

### Takeaway
Semi-CRFs give discriminative segment-level scoring with exact inference at O(T·Dmax·K²)-ish cost, but need segment features and structured training; with 7 matches, a generative HSMM with few parameters (duration histograms + transition matrix) plus discriminative per-frame emission is the better bias/variance trade.

### Cited Findings
- Sarawagi & Cohen (NIPS 2004): semi-CRFs are conditionally trained semi-Markov chains with labels on segments; exact learning/inference polynomial, often only a small constant factor slower than linear-chain CRFs; beat CRFs on five NER tasks — [NeurIPS abstract](https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)
- Filtered Semi-Markov CRF (2023): semi-CRF cost is quadratic in sequence length and the large solution space can make it underperform a standard CRF; defines segmentations as non-overlapping start/end/label triples — [arXiv 2311.18028](https://arxiv.org/pdf/2311.18028)
- A segmental CRF for chord recognition states Viterbi inference linear in input length (presumably via bounded segment length; not confirmed) — [arXiv 1810.10002](https://arxiv.org/pdf/1810.10002v2)

### Inferences
- With max segment length Dmax, semi-CRF decoding is O(T·Dmax·K²) like the HSMM. Segment features can include: duration, mean ball-speed, mean control-score margin, count of team flips inside, boundary-event evidence (kick, tackle, out-of-play). Weights trained by structured perceptron / L-BFGS on IDSSE with L2 regularisation; train-time cost multiplies by forward-backward.
- Structured SVM / perceptron with a loss augmented by (1 - segment IoU) targets the evaluation directly; unverified benefit, no source found.
- Small-data risk: IDSSE has a modest number of matches (the team has 7 including 2 locked ones per CLAUDE.md; only 5 usable to tune); use leave-one-match-out CV and report spread.

### Gaps
- No semi-CRF Python package with licence verified; I did not search for one. Writing custom NumPy is feasible.
- No evidence of semi-CRF gains on sports possession data.

## 5. Learning transition matrices; feature-driven time-varying transitions (input-output HMM)

### Takeaway
Source coverage is empty here; this section is entirely inference. Counting transitions from IDSSE labels gives the baseline; letting switch probabilities depend on per-frame covariates (ball speed, ball-to-nearest-player distance, ball-in-play-state changes, pitch zone) via logistic regression is cheap, deterministic and implementable in the Viterbi as time-varying logA_t.

### Cited Findings
- None retrieved in this research pass (no reliable source found).

### Inferences
- Counting: with ground-truth frame labels y_t on IDSSE, A[i,j] = n_ij/n_i., but frame-level counts give geometric durations (mean = 1/(1-A_ii)); true durations are heavy-tailed, so use duration histogram per state (HSMM) and a segment-level transition matrix on spell-to-spell changes (H->D, A->D, H->A etc.).
- Input-output HMM / time-varying transitions: P(switch at t | x_t) = sigmoid(w·x_t + b_s), fit by logistic regression on frames labelled "boundary vs not" from IDSSE truth. In Viterbi use penalty_t = -log P(switch_t | x_t) for the cross-state move and -log(1 - P) for the self-loop (cumulative sum trick keeps cost O(TK²) for the HMM case).
- Equivalent, more robust alternative: boundary-evidence signal b_t (peak detector over possession-change cues) added to the transition cost; this is a "feature-driven penalty" and needs only 1-3 tunable scalars, which fits small data.
- Detail that matters: a dead-ball state D has deterministic structure (any H/A spell is preceded and followed by either D or a team flip); enforce with transition masks.
- Risk: features computed from tracking are noisy exactly where possession changes (contested ball), so the learned coefficients may be weak; check with log-likelihood or boundary F1 on held-out IDSSE match, not on BAR-ATH.

### Gaps
- No literature found in this pass on IOHMM-based segmentation of football possession.
- Unverified claim: any gain from time-varying transitions on spell-F1 (needs experiment).

## 6. Change-point detection (PELT, BOCPD) as an alternative

### Takeaway
PELT (ruptures, BSD-2-Clause) and BOCPD can propose candidate boundaries cheaply and deterministically, but they are unsupervised on a score signal and don't know class labels; best used to generate boundary candidates that a constrained decoder labels, or as a second-opinion count of segments.

### Cited Findings
- ruptures is BSD-2-Clause (ENS Paris-Saclay, CNRS); PELT is Killick, Fearnhead & Eckley (2012) "Optimal detection of changepoints with a linear computational cost"; ruptures asks to cite Truong, Oudre & Vayatis (2020); linear average cost under mild conditions on the changepoint distribution, cost model "l2" much faster than others — [cloudsmith ruptures page](https://cloudsmith.com/navigator/pypi/ruptures); [tsseg PELT docs](https://fchavelli.github.io/tsseg/api/tsseg.algorithms.pelt.html); [rupturesRcpp docs](https://www.rdocumentation.org/packages/rupturesRcpp/versions/1.0.2)
- BOCPD (Adams & MacKay 2007): exact online posterior over current run length by message passing; a geometric gap distribution gives constant hazard H(r)=1/lambda; recursion carried out in log-space and renormalised — [arXiv 0710.3742](https://ar5iv.arxiv.org/html/0710.3742); [fast-bocpd docs](https://fast-bocpd.readthedocs.io/en/latest/theory/bayesian_changepoint.html)

### Inferences
- PELT objective: minimise sum of segment costs + beta·(#changepoints). The penalty beta directly controls segment count, and a BIC-like beta = c·log T·sigma² can be calibrated so predicted count matches the IDSSE-typical count per minute. Worst case is O(T²) if pruning fails (my knowledge; the search source did not state it).
- Equivalent to a constant-penalty Viterbi in spirit: PELT with L2 cost on a 1-D margin signal (home-minus-away score) is the same family as the current decoder, so unlikely to fix under-production by itself. It helps only through different cost models (e.g. multivariate mean shift on 3-class probabilities).
- BOCPD with a non-geometric hazard (e.g. discretised log-normal hazard from duration model) is the online analogue of an HSMM and costs O(T·Rmax) with run-length truncation. Look-ahead is allowed (rule #15), so offline HSMM is strictly preferable.
- License: ruptures BSD-2 is compatible; adding it needs a dependency approval from the human per CLAUDE.md. A NumPy PELT is about 30 lines.

### Gaps
- No evidence of PELT/BOCPD used on soccer possession segmentation.
- Worst-case complexity of PELT not confirmed from a source here.

## Cross-cutting recommended experiments (all IDSSE-only, leave-one-match-out, report spread; all inferences)
1. Replace constant-penalty Viterbi with explicit-duration HSMM using empirical duration histograms (Dmax around 99.5th percentile) and a transition mask; measure predicted/true spell count and spellF1 per match. Noise floor ~0.04 on one match per CLAUDE.md.
2. Add MBR selection over candidates using FFBS samples and the organiser's metric; calibrate emission temperature on IDSSE.
3. Per-spell keep-threshold derived from posterior (low threshold, since F is low).
4. Feature-driven switch penalties from a logistic boundary model.
5. Boundary refinement: after segment-level decode at 5 fps, snap boundaries to local maxima of boundary evidence within +-1 s (IoU is tolerant, so this matters mostly for 1-3 s spells where a 0.5 s shift drops IoU below 0.5).
