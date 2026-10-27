# Ball imputation and ball-owner inference from player trajectories

Scope: methods that infer or impute the ball position (and possessor) from player trajectories. Notes are based on web fetches done 2026-10-27. Items I could not verify are in Gaps. I never touched competition data.

## Ball Radar (Kim et al., KDD 2023): what it predicts, accuracy, code, licence, anonymity, partial observations

### Takeaway
Ball Radar predicts per-timestep ball (x, y) plus a per-player possession probability from the 22 players' trajectories. Its set-based design needs no player identities. The main model uses no ball observations, but a masking variant conditions on partial ball observations and drops error from about 5 m to about 1.3 m at 80% masked. Public code exists (train from scratch, no weights) but the repo has no licence.

### Cited Findings
- Authors: Kim, Choi, Kim, Yoon, Ko; KDD 2023; arXiv 2306.08206. It infers ball trajectory from player trajectories as a cheaper alternative to ball tracking. Set Transformers give order-invariant or equivariant context. A hierarchical design first estimates which player holds the ball, then regresses the trajectory. A "reality loss" and rule-based postprocessing make trajectories physically plausible. Authors list missing-trajectory imputation as an application. — [arXiv](https://arxiv.org/abs/2306.08206v1), [search summary](https://liner.com/review/ball-trajectory-inference-from-multiagent-sports-contexts-using-set-transformer)
- Two parts: Ball Possessor Classifier (set-transformer encoders feeding shared-weight player-wise Bi-LSTMs, output per-player possession probability) and Ball Trajectory Regressor (per-team Set Transformers, then a Bi-LSTM outputting ball x, y). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Inputs: 22 players x 6 features (x, y, vx, vy, speed, acceleration). Ball out of play is encoded as four pitch-line midpoints. Goalkeepers missing from GPS data were predicted by a separate model (5.4 m error on Metrica). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Data: 15 K League 2020 matches of Fitogether GPS (10 Hz, not public) plus 3 Metrica sample games (25 Hz downsampled to 10 Hz). Metrica split: games 1 and 2 train, game 3 first half validation, second half test (about 21k test frames). Windows are 10 s (100 steps), with random pitch flips. Fitogether ground-truth ball is reconstructed from manual events plus GPS using rules, not measured. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Architecture size: Bi-LSTM 2 layers, 256 hidden, dropout 0.2; Adam lr 5e-4; full model has 8,699,846 parameters. Runtime is not reported. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Main result (test, no ball observations given): H-LSTM-RL position error 3.66 m, player possession accuracy 64.70%, team possession accuracy 85.85%. After rule-based postprocessing (H-LSTM-RL-PP) error rises to 4.07 m, team possession 85.34%. Plain LSTM 5.47 m, Transformer 7.47 m, VRNN 12.37 m, hierarchical Transformer about 4.9 m. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Ablations: location only 4.65 m / team acc 82.04%; location + velocity 3.92 m / 84.90%; all six features 3.66 m / 85.85%. Naive ordering of players (by shirt number) gives 7.42 m and 19.72% player accuracy, versus about 3.7 m with the set encoders. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Partial ball observation (Section 5.1, random masking of the true ball track): masked 100%: 5.35 m / player acc 61.9% (before postprocessing); 95%: 3.00 m / 79.5%; 90%: 2.09 m / 87.1%; 80%: 1.31 m / 93.1%. After postprocessing 80%: 1.41 m / 91.3%. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206). A separate search summary says the model was trained by masking known ball positions at random so reliable detections act as anchors. — [search result summary](https://arxiv.org/pdf/2306.08206)
- Permutation handling: set-based models do not depend on player identity or order; only the naive baseline uses shirt-number ordering. Team membership is used (per-team set transformers). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Code: github.com/pientist/ballradar, Python 3.10, PyTorch 1.13.0. Training via `sh scripts/player_ball.sh`; evaluation via `model_eval.ipynb`. Preprocessed Metrica events are in `data/metrica_events/`; Metrica tracking must be downloaded separately. No pretrained weights are mentioned. The Fitogether data is not shared. The README says the Metrica-only pipeline loses little performance. — [GitHub](https://github.com/pientist/ballradar)
- Licence: GitHub API reports `license: null` (no licence), 34 stars, last push 2026-02-03. — [GitHub API](https://api.github.com/repos/pientist/ballradar). Without a licence, default copyright applies, so reuse of code is legally unclear. Reimplementing from the paper is safer.
- Independent re-evaluation: in TranSPORTmer's soccer benchmark (LaLiga), ballradar scored 3.89 m ADE for full inference and 0.88 m at 80% imputation (5.43 m for the variant without possessor classification, 0.97 m at 80%). — [TranSPORTmer arXiv HTML](https://arxiv.org/html/2410.17785v2)

### Inferences
- With team-labelled anonymous tracklets, the set encoders suit our setting. Per-team grouping is needed, which we have.
- Our missing ball is not random: it is missing in crowded moments, when inference is hardest. Random-mask results (1.3 to 3 m) are probably optimistic for us. The 3.7 to 5.4 m full-inference figures are a better guide for long gaps.
- For H/A/D we need only team possession, not exact position. Team possession accuracy of about 85% (no ball at all) is the relevant number. That is frame-level on 10 Hz Metrica/K League data, not our metric (spellF1).
- Training on IDSSE is plausible (full tracking with ball and 25 fps; downsample to 10 or 12.5 Hz or retrain at 25 Hz). We would need to compute player-level possession labels ourselves from the ball-nearest-player rule or events. That is an assumption, not tested.
- Tracking noise: Ball Radar was trained on clean optical or GPS tracks. Our noisy tracklets with ID switches are a domain gap that I found no evidence on.

### Gaps
- Exact licence terms: none is published, and I did not check for a separate LICENSE file.
- Inference runtime, not reported. See the CPU question below.
- I could not read the original PDF (binary), so numbers come from ar5iv (HTML) via a summarising model. Cross-check key numbers in the paper before citing.
- The ar5iv summary says its Table 5 is a zoom-in ROI result, not "Metrica-only". The README claim about Metrica-only performance refers to the paper (the user-supplied pointer to Table 5 is not confirmed).

## Other works: Amirli and Alemdar 2022, Capellera/Peral, TranSPORTmer, NRTSI, Graph Imputer, heuristics

### Takeaway
TranSPORTmer (about 2.7 m inference, 80.8% state accuracy) and the newer Peral/Capellera WACV 2026 paper are the current state of the art but have no confirmed public code. Graph Imputer and NRTSI are general imputation models, with ball support unclear. Older methods are much worse (5 to 8 m MAE).

### Cited Findings
- Amirli and Alemdar 2022, "Prediction of the Ball Location on the 2D Plane in Football Using Optical Tracking Data" (Academic Platform Journal of Engineering and Smart Systems): neural networks on optical tracking from 300 Turkish Super League 2017-18 matches, using speed and positional distribution features. Test R2 79% (x), 92% (y); MAE 7.56 m (x), 5.01 m (y). Not possessor-based. — [open.metu.edu.tr](https://open.metu.edu.tr/bitstream/handle/11511/102180/10.21541-apjess.1060725-2203586.pdf), [DergiPark](https://dergipark.org.tr/en/pub/apjess/article/1060725)
- TranSPORTmer (Capellera et al., arXiv 2410.17785): unified Set-Attention-Block transformer for forecasting, imputation, ball inference and ball imputation, with a CLS agent classifying per-frame state (pass, possession, uncontrolled, out of play). Settings: 128-dim embeddings, 16 heads, 512 hidden. Trained on LaLiga 2022-23 optical tracking, 283 matches, 6.25 Hz, 9.6 s sequences (proprietary), plus basketball sets. — [arXiv HTML](https://arxiv.org/html/2410.17785v2)
- TranSPORTmer ball results (soccer, ADE): full inference 2.71 m (max error 7.39 m) vs ballradar 3.89 m; state classification accuracy 80.84% for ball inference (81.64% on the original dataset). "Uncontrolled" is the weakest class. Imputation (80% hidden) ballradar 0.88 m; TranSPORTmer's own 80% number was not readable. Soccer forecasting inference about 88 ms for TranSPORTmer vs 340 ms for a standard Transformer baseline (hardware not stated; training on an RTX A6000). The arXiv page is CC BY-NC-SA 4.0 (applies to the paper). No code link on the page. — [arXiv HTML](https://arxiv.org/html/2410.17785v2)
- Peral, Capellera, Ferraz, Rubio, Agudo, "Multi-Modal Soccer Scene Analysis with Masked Pre-Training" (WACV 2026, arXiv 2512.19528): ball trajectory inference, ball state classification and possessor identification from player trajectories, player types and player image crops (CropDrop pre-training). Ball inference needs no past or future ball positions. Reported "substantial improvements over state-of-the-art" but the abstract page gives no numbers, dataset name or code link. Needs image crops, so not usable for us with tracking alone. — [arXiv](https://arxiv.org/abs/2512.19528), [WACV](https://openaccess.thecvf.com/content/WACV2026/html/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.html)
- Graph Imputer (Omidshafiei et al., DeepMind and Liverpool FC; arXiv 2106.04219; Sci. Reports 2022 as "Multiagent off-screen behavior prediction in football"): bidirectional autoregressive LSTMs plus graph networks plus VRNN-style latents, forward and backward estimates fused by averaging or by distance to nearest real observation. Graph of 23 nodes (22 players and ball). Trained on 105 full-length real matches (proprietary). It targets players occluded off-camera, evaluated with simulated camera occlusion; baselines include Social LSTM and GVRNN. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.04219), [search summary](https://arxiv.org/abs/2106.04219)
- NRTSI (Shan, Li, Oliva, ICASSP 2023): non-recurrent time-series imputation, treating the series as a permutation-equivariant set with a Transformer, filling gaps hierarchically by distance from observations. Official code github.com/lupalab/nrtsi (no licence; last push 2023-06-02). It is a general time-series model, not football-specific. — [GitHub](https://github.com/lupalab/nrtsi), [GitHub API](https://api.github.com/repos/lupalab/nrtsi)
- Interpolation baselines: a 2023 Italian Journal of Applied Statistics paper (Kontos and Karlis) compares interpolation methods for gaps in broadcast tracking. An older multi-camera system modelled ball phases (rolling, flying, possessed, out of play) with spatio-temporal constraints. A 2004 ICIP geometric approach had over 90% of estimates within 2.5 m of ground truth. — [search results](https://oajournals.fupress.net/index.php/ijas/article/view/15707)
- Ball Radar paper's own ablation: linear-style baselines with no possessor stage (plain LSTM 5.47 m) are worse than hierarchical models. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)

### Inferences
- Learned possessor-aware models (3 to 4 m, no ball) beat non-possession models (5 to 8 m). Possessor modelling is the main gain.
- TranSPORTmer's per-frame state head (including out of play) is the closest literature match to our H/A/D target, but I found no released code or weights, so it would need reimplementation. Its data was 283 proprietary matches; IDSSE has 7, so expect lower accuracy.
- Simple heuristics for our pipeline (not in the literature I found, an inference): interpolate the ball across short gaps between observed frames, with possession held across gaps when the same team's nearest player is on both sides; use a learned model only for long gaps. I have no number for this.
- Camera centre heuristic: I found no source on using the camera centre as ball proxy. Not evaluated.
- NRTSI and Graph Imputer are not ball-specific; there is no evidence of ball results in what I read.

### Gaps
- No public code found for TranSPORTmer, Peral et al., or Graph Imputer (a GitHub search for TranSPORTmer returned nothing; Graph Imputer's page text I read did not mention code).
- Graph Imputer metre errors and runtime were not in the text I could read (the page was truncated at 100k of 154k characters, and the PMC page was blocked by a bot check).
- Peral et al. dataset and numbers not found.
- I did not locate a Kontos and Karlis ball-gap RMSE.
- No source found for a camera-centre ball proxy or for a purely heuristic imputation benchmark on soccer.

## Conditioning on partial ball observations (imputation vs pure inference)

### Takeaway
Yes. Ball Radar was evaluated with random masking and shows a steep gain with partial observations; TranSPORTmer also supports ball imputation. In both cases the masks were random, not occlusion-correlated.

### Cited Findings
- Ball Radar masked-ball results: 100% masked 5.35 m; 95% 3.00 m; 90% 2.09 m; 80% 1.31 m; possessor accuracy rises from 61.9% to 93.1%. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- Pass annotation F1 rises from 0.39 (100% masked) to 0.87 (80% masked). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- TranSPORTmer has ball inference and ball imputation as distinct tasks, evaluated by ADE in metres. — [arXiv HTML](https://arxiv.org/html/2410.17785v2)
- Graph Imputer uses forward and backward passes fused by weighting each direction by distance to the nearest real observation, which is a native partial-observation design. It was evaluated on player occlusion. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.04219)
- NRTSI fills gaps hierarchically from the nearest observed points. — [GitHub](https://github.com/lupalab/nrtsi)

### Inferences
- Our ~41% missing, clustered in crowded moments, sits between the 80% and 100% masked regimes only in fraction, not in difficulty. Training should mimic our real missingness pattern (long, clustered gaps), which we can get from IDSSE by dropping ball frames in chunks, or by copying gap-length statistics from public data only.
- Observed ball frames before and after a gap are strong anchors for the possessor, so an imputation framing is likely much better than pure inference. Look-ahead is allowed per the rules (issue #15 in CLAUDE.md).

### Gaps
- No paper found that tests masks correlated with crowding or occlusion.
- Performance on noisy automatic tracking (ours) is not reported by any source I found.

## CPU inference speed for about 160k frames, data and licence constraints

### Takeaway
No source reports CPU throughput. By parameter count, Ball Radar (8.7M params) should be feasible on CPU, but the number below is my estimate, not a measurement.

### Cited Findings
- Ball Radar: 8,699,846 parameters; 10-second windows of 100 steps at 10 Hz; runtime not reported. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206)
- TranSPORTmer: 88 ms per soccer forecasting batch/sample on GPU-class hardware per the paper (hardware not stated for inference); 16 heads, 128-dim. — [arXiv HTML](https://arxiv.org/html/2410.17785v2)
- Ball Radar requires PyTorch 1.13 and Python 3.10 (pinnable for byte-exact reproducibility, though that older PyTorch may constrain the Windows environment). — [GitHub](https://github.com/pientist/ballradar)
- Metrica sample games used by Ball Radar are public (3 games); IDSSE is CC BY 4.0 (per project instructions). — [GitHub](https://github.com/pientist/ballradar)

### Inferences
- Rough estimate (unmeasured): 160k frames at 25 fps is 6,400 s. At 10 Hz that is 64,000 steps, about 640 non-overlapping 10 s windows (more with overlap, up to 64k if slid by 0.1 s as in training). A 8.7M-param LSTM + set-transformer on 22 players over 100 steps should take on the order of tens of milliseconds to a few hundred milliseconds per window on a modern CPU. Non-overlapping or 50% overlap windows would take minutes in total; fully overlapping stride-1 windows would be much slower (hours at worst) and should be avoided. Benchmark before committing.
- Determinism: CPU inference with fixed seeds and pinned dependencies, plus committed weights, satisfies the byte-for-byte rule. Averaging overlapping windows must use a fixed order.
- Licensing for our use: Ball Radar code has no licence, so we should reimplement rather than vendor it (ask the human). Cite the paper. IDSSE is CC BY 4.0 and must be cited. Metrica sample data licence was not verified here.
- Adding any dependency (for example PyTorch if not already pinned) needs human approval per CLAUDE.md.

### Gaps
- No measured CPU latency for any model; needs a benchmark on our machine using IDSSE.
- Metrica sample data licence not checked.
- IDSSE licence taken from project instructions, not re-verified here.
