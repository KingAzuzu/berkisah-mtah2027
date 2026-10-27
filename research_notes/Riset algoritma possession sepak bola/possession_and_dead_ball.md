# Football possession and ball-in-play detection from tracking data (methods, thresholds, accuracies)

Scope note: only public literature and public code docs were used. No competition data was requested or used. Several primary pages (Nature/Springer full text, PMC for Lang et al., Figshare, kloppy source) could not be fetched, so some items are in Gaps.

## Q1. Rule-based possession from tracking: control radius, minimum duration, ball speed / direction-change cues, contested states

### Takeaway
The standard recipe is a three-step rule: (1) nearest player within a possession-zone radius (0.5-1.5 m typical) gives a candidate; (2) the candidate is validated only if the ball shows a kick or touch signature (direction change or speed or acceleration jump); (3) possession persists until the next validated gain, so flight time is credited to the passer. Two or more opponents within 1 m of the ball are flagged as a duel (contested), which overrides possession. Reported accuracy against manual annotation is roughly F 0.83 (Link & Hoernig) and above 90% detection of passes (Vidal-Codina).

### Cited Findings
- Link & Hoernig use 25 Hz xy of players and ball. They remove ball jumps and sections shorter than 1 s, then apply an RTS smoother to derive speed and acceleration. — [Link & Hoernig 2017, PLOS ONE (PMC5503225)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- Individual ball possession (IBP) start, naive version: the player-ball distance is below a threshold T_P and that player is the nearest. The paper trained T_P (and T_K, T_A, T_B) by grid search against manual annotation of 5 matches but does not report the values. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- IBP start, kick-detection version: the naive condition holds with threshold T_K and the smoothed ball acceleration norm has a local maximum of at least 4 m/s^2. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- Individual ball action (IBA) end: project the ball (or the player) 1 s ahead using current velocity. The IBA ends when the player is farther than T_A (or T_B) from the projection for 1 s. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- Link & Hoernig Table 1 (precision/recall/F/"timeline" score): IBP naive R .90, P .68, F .77, timeline .82; IBP kick detection R .80, P .86, F .83, timeline .87; IBA ball prediction R .86, P .91, F .88, timeline .92; IBA player prediction R .84, P .92, F .87, timeline .91. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- Ball-control classification is a Bayesian network (K2 structure learning). Features: IBA duration, mean and variance of ball velocity and acceleration, mean player-ball distance, counts of opponents within 0.5 m, 1 m, up to 5 m. It reports 97.0% of control intervals and 50% of non-control intervals correct, overall accuracy 96.7%, kappa .67, versus human inter-rater kappa .72. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- A conference version reports precision and recall of 80-92% on clean data and about four points lower on uncorrected real data. — [LIDA record via search](https://lida.sport-iat.de/twm/Record/3048516?lng=de)
- Link & Hoernig stated limitations: touching the ball is not required, so a player running alongside is credited late; light touches under the acceleration threshold are missed; nearest-player can be wrong (back to ball); no z coordinate, so an overhead ball can be credited wrongly. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- Vidal-Codina et al. (MIT Sports Lab / FIFA): possession zone (PZ) is a circle round each player. Radius is 50 cm for provider A and 1 m for providers B and C. If the ball is in a player's PZ, that player has possession. — [Vidal-Codina et al. 2022, arXiv 2202.00804 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Duel zone (DZ): radius 1 m around the ball for all providers. A duel is flagged when at least two opponents are inside it, and a duel overrides possession. A control frame is any frame labelled possession or duel. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Vidal-Codina smoothing: Savitzky-Golay filter (order 2, 7-frame window, about 0.25 s at 25 Hz) on ball x and y when noisy. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Vidal-Codina gain/loss logic. Gain: within a control sequence of the same player, either the direction change (d0 . d1 < eps_theta) or a speed change (|v0 - v1| > eps_v on at least one frame) must occur, otherwise the sequence is a false positive and is dropped. Loss: ball outside the player's PZ at f+1 with displacement above eps_s, and the player absent from the next control frame. Numeric values of eps_s, eps_theta, eps_v are in an Online Resource not read here. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- databallpy implements the Vidal-Codina method in `get_individual_player_possession`. Defaults: `pz_radius` 1.5 m, `bv_threshold` 5.0 m/s (minimum ball speed change), `ba_threshold` 10.0 degrees (minimum direction change), `min_frames_pz` 0. The doc walkthrough uses other values in places (2.5 m radius; 1.5 m/s with 10 degrees), so defaults and examples disagree. — [databallpy docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- databallpy step logic: the closest player within `pz_radius` is the candidate; a run of candidate frames is valid if long enough and the ball direction changes by more than `ba_threshold` or speed changes by more than `bv_threshold` in any frame step; the player keeps possession until the last frame before the next valid gain in which the ball is still within `pz_radius`, so frames in between (sprint with ball) go to that player. — [databallpy docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- databallpy team possession (`add_team_possession`) is event-based, not tracking-based: the team of the last on-ball event (pass, dribble, shot) holds possession, and it switches only when the other team has a successful on-ball event. No thresholds. — [databallpy docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- Vidal-Codina overall accuracy against manual events: passes above 90% precision and recall; corners, throw-ins, goal kicks above 90%; shots precision 78% and recall 53% (about 70% of 254 misclassifications were blocked shots labelled as passes); free kicks weakest; runtime about 3 minutes per 90-minute match. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- floodlight treats possession and ball status as codes loaded from provider files; I found no documented estimator in it. — [floodlight getting started](https://floodlight.readthedocs.io/en/latest/_sources/guides/getting_started.rst.txt)
- I did not verify code licences for databallpy, floodlight, kloppy or socceraction in this session (see Gaps).

### Inferences
- The "validation by ball kick signature" step (direction change above about 10 degrees or speed jump above about 1.5-5 m/s, or acceleration above 4 m/s^2) exists to suppress false possession when the ball merely passes near a player. With a 41%-missing ball, these cues are only computable on observed stretches.
- A 1 m two-opponent duel state is a cheap way to avoid committing to a team during scrambles. It could be used as "carry previous label" rather than a new state, which reduces fragmentation (this is our inference, not tested).
- The "flight credited to passer until next gain" rule is exactly what gives long contiguous spells, which is what spellF1 rewards. A nearest-player-only rule produces short flickering runs. This plausibly explains under-production or fragmentation on the current method; it is a hypothesis to test on IDSSE only.
- Because the Link & Hoernig threshold values are not published, thresholds for our data must be tuned on IDSSE, not copied.

### Gaps
- Actual values of Link & Hoernig T_P, T_K, T_A, T_B and Vidal-Codina eps_s, eps_theta, eps_v: not in the main texts read.
- Accuracy of these rule-based methods specifically against DFL BallPossession truth: not found.
- socceraction possession utilities: not researched (socceraction works on event SPADL data; I did not verify it contains a tracking possession estimator).
- kloppy: no possession estimator found; not verified in depth.
- Licences for databallpy, floodlight, kloppy, socceraction: not checked.

## Q2. Learned possession / ball-owner models from tracking

### Takeaway
PathCRF (2026) is the closest published learned model: it infers possession from player trajectories only, as a Viterbi-decoded path over a graph where each state is one edge (self-loop = player in control, directed edge = pass in flight, outside-node self-loop = out of play). On a held-out DFL Sportec match it reaches 92.4% frame-level team-possession accuracy at 5 FPS. Feature-level importance is not reported.

### Cited Findings
- PathCRF models the 22 players plus four "outside" nodes (ball leaving each boundary) as a fully connected dynamic graph. At each time step it selects one edge. Masking forbids impossible transitions, for example a pass must start from the player in control. Decoding is Viterbi; events are detected when the selected edge changes. — [PathCRF, arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Architecture: two stacked socio-temporal encoders (partially permutation-equivariant plus fully permutation-equivariant social module; attention-based temporal module), a Dynamic Masked CRF head, and an auxiliary sender/receiver head. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Data: Sportec Open DFL dataset (the same family as IDSSE), seven matches; 5 train, 1 validation, 1 test; tracking downsampled from 25 to 5 FPS. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Test-match results: edge accuracy 69.64%, sender accuracy 80.44%, receiver accuracy 82.28%, violation rate 0%, event precision 73.18%, recall 78.37%, F1 75.69%; team possession share 62.64% predicted vs 62.13% true; frame-level team-possession accuracy 92.40%. At 25 FPS event F1 is 77.21% but training time per epoch grows more than tenfold. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Ablations (event F1): ball-trajectory post-processing baseline 48.39%, non-CRF 65.34%, static masked CRF 73.83%, dynamic masked CRF 75.69%. The social-module choice gives the largest gain; the temporal-module choice is marginal. No feature-level importance reported. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Code is stated to be at github.com/hyunsungkim-ds/pathcrf; the paper is CC BY 4.0; the code licence was not stated on the page read. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Peral et al. (WACV 2026) use a multi-modal transformer on tactical-camera data for ball trajectory inference, ball state classification and possessor identification, with inputs of player trajectories, player types and image crops, and a masking pre-training scheme (CropDrop). The page read gave no possessor accuracy figures. — [arXiv 2512.19528](https://arxiv.org/abs/2512.19528)
- Link & Hoernig's Bayesian network (Q1) is an earlier learned layer on top of rules; its features (IBA duration, ball speed/acceleration statistics, player-ball distance, opponent counts at 0.5-5 m) indicate which signals were considered useful. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)

### Inferences
- PathCRF's masked-transition design is structurally the same idea as our 3-state Viterbi with switch costs, extended to hidden states of "which player" and "in flight". Adding in-flight and out-of-play states with transition constraints may be a cheap upgrade to our decoder without needing a neural net.
- 92.4% frame accuracy is likely measured with ball-free input on a single test match, so it is an upper-bound-style reference, not a spell-level number. It is not comparable to spellF1.
- A GBDT on hand-crafted features (nearest-player distances per team, ball speed, ball-visible flag, team mean speed, etc.) is a plausible CPU-friendly alternative, but I found no paper reporting such a model's feature importances for possession.

### Gaps
- No feature-importance results found for learned possession models (PathCRF reports only component ablations).
- Gradient-boosting and GNN possession papers beyond PathCRF were not found in this session; unravelsports was not examined.
- PathCRF per-player input features and the code licence are not confirmed.
- Spell-level (possession-run) metrics for any of these methods: not reported.

## Q3. Dead-ball / stoppage / ball-in-play detection from player movement

### Takeaway
Two sources give quantitative evidence. Lang et al. (2022) trained classifiers on player positions from 102 Bundesliga matches and report up to 92% frame accuracy and 81% precision for stoppages longer than 2 s (AdaBoost best). Vidal-Codina et al. use geometric restart patterns (kickoff, goal kick, corner, throw-in, penalty) over dead-ball intervals; these are the concrete positional rules.

### Cited Findings
- Lang et al. 2022 (Scientific Reports): four classifiers trained on spatiotemporal data with manually annotated in-game status of 102 Bundesliga matches; best model AdaBoost reached up to 92% frame-level accuracy on unseen matches and 81% precision for detecting stoppages longer than 2 s. Purpose: compute indicators relative to effective playing time. — [search summary of Lang et al. 2022](https://link.springer.com/10.1038/s41598-022-19948-1) and [DOAJ record](https://doaj.org/article/4899fef7e03b4b3aaf5e0d0f1ccb0e3a)
- Hoernig et al. earlier ball-in-play detection on one top-league match: roughly 78-80% recall and 77-86% precision depending on whether tracking errors were excluded, with a 0.6 s tolerance window. — [search result summary, Hoernig et al. (ISCSS proceedings)](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/978-3-319-24560-7_13.pdf)
- Vidal-Codina dead-ball rule: ball is dead when the provider boolean is off or ball tracking is absent. Tracab supplies an explicit in-play boolean; Track160 and Hawk-Eye treat missing ball data as dead. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Vidal-Codina set-piece triggers over a dead-ball interval: kickoff = all players in own halves plus at least one player near the centre mark; penalty = one player on the goal line between the posts, one opponent near the penalty mark, all others outside the box and beyond 9.15 m; goal kick = at least one player in own goal area; corner = at least one player near an active corner mark; throw-in = at least one player beyond the sideline minus a tolerance; free kick = default when nothing matches. Tolerance values are not in the main text. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Confirmation: a set piece is confirmed when the triggering player is in the pattern zone and the ball is inside that player's PZ on the first in-play frame; with several candidates the closest to the ball is chosen. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- Effective playing time in soccer is typically under 60 min per game. — [Wass et al. 2020 via search summary](https://repository.mdx.ac.uk/item/8888v)
- Link & Hoernig note running-flag errors and tracking losses as main error sources; the running flag is used only at the final interval calculation. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- PathCRF has an explicit "out of play" state (self-loop on an outside node) in its path, so out-of-bounds is part of the same decoding as possession. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)

### Inferences
- Restart shapes are detectable from player positions alone and could be used as a dead-ball prior in our Viterbi: players in own halves with one at centre (kickoff), a player in own goal area (goal kick), a player at a corner mark, a player at the touchline (throw-in). These could be used as emission features for D, tuned on IDSSE.
- The Lang et al. results (positions-only, 92% frame accuracy) suggest a supervised positions-only D classifier trained on IDSSE (truth: DFL BallStatus) is feasible on CPU. Details of its features were not retrieved, so what features carry the signal is unknown from this research.
- Because DFL data have about 45% dead frames, a classifier's long-stoppage precision (81% for over 2 s) matters more than short-gap detection; short stoppages are what produce fragmentation either way.

### Gaps
- Full text of Lang et al. not retrieved (Springer/Nature and PMC blocked): features, feature importance, per-class numbers and frame rate not confirmed.
- No paper found that quantifies collective-speed or team-shape thresholds for dead ball specifically; I found no source for "players' mean speed below X m/s means dead".
- The numeric tolerances for the Vidal-Codina set-piece rules are in the Online Resource, not read.

## Q4. Handling missing ball observations

### Takeaway
Published methods mostly do not impute. Vidal-Codina treat missing ball as dead for providers lacking a boolean; Link & Hoernig drop short or implausible ball sections; PathCRF avoids the problem by not using the ball at all.

### Cited Findings
- Link & Hoernig detect trajectory jumps with adjacent-point distance tests, drop sections shorter than 1 s, smooth with RTS. No imputation method is described. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- Vidal-Codina: Track160 and Hawk-Eye treat missing ball data as dead ball; missing or inaccurate ball data is named a key limitation; player swaps and wrong in-play flags named as error sources; no imputation described. — [ar5iv 2202.00804](https://ar5iv.labs.arxiv.org/html/2202.00804)
- PathCRF: ball never tracked; possession path is inferred from player trajectories alone. Ball-trajectory post-processing baseline reaches event F1 48.39% against 75.69% for the full model. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- Peral et al. infer ball trajectory and ball state from player trajectories and crops, so the ball is reconstructed rather than observed. — [arXiv 2512.19528](https://arxiv.org/abs/2512.19528)

### Inferences
- The PathCRF comparison suggests player-trajectory-driven inference beats ball-trajectory post-processing, which supports using player-based evidence (who is converging on the ball location, team shape) when the ball is missing instead of carrying nearest-player logic only on observed frames.
- Treating "ball missing" as dead (Vidal-Codina) would be wrong here: about 41% of frames lack the ball while about 45% are dead in DFL, so missing is not equivalent to dead in our data. This is an inference from the numbers in the task statement, not from a paper.

### Gaps
- No paper found with a quantified comparison of imputation strategies for missing ball in possession detection.
- Interpolation limits (maximum gap length) used in the literature were not found.

## Q5. Possession during passes / ball in flight, and how DFL defines BallPossession and BallStatus

### Takeaway
Rule-based methods credit the whole interval up to the next player's validated gain to the passer; PathCRF models flight as a distinct "directed edge" state. I could not find the DFL definition of BallPossession or BallStatus in any page I could fetch.

### Cited Findings
- Link & Hoernig: possession switches between players and the whole interval up to the next player's IBP is credited to the passer, on the assumption that the pass reflects intent. Flight time is therefore not a gap in possession; IBAs can still be separated by void phases (ball in the air), which count toward team playmaking time. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225)
- databallpy: frames between valid gains stay with the last possessor (sprint with ball case). — [databallpy docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- PathCRF: a directed edge between two players is a kick in flight, a self-loop is control; flight is therefore attributed to a sender-receiver pair rather than a single player. — [arXiv 2602.12080](https://arxiv.org/html/2602.12080)
- The IDSSE dataset (Bassek, Rein, Weber, Memmert 2025, Scientific Data, DOI 10.1038/s41597-025-04505-y) covers seven Bundesliga matches (two 1. Bundesliga, five 2. Bundesliga, 2022/23 per the DSHS announcement), 25 fps TRACAB optical tracking from Sportec Solutions, CC BY 4.0, loadable with kloppy. — [Hugging Face IDSSE README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md) and [DSHS announcement](https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/detail/meldung/official-game-data-in-the-journal-scientific-data)
- The Hugging Face dataset card does not define BallPossession or BallStatus. — [Hugging Face IDSSE README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- The Common Data Format paper defines a generic ball_status field and a possession team identifier, but that is a standard, not the DFL schema. — [CDF, arXiv 2505.15820](https://arxiv.org/pdf/2505.15820)

### Inferences
- The task statement says truth is DFL `BallStatus` (0 = dead) and `BallPossession` (1 = home, 2 = away). Since DFL possession is a per-frame code, it is probably assigned through flight as with the passer-credit convention, but this is not confirmed by any source I read. Label behaviour during flight and after loose balls should be checked on IDSSE itself, with aggregate statistics only.
- If DFL possession follows the last-touch convention, then a decoder that holds the last team through flight (not "nearest player") matches the truth better and gives longer spells.

### Gaps
- DFL definitions of BallPossession and BallStatus (who annotates, whether manual or algorithmic, how contested or loose-ball frames are coded): not found. Check the Scientific Data paper's data-description section and the Figshare record (10.6084/m9.figshare.28196177); Nature and Figshare were not fetchable here.
- How kloppy maps these fields to ball_state and ball_owning_team: not confirmed (source URL returned 404).
- Whether the Sportec open tracking ball-missing frames carry a dead status: not found.
