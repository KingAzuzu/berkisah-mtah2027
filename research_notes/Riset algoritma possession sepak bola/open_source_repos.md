# Open-source repos and public material on football possession inference (MTAH 2027 and related)

Research date 2026-10-27. Method: WebFetch of GitHub pages, which a small model summarises, so details are second-hand. Treat as leads, not verified code reads. No competition data or submission CSVs were requested or read.

## What do repos by danieltyukov, alperglcoglu, PLOG and other public MTAH entries implement?

### Takeaway
I found no public MTAH possession implementation by danieltyukov, alperglcoglu or "PLOG". The only public method description is the organiser baseline (nearest player to ball, then window majority vote). The 25 active forks I could see look like unmodified clones (no descriptions, no stars to speak of), and PriyaanGada/off-ball is a different task (defender ghost model).

### Cited Findings
- The organiser repo (MIT code and docs; match data licensed separately for the competition only) has 137 forks, 157 stars. Its four baselines are "the same algorithm four times": nearest tracked player to the ball, then a majority vote over a window; only the window changes — [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
- Reported public 5-minute ground-truth scores (spellF1 / macroF1): smooth125 0.4242 / 0.5812; smooth50 0.3256 / 0.6207; smooth25 0.2868 / 0.6200; raw 0.0378 / 0.5972. The README says spellF1 moves about elevenfold across baselines while macroF1 barely changes. The bar to beat is spellF1 0.2433 on the full private match. The exact window sizes (125/50/25 frames) are inferred from the names, and the code is in `starter/baseline.py` (not read) — [same repo](https://github.com/metrica-sports/more-than-a-hack-2027)
- The README says the ball is invisible on 41% of scored frames — [same repo](https://github.com/metrica-sports/more-than-a-hack-2027)
- The profile pages of danieltyukov (pinned: studocuhack, tue-notes, marine-debris-ml-model, etc.) and alperglcoglu (orkestr, Advanced-RAG, book recommender, TinyML, STM32, HomeMate) show no football repos; only top or featured repos were visible, so a hidden one cannot be excluded — [danieltyukov](https://github.com/danieltyukov), [alperglcoglu](https://github.com/alperglcoglu)
- PriyaanGada's only public repo is off-ball. It is a GRU "ghost defender" model (inputs: ball and 11 attackers over 2 s; role embedding; 3 seeds) tested on six A-League matches, credited to SkillCorner open data (MIT) and Metrica sample data. Mean distance to the real defender: baseline 10.50 m, network 6.03 m. It has no possession model and its licence was not stated on the page — [PriyaanGada/off-ball](https://github.com/PriyaanGada/off-ball), [profile](https://github.com/PriyaanGada)
- Active forks seen (page 1 of 2): Tejeshhh78/MesQueUnCode, sergicasesalonso/Pompeu-Data-Team, PranavVijay08/FCFlickBall, VARcelona-MTAH2027, SiamRahman29/road-to-barca, GTT-TalentArena (11 open issues, appears to be a Talent Arena organiser or related account), and about 18 others named more-than-a-hack-2027. Fetching MesQueUnCode, Pompeu-Data-Team and FCFlickBall returned only the upstream README text, so I could not see any team-specific algorithm — [forks list](https://github.com/metrica-sports/more-than-a-hack-2027/forks?include=active&page=1)
- Searches for "PLOG" and for MTAH-specific entries returned nothing relevant — search results, no URL worth citing.

### Inferences
- Any public entry that scores well is probably either private or lives in a fork's branch. The baseline spellF1 pattern (longer window helps spells enormously) suggests spell-level scoring rewards stable, long segments; this matches our decoding focus.
- Our notes naming those three authors probably came from a source other than public GitHub repos (Discord, Slack or private repos).

### Gaps
- Contents of `starter/baseline.py`, other forks' branches and page 2 of the forks list were not inspected.
- "PLOG" could not be identified at all.
- Not searched: Kaggle (no tool route found), earlier MTAH editions' repos.

## Issues on the organiser repo that clarify rules

### Takeaway
All 10 listed issues are closed, and the organiser replies (user jcnunezmoreno) settle the rules that matter to us. They match CLAUDE.md. There is no Discussions tab.

### Cited Findings
- #16 (Alethic Labs): using public IDSSE (CC BY 4.0) tracking to fit a team-shape prior is "fully permissible" if it is other games only, openly hosted and inspectable (training script plus committed weights). It must be cited in the tech doc and README. Hyperparameters and thresholds are chosen on IDSSE or other games, and the 5 public minutes only check the pipeline, not tune it — [issue 16](https://github.com/metrica-sports/more-than-a-hack-2027/issues/16)
- #15: the task is post-match, so look-ahead is allowed (labelling 08:47 with frames from 09:01 is fine). A causal approach is also welcome and scored the same, but should be declared — [issue 15](https://github.com/metrica-sports/more-than-a-hack-2027/issues/15)
- #14: committing intermediate model outputs (e.g. audio-event or speech-to-text outputs as event times and scores) is allowed with the script, model version and settings. Watching video to understand the data and turning observations into rules is allowed, but turning them into labels, corrections or constants tied to specific game frames is not — [issue 14](https://github.com/metrica-sports/more-than-a-hack-2027/issues/14)
- #11: training need not be exactly reproducible, but the CSV must be byte-identical from published weights. Recommended: commit weights, CPU inference, pinned dependencies, fixed seeds. Hardware, OS and time limit were not answered — [issue 11](https://github.com/metrica-sports/more-than-a-hack-2027/issues/11)
- #10 (Team SyntaxEror): hand labels on BAR-ATH may not be used for training, validation or parameter choice, even if never a pipeline input. The suggestion is to annotate a different match, tune there, and apply values unchanged, documenting it (and sharing video and annotations with organisers if rights allow). The team withdrew to pre-labelling values — [issue 10](https://github.com/metrica-sports/more-than-a-hack-2027/issues/10)
- #8: the limit concerns which match is labelled, not the amount. No hand labels on BAR-ATH; no cap on other matches, but external labels must be accessible and described (what, how, by whom, how much), with rights confirmed — [issue 8](https://github.com/metrica-sports/more-than-a-hack-2027/issues/8)
- #2 (SyntaxEror) concerns detector weights trained on the match video and a GPU requirement; #3, #4, #5 are download and access issues. Replies for these were not read (only titles seen) — [issues list](https://github.com/metrica-sports/more-than-a-hack-2027/issues?q=is%3Aissue)
- Side fact: SyntaxEror's issue says its rule-based model has about a dozen numeric settings and uses only provided tracking and Smart Tagging — [issue 10](https://github.com/metrica-sports/more-than-a-hack-2027/issues/10). Alethic Labs plans a logistic-regression "team shape prior" (Stage S4) trained on IDSSE with kloppy — [issue 16](https://github.com/metrica-sports/more-than-a-hack-2027/issues/16). #14's author describes ball-tracking dropout rules from video and audio event classification as ideas. These are what rivals say they do, not code.

### Inferences
- Ideas rivals mention that we may not use: a learned team-shape prior from IDSSE to infer possession when the ball is missing; audio/video-derived event times (committed as intermediates); rules for why ball tracking drops out.
- The #14 limit means any video-derived rule must have constants tuned on other games.

### Gaps
- Replies to #2, #3, #4, #5 not read. Could not confirm whether the cached issue list was complete (numbers 1, 6, 7, 9, 12, 13 are absent, likely pull requests or deleted).

## Possession and ball-status utilities in libraries

### Takeaway
Libraries offer little beyond what we have. databallpy computes team possession from synchronised event data and individual player possession from tracking (Vidal-Codina 2022); PathCRF is the one relevant modelling approach (tracking-only possession-state decoding with a CRF). None gives reported spell-level scores.

### Cited Findings
- databallpy (MIT): team possession is computed from synchronised event data; individual player possession is computed from tracking following Vidal-Codina et al. (2022). The README gives no thresholds or ball alive/dead logic; the docs page I tried returned 404 — [alek050/databallpy](https://github.com/alek050/databallpy)
- Vidal-Codina et al. 2022 (Sports Engineering, MIT Sports Lab with FIFA): a possession step evaluates which player has the ball per frame and also uses player configurations while the ball is out of play to inform set-piece detection; reports above 90% detection rate for most event categories vs manual annotation across tournaments and providers. Methods detail not read — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- PathCRF (KDD'26 per search summary; code under MPL-2.0): ball-free event detection. Each frame is a 26-node graph (22 players plus 4 out-of-bounds nodes); the possession state is one edge (self-loop = control, sender to receiver = flight); set-attention backbone; dynamic masked CRF with impossible transitions masked; Viterbi decoding; events where the edge changes. Input is tracking only at 5 FPS in 10 s windows. Test match on Sportec/DFL open data (7 matches: 5 train, 1 val, 1 test): edge accuracy 69.64%, event F1 75.69%, predicted home possession share 62.64% vs 62.13% true; full match inference under a minute. Pretrained weights in `saved/`; uses kloppy — [arXiv 2602.12080](https://arxiv.org/html/2602.12080), [hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf)
- floodlight (MIT): lists "ball possession" as a Codes data object read from provider feeds (Tracab, DFL); no possession-from-tracking or ball-in-play inference described on the page — [floodlight-sports/floodlight](https://github.com/floodlight-sports/floodlight)
- Other possession-related results (video based, not usable for us): tryolabs/soccer-video-analytics (author calls its ball detector a toy model), AhmedQassemDev2004/computer-vision-football-analysis (YOLO pipeline) — search listing: [search result](https://github.com/alek050/databallpy)

### Inferences
- PathCRF's idea of decoding a possession-state sequence with a transition-masked model is the same shape as HMM/Viterbi decoding with forbidden transitions (e.g. no instant H to A through dead ball without a transition). It could be reused with IDSSE for training, but needs ball-free generalisation to Metrica ATD and has 70% edge accuracy, so it is a reference rather than a drop-in.
- Using IDSSE `BallStatus` and `BallPossession` as labels (as our CLAUDE.md does) is consistent with the DFL dataset PathCRF trains on.

### Gaps
- Not examined: kloppy, unravelsports, socceraction, metrica-sports/sample-data and Laurie Shaw's LaurieOnTracking/Friends-of-Tracking code (search returned nothing useful; I did not confirm the repo URL or licence). Not examined: databallpy source for thresholds, Kaggle notebooks, the last 30k characters of the PathCRF paper (appendix and limitations).
- No source reported spellF1-like metrics for any open library.
