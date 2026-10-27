# Ball action spotting and event detection models for football video and tracking

## Q1. SoccerNet Ball Action Spotting 2023/2024 and Team Ball Action Spotting 2025: classes, winning methods, checkpoints, licences

### Takeaway
T-DEED (Xarles et al.) won BAS 2024 (73.39 mAP@1) and is the base of the 2025 team baseline (51.72 Team-mAP@1 on challenge split). The 2025 winner (dudek, 60.03) has no confirmed public checkpoint. The code is GPL-3.0 and checkpoints are on Google Drive. The video data is NDA-gated, and the models were trained on English-league broadcast footage.

### Cited Findings
- BAS 2024 data: 77 English Football League games, 720p, 12 ball-action classes (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal). 22 further games held out for the challenge set. fps not stated in the report (one participant mentions 25 fps). 88 teams, 61 submissions. — [SoccerNet 2024 results](https://arxiv.org/html/2409.10587); [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)
- BAS 2024 leaderboard mAP@1: T-DEED 73.39, UniBw Munich VIS 71.35, FS-TAHAKOM 67.09, MobiusLabs 62.53, AI4Sports 62.44. Baseline 56.15 mAP@1. — [SoccerNet 2024 results](https://arxiv.org/html/2409.10587)
- T-DEED 2024 winning entry: 2D backbone with Gate-Shift Fuse modules, then a temporal encoder-decoder refining per-frame tokens. Trained jointly on SoccerNet Ball and original SoccerNet (two heads). Final result is an ensemble of two models with different clip sampling. — [SoccerNet 2024 results](https://arxiv.org/html/2409.10587)
- T-DEED repo (arturxe2/T-DEED): PyTorch, GPL-3.0 (plus a separate "LICENSE E2E-Spot" file). Supports SoccerNet AS and BAS. Checkpoints via Google Drive, one per model. SoccerNetBall example runs at 796x448 with inference threshold 0.3. Repo gives no hardware or inference-cost figures. — [T-DEED GitHub](https://github.com/arturxe2/T-DEED)
- T-DEED paper claims state of the art on FigureSkating and FineDiving. Code is public. — [arXiv 2404.05392](https://arxiv.org/abs/2404.05392)
- Team Ball Action Spotting 2025 (sn-teamspotting): same 12 actions, each with a team side (left/right), giving 24 action-team combinations. Metric is Team-mAP@1 (1 s tolerance). Baseline is T-DEED plus a sigmoid team head trained with BCE. — [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting)
- Baseline results: test split Team-mAP@1 47.18 and mAP@1 53.59. Challenge split Team-mAP@1 51.72 and mAP@1 58.38. Repo is GPL-3.0 and a baseline checkpoint is on Google Drive. — [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting)
- 2025 leaderboard (16 teams, 67 submissions): dudek 60.03, Intellindust-AI-Lab 56.78, KIST-BAS 56.08, kistimrcteam 55.98, FSI-Tahakom 55.51, liujian 55.41, UniBW Munich VIS 55.36. UniBW Munich VIS had the best plain mAP@1 (68.92). Some teams were strong team-agnostic but weak on team attribution. — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1); [search summary](https://arxiv.org/abs/2508.19182)
- 2025 winner (dudekTBAS-1, Kamil Czarnogorski): T-DEED re-implemented in the open-source dude.k framework. One joint team-action head replaces the dual heads. Augmentations are horizontal mirroring with label swap, random crops, brightness jitter and synthetic camera-pan transforms. Two-phase training: pretrain on 500+ broadcast matches, then fine-tune on the challenge set. Bayesian hyperparameter search (LR, focal-loss weights, anchor scales, soft-NMS). — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)
- Other 2025 approaches: unified action-team classes with a post-hoc "team alignment" step that fixes team labels using related prior actions (TBAS-2); dual frozen T-DEED backbones with and without team labels (TBAS-5); per-frame team-in-possession prediction and temporal-consistency mechanisms. — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)
- Data access: SN-BAS-2025 on Hugging Face (19.3 GB). Access needs a manual request and a personalised NDA. Licence listed only as "other". — [HF SN-BAS-2025](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025)
- 2025 data: 77 annotated games for training, 22 held out. Many teams also used the 500-game SoccerNet Action Spotting dataset for pretraining. — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)

### Inferences
- The class set maps well onto our needs. Pass, Out, Throw In, Free Kick, Player Successful Tackle and Ball Player Block are possession-change events. Goal, Shot and Cross also end possession.
- The team output is left/right in image space, not home/away. For a panoramic tactical camera, left/right would need mapping to home/away through pitch geometry or team sides per half. This is an inference, not documented.
- GPL-3.0 on the code applies to code distribution, which matters little for an internal pipeline. Checkpoints were trained on NDA SoccerNet video. Their redistribution terms are unverified, so ask the team before committing weights to the repo.
- Rules fit: a public pretrained checkpoint is allowed. Fine-tuning on BAR-ATH using tracking as labels is not allowed. Commit any GPU-derived intermediate output with its script. Check the NDA terms of the checkpoint source before use.

### Gaps
- Exact 2023 BAS challenge numbers were not retrieved.
- Whether dudek's (2025 winner) weights are public: not found.
- Whether the Google Drive checkpoints are redistributable: not found. The PDF versions of the T-DEED papers could not be parsed (compressed PDF), so the paper's backbone variant, frames per clip and training compute are unverified.
- The 2025 report text did not state the video fps.

## Q2. Evidence on transfer from broadcast-trained models to a fixed wide tactical camera

### Takeaway
I found no published evaluation of broadcast-trained spotting models on a fixed wide or panoramic tactical camera. The closest evidence is cross-domain work showing sensitivity to broadcast style.

### Cited Findings
- Cabado et al. (CVPRW 2024) tested how action-spotting models transfer across leagues and broadcast styles. Their finding was that broadcast editing quality mattered more than the league. The paper is framed as a call for domain adaptation research. — [Beyond the Premier](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.html)
- FOOTPASS (2025) is a play-by-play action spotting benchmark on full-match broadcast video with tactical context. Its baselines are sensitive to occlusion and replay conditions. Videos require NDA through SoccerNet. — [arXiv 2511.16183](https://arxiv.org/pdf/2511.16183)
- The 2025 winner used synthetic camera-pan augmentation and heavy pretraining on 500+ broadcast matches. This suggests camera-motion variability matters even within broadcast. — [SoccerNet 2025 results](https://arxiv.org/html/2508.19182v1)
- Vidal-Codina et al. suggest their tracking-based method could transfer to lower-quality data such as tactical or broadcast footage with parameter adjustment, but did not test it. — [arXiv 2202.00804v3](https://arxiv.org/html/2202.00804v3)

### Inferences
- A wide tactical view makes the ball a few pixels across. The models above use 796x448 input from broadcast framing, where the ball is larger and players are bigger. A direct zero-shot transfer is likely to degrade a lot. This is reasoning, not measured.
- The broadcast-trained models also depend on cues absent from a tactical camera: close-ups, replays and camera cuts.
- Low-risk alternative: use these models only on the broadcast feed if available. The CLAUDE.md notes say the broadcast feed is not synced to the tactical video, so alignment would be needed first.
- Any test of transfer on BAR-ATH must be run by the human, with aggregate numbers only.

### Gaps
- No paper found on fixed-wide-camera action spotting. Searches for "panoramic" or "tactical camera" with action spotting gave nothing direct. No SoccerNet-trained model result on a tactical camera was found.

## Q3. Tracking-only event detection (Vidal-Codina 2022, ETSY, others) with a missing ball

### Takeaway
Vidal-Codina et al. detect passes and set pieces from tracking with over 90% precision and recall on high-quality data, but they need ball tracking and have no handling of long ball gaps. ETSY syncs events to tracking and does not detect them. Neither is tested at 41% missing ball.

### Cited Findings
- Vidal-Codina et al. is a deterministic decision-tree algorithm in two steps: a possession model (which player has the ball per frame) and an event model from possession changes plus laws of football. Events covered include passes, shots, crosses, receptions, interceptions and set pieces. — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804)
- Data: Track160, Tracab and Hawk-Eye at 25 Hz, from FIFA Club World Cup 2019/2020 and Bundesliga 2019-20. Only 2D ball coordinates are used. — [arXiv 2202.00804v3 HTML](https://arxiv.org/html/2202.00804v3)
- Possession is defined by a per-player radius (0.5 m for Track160, 1 m for Tracab and Hawk-Eye). Duel zone 1 m. A Savitzky-Golay filter smooths the ball. — [arXiv 2202.00804v3 HTML](https://arxiv.org/html/2202.00804v3)
- Accuracy against official Sportec events: passes, corners, throw-ins and goal kicks have precision and recall above 90%. Shots have 78% precision and 53% recall. Crosses are often labeled as passes. Free kicks are weakest. — [arXiv 2202.00804v3 HTML](https://arxiv.org/html/2202.00804v3)
- Missing ball data is used only to infer dead-ball status for Track160 and Hawk-Eye. Occlusion handling is not specified. The paper reports no timing-error metric. Ball tracking must be available. In-play/dead-ball information is critical for set-piece detection. — [arXiv 2202.00804v3 HTML](https://arxiv.org/html/2202.00804v3)
- ETSY (KU Leuven, Van Roy, Cascioli, Davis) synchronises SPADL event data to tracking with rules (kickoff alignment, action-specific filters, scoring using ball acceleration, ball height and ball-player distance). It needs events as input. It matched about 91.6% of events on 313 matches and has no ground truth. Code: ML-KULeuven/ETSY, licence not confirmed. — [ETSY blog](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization)
- ELASTIC (2025) reports 98.5% valid vs ETSY 86.5% on pass-like events. These are the authors' own numbers. It is also an event-tracking synchroniser. — [ETSY/ELASTIC search](https://arxiv.org/pdf/2508.09238)

### Inferences
- Vidal-Codina's logic (possession zone, ball direction/speed change, set pieces from player configuration plus dead-ball flag) is a reasonable template for rule-based boundary detection. It was validated on clean, near-complete ball data. With 41% of frames missing and noisy ball positions, recall will fall. No evidence says by how much.
- ETSY and ELASTIC need human event data. Per the competition rules, no human-produced data about BAR-ATH is allowed, so they are not usable on BAR-ATH. They may be useful on IDSSE, which has events.
- Interpolating the ball through gaps, or using the nearest-player proxy, could be tested on IDSSE with artificial ball dropout at 41%. This is a suggestion, not a finding from the sources.

### Gaps
- Per-category numbers and timing error for Vidal-Codina were only in confusion-matrix figures, which could not be read.
- No source found that quantifies event detection from tracking with a large fraction of ball frames missing.
- ETSY repo licence not verified. Other tracking-only detectors were not covered in this pass (limited search budget).

## Q4. CPU/GPU inference cost for about 2 hours of 25 fps video

### Takeaway
No source gives measured inference cost for T-DEED or the 2025 entries. The volume is about 180,000 frames. Only rough scaling arguments are possible.

### Cited Findings
- The T-DEED repo gives no hardware or inference-time figures. SoccerNetBall inference input is 796x448. — [T-DEED GitHub](https://github.com/arturxe2/T-DEED)
- The full submission has 199,319 rows at 25 fps, which is roughly 2.2 hours. — project CLAUDE.md
- ETSY processes about 2.7 minutes per game on CPU (event sync, not video). — [ETSY blog](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization)

### Inferences
- GPU inference of a clip-based 2D backbone model over about 200k frames is likely minutes to tens of minutes. CPU is likely hours. Both are unmeasured guesses.
- Plan consistent with the rules: run the model on GPU once, commit the per-frame or per-event output with its script, and let the final CSV step run on CPU from that.
- Fixed seeds and pinned dependencies are needed for byte-for-byte reproducibility. GPU nondeterminism is a risk for the intermediate file, so commit the output itself.

### Gaps
- No measured FPS, VRAM or CPU throughput for T-DEED. Benchmark on IDSSE or a public clip.
- The T-DEED backbone variant and clip length are unverified because the PDFs could not be read.
