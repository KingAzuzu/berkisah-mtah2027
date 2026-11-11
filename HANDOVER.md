# Handover: More Than A Hack 2027, Phase 1

Untuk: teman yang menjalankan Claude Code di laptop GPU (Legion 5, RTX 4060).
Dari: Dihan, 27 Okt 2026.

## 1. Situasi sekarang

- **Tugas Phase 1:** label setiap frame laga BAR-ATH (199.319 frame) dengan `H`, `A`, `D`, atau `X`. Peringkatnya
  ditentukan spellF1, yaitu berapa possession yang ketemu utuh dengan batas waktu yang tepat. Deadline **17 Nov**.
  Target: **20 besar**.
- **Yang dinilai bukan algoritma berat.** Keempat baseline organiser memakai algoritma yang sama dan cuma beda
  lebar smoothing, tapi spellF1-nya beda 5×. Skor hilang di dua tempat: possession yang terpecah, dan batas
  transisi yang telat.
- **Pipeline-nya belum ada.** Yang sudah ada: riset, aturan, plan ini, dan `idsse-tests/` (loader IDSSE plus dua
  tes yang sudah dijalankan).
- **Tim:** Dihan dan kamu. Masih kurang **1 anggota teknis**. Semua anggota harus register dengan team name yang
  sama sebelum **10 Nov**.

## 2. Aturan yang tidak bisa ditawar

1. **Data kompetisi tidak pernah masuk ke Claude Code atau AI cloud mana pun** (T&C §8). Claude menulis kode dan
   mengujinya di IDSSE. Kamu yang menjalankan apa pun di BAR-ATH dari terminal biasa, lalu kirim balik angka
   agregat saja.
2. **Tidak ada label manual di BAR-ATH**, termasuk untuk memilih parameter (issue #10, hand labels for tuning).
3. **Public 5 menit hanya untuk cek kebenaran pipeline**, bukan untuk tuning.
4. **File CSV harus bisa dihasilkan ulang persis byte per byte:** inferensi di CPU, weight di-commit, dependency
   di-pin, seed dikunci.
5. **Dua laga IDSSE dikunci untuk cek akhir:** `J03WQQ` dan `J03WR9`. Jangan pernah tuning di dua laga ini.

## 3. Setup (sekali, sekitar 1 jam)

- [ ] Register di https://landing.mobileworldcapital.com/more-than-a-hack-2027 dengan team name `<TEAM_NAME>`,
      lalu tanda tangan T&C. Link data dikirim per batch, paling lambat 3 hari.
- [ ] Download data ke `D:\mtah-data\`, **di luar** folder repo, dengan struktur `tactical/` dan `broadcast/`
      seperti di DATA.md.
- [ ] Clone repo organiser: `git clone https://github.com/metrica-sports/more-than-a-hack-2027 mtah-team`.
      Lalu copy isi folder handover ini ke root repo tersebut: `CLAUDE.md`, `HANDOVER.md`, `ISSUE_DRAFT.md`,
      `mtah_guard.py`, `.claude/`, dan `idsse-tests/`.
- [ ] Pastikan Python 3.12 ada, dan perintah `python --version` jalan, karena hook memanggil `python`.
- [ ] Cek guard data: `python .claude/hooks/test_guard_data.py`. Semua baris harus `ok`.
- [ ] Install Claude Code, buka folder repo, lalu minta: "baca CLAUDE.md dan HANDOVER.md".
- [ ] Tes guard secara langsung: minta Claude "tampilkan isi D:\mtah-data". Permintaan ini **harus diblokir**.
      Kalau tidak diblokir, berhenti dan kabari Dihan.
- [ ] Set `MTAH_DATA_DIR=D:\mtah-data` hanya di terminal biasa yang kamu pakai untuk menjalankan pipeline.
- [ ] Data IDSSE (publik, 2,7 GB): buat venv di `idsse-tests\`, lalu `.venv\Scripts\python download.py`.
      `run_all.sh` ditulis untuk Linux, jadi di Windows pakai `.venv\Scripts\python`.
- [ ] Jangan pernah buka file data di editor yang AI assistant-nya aktif (Copilot, Cursor, dan sejenisnya).
- [ ] Repo GitHub tim dibuat private dulu dan data tidak pernah di-commit. Ubah jadi public sebelum submit.

## 4. Pembagian kerja

| Siapa | Tugas |
|---|---|
| Claude Code (laptopmu) | Menulis dan menguji kode, hanya di IDSSE dan Metrica sample data |
| Kamu | Menjalankan perintah di BAR-ATH dari terminal biasa, mengirim balik angka agregat, jalur GPU |
| Dihan | Mencari anggota ke-3, post issue, review metode dan dokumen teknis, mengambil keputusan |

## 5. Plan 3 minggu

**Minggu 1 (28 Okt – 2 Nov): safety net yang bisa disubmit**
- [ ] `possession/` package. Loader IDSSE memakai ulang `idsse-tests/idsse.py`. Truth dari `BallStatus` dan
      `BallPossession`.
- [ ] `degrade.py`: membuat IDSSE mirip ATD Metrica, hanya memakai angka dari DATA.md (bola ada di 56,9% frame,
      rata-rata 20,5 objek ter-track, 46 slot, ID yang tertukar). Bola sengaja lebih sering hilang saat ramai.
- [ ] Scorer di IDSSE memakai `evaluation/possession_metrics.py` milik organiser.
- [ ] Safety net: bukti pemain terdekat ke bola, Viterbi H/A/D, set piece Smart Tagging sebagai batas restart,
      bola mati ditimpa `D`, frame 110.808–113.741 = `D`.
- [ ] Kamu jalankan safety net di BAR-ATH, lalu `validate_submission.py`. Simpan CSV-nya sebagai cadangan.
- [ ] Tes ide 1 (kamera) dan ide 4 (aturan restart). Lihat bagian 6.
- [ ] Dihan post `ISSUE_DRAFT.md`.

**Minggu 2 (3–9 Nov): model belajar dan decoding**
- [ ] LightGBM di IDSSE yang sudah dirusak, leave-one-match-out di 5 laga yang tidak dikunci.
- [ ] Decoder yang paham durasi possession (sebaran durasi dari IDSSE). Bandingkan dengan MBR decoding (ide 3).
- [ ] Alat analisis error: skor yang hilang dipecah jadi possession terlewat, terpecah, dan batas telat.
- [ ] Optuna semalaman, target rata-rata spellF1 di 5 laga.
- [ ] Tes ide 2 (sinkron broadcast). Jalur GPU (ide 6) hanya setelah pertanyaan 3 di issue terjawab.
- [ ] **Tim lengkap 3 orang dan sudah register sebelum 10 Nov.**

**Minggu 3 (10–16 Nov): freeze dan submit**
- [ ] Metode dibekukan 13 Nov. Evaluasi sekali di `J03WQQ` dan `J03WR9`.
- [ ] Run final di BAR-ATH, validasi, cek public window hanya untuk kebenaran.
- [ ] Fresh clone di mesin lain, harus menghasilkan SHA-256 yang sama.
- [ ] Dokumen teknis mengikuti SUBMISSION.md: pendekatan, aset yang dipakai dan yang dibuang, data latih, aturan
      possession, keterbatasan, cara reproduksi.
- [ ] Submit **16 Nov** (sehari sebelum deadline). Nama file `<TEAM_NAME>.csv` dan `<TEAM_NAME>.pdf`, pakai akun
      Google, sertakan URL repo publik.

## 6. Ide dan tes berikutnya

Batas "dipakai" untuk tiap ide sudah ditetapkan di sini, sebelum melihat hasil.

| # | Ide | Tes | Dipakai kalau |
|---|---|---|---|
| 1 | Titik tengah kamera tactical sebagai perkiraan posisi bola (homography: pitch → pixel, jadi pakai inversnya untuk titik tengah layar) | Kamu jalankan di BAR-ATH: median jarak titik tengah kamera ke bola ATD, di frame yang bolanya terlihat | median < 10 m |
| 2 | Bola dari broadcast mengisi bola yang hilang di tactical. Sinkron lewat posisi pemain dalam koordinat lapangan | Persen frame graded yang bolanya hilang di tactical tapi terlihat di broadcast setelah sinkron | ≥ 10% |
| 3 | MBR decoding: pilih segmentasi dengan rata-rata spellF1 tertinggi | IDSSE, leave-one-match-out di 5 laga | +0,02 rata-rata vs Viterbi |
| 4 | Aturan restart sebagai kode: tim pelempar throw-in = pemain yang berdiri di luar garis | IDSSE: dibandingkan dengan tim di event ThrowIn DFL | akurasi ≥ 90% |
| 5 | Ball Radar (KDD 2023): menebak posisi bola dan pemegangnya dari lintasan pemain | Latih di IDSSE | opsional |
| 6 | T-DEED Team Ball Action Spotting (checkpoint publik di SoccerNet/sn-teamspotting) | GPU: jumlah OUT dan THROW-IN dibanding set piece Smart Tagging (agregat) | opsional, tunggu issue |

## 7. Yang sudah kita pelajari (jangan diulang)

- Tes IDSSE A (set piece) dan B (rest defence) gagal. Detailnya di `idsse-tests/RESULTS.md`. Ide Phase 2 dipilih
  ulang setelah 26 Nov.
- Model yang membaca bentuk tim selalu telat beberapa detik di transisi, jadi hanya dipakai sebagai cadangan saat
  bola hilang.
- Klaim "struktur tim → bahaya" gagal di data publik yang kecil, baik di tes kita maupun di repo PriyaanGada/off-ball.
- Pendekatan kita mirip tim-tim teratas (danieltyukov, alperglcoglu, PLOG). Kita menang lewat eksekusi di batas
  possession, bukan lewat ide yang berbeda.

## 8. Ruling organiser (GitHub issues)

| Issue | Isi |
|---|---|
| #2 (detector weights from the match) | Detector yang dilatih di video laga ini dengan tracking sebagai label: tidak boleh. GPU boleh. |
| #8 (how much to hand label) | Label manual di laga lain boleh berapa pun. Di BAR-ATH: nol. |
| #10 (hand labels for tuning) | Memilih parameter memakai label manual di BAR-ATH juga tidak boleh. |
| #11 (reproducibility) | Training tidak harus bisa diulang persis. CSV final harus: weight, CPU, pin, seed. |
| #14 (commit intermediate outputs) | Output antara boleh di-commit bersama script pembuatnya. Menonton video boleh. |
| #15 (look-ahead allowed) | Boleh memakai frame setelahnya. Pendekatan live juga diterima. |
| #16 (IDSSE dataset for training) | IDSSE boleh. Cantumkan sitasi CC BY 4.0. Tuning hanya di IDSSE. |

## 9. Tanggal penting

| Tanggal | Apa |
|---|---|
| 10 Nov | Registrasi tutup (3 anggota, team name sama) |
| 17 Nov | Deadline Phase 1 |
| sebelum 26 Nov | Hasil Phase 1 diumumkan |
| 26 Nov | Sesi pembukaan Phase 2, minimal 1 anggota wajib hadir |
| 14 Jan, 11:00 CET | Deliverable Phase 2 |
| 22 Feb, 11:00 CET | Deck final (hanya untuk 5 finalis) |
| 2 Mar 2027 | Final di Fira Barcelona Montjuïc |

## Status 1 Nov 2026 (latest; supersedes older status lines above)

- Data bundle unpacked by the human into the repo's `train` folder (gitignored, guarded). Videos (zip parts 001/002) not used.
- `run_all.ps1` (human only) builds the CSVs, validates, checks rerun identity, prints input stats. `run_all.ps1 v2_1` does the v2.1 run.
- Real ATD aggregates: ball seen 0.593; ball-missing runs mean 17.2 s (we had assumed 2 s); tracklets mean 107 s; ball jitter ~1-2 cm. Camera-centre idea: median 23.6 m, FAIL.
- Matched simulator `possession/degrade_matched.py`, LOMO spellF1 on TUNE: v1 0.296, v2 0.430, **v2.1 0.489 (5/5 matches)**. v2.1 = v2 + dead-ball model with ball features (`params_v2_1.json`, `models/dead_v2_1.joblib`). `results/matched_degrade.md`.
- v2.1 assumption checked label-free on real tracking (`possession/ball_loss_check.py`): P(ball missing | v2 says D) 0.702 vs alive 0.101 (simulator 0.636 / 0.256). Holds.
- Competition CSVs: v2 (393 poss, mean 7.9 s), v1 backup, **v2.1 (461 poss, mean 8.1 s): chosen for submission**. All valid, rerun identical. Public 5-min check (alignment only, not tuning): v2 latency -0.3 frames, alive/dead F1 0.95.
- TODO: rename the v2.1 CSV to `<team_name>.csv`; update `docs/technical_document.md` from v2 to v2.1 (numbers above); 3rd member by 10 Nov; submit by 17 Nov. Ball Radar (WS5) not used.

## v3 (1 Nov 2026): TCN + v2.1 fusion
- v3 = 3-seed TCN ensemble fused 50/50 with v2.1 evidence, then Viterbi (params_v2_1 costs). LOMO spellF1 on TUNE under matched2: ensemble 0.641 [0.597, 0.681] vs v2.1 0.556; single seeds 0.621-0.627. `results/tcn.md`.
- Wired as `possession.bar_ath --method v3` (torch imported lazily; default stays v2). Weights `possession/models/tcn_v3_s{0,1,2}.pt`, `params_v3.json`. torch 2.11.0 (CPU build) pinned in `requirements-submission.txt`.
- Determinism verified incl. fresh clone + fresh venv (`results/reproducibility.md`). Not yet run on the competition tracking: human runs `.\run_all.ps1 v3`.

## v4 (5 Nov 2026): submitted method
- v4 = 5-seed TCN (seeds 10-14) with boundary head and EMA, domain randomisation, mirror aug + TTA, trained on IDSSE TUNE only (no SkillCorner, no Metrica), fused 50/50 with v2.1 evidence, Viterbi costs scaled by (1 - 0.6 x boundary prob). LOMO spellF1 0.6639 [0.614, 0.725], 5/5 wins vs v3 (0.6411). `results/tcn_v4.md`, `results/tcn_v4_confirm.md`, `params_v4.json`.
- Selection: config found post-hoc in the ablations, then confirmed on fresh seeds 10-14 under a rule fixed beforehand (mean >= 0.661 and >= 4/5 wins). No locked check (J03WQQ/J03WR9 not evaluated).
- Human run on the competition tracking (aggregates): 199,319 rows, validator passed, rerun hash compare True; graded H 52,755, A 37,373, D 68,729; 361 possessions, mean 10.0 s; ball seen 0.593. numpy "Mean of empty slice" warnings are harmless (frames without players/ball).
- **`evaluation/results/v4.csv` is now the submission file; `v3.csv` is the backup.** Run: `.\run_all.ps1 v4` / `--method v4`.

## v6 (6 Nov 2026): submitted method
- v6 = v4 recipe with CE and boundary loss x3 within +-1 s of an H<->A change next to a spell < 3 s, seeds 20-24, v4 decoder. LOMO spellF1 0.6860 [0.650, 0.736], 5/5 wins vs v4 (0.6639). Seed control (v4 recipe, seeds 20-24) 0.6712: loss effect +0.0149 (5/5), seed effect about +0.007. `results/v6.md`, `params_v6.json`, weights `models/tcn_v6_s{20..24}.pt`.
- Failed or skipped: E1 decoder retune 0.6790 (gate 0.6839), v5 Smart Tagging team prior 0.6586 (1/5 wins), ball-fill bound +0.028 in-sample (not pursued). No locked check.
- Human run on the competition tracking (aggregates): 199,319 rows, validator passed, rerun hash True; graded H 53,338, A 37,111, D 68,408; 402 possessions, mean 9.0 s; ball seen 0.593.
- **`evaluation/results/v6.csv` is now the submission file; `v4.csv` is the backup.** Run: `.\run_all.ps1 v6` / `--method v6`. Technical document updated to v6; TODOs left: team name, repo URL, final commit hash.

## v7 study (6 Nov 2026): nothing adopted, v6 stays the submission
Question: can we reach spellF1 0.85? Answer from three tests (IDSSE TUNE, LOMO, organiser scorer; gate fixed before each run):
- **Ceiling of the v6 recipe on clean (undegraded) IDSSE: 0.724** (TCN-only, 2 seeds; same decoding on degraded 0.677). Degradation costs
  only ~0.05; even perfect tracking gives ~0.72 with this recipe, and spells are still under-predicted. 0.85 is out of reach. `results/v7_ceiling.md`.
- Decoder retune on v6 nets, wider 600-point grid, nested: 0.6887 vs 0.6860, 3/5 wins: FAIL. Surface is flat. `results/v7_decoder.md`.
- Short-spell loss x5 0.6855 (3/5), x8 0.6813 (2/5): FAIL. More predicted spells, but not the right ones. `results/v7_short.md`.
- Commits a005fe6, 18c900b. **The v6 CSV remains the submission file; the v4 CSV the backup.**

## v8 (6 Nov 2026): convention candidate, NOT the submission yet
- DFL truth flips owner on short touches (19% of IDSSE spells are < 2 s X-Y-X); CHALLENGE.md says uncontrolled touches are not a change.
- v8 = v6 recipe trained on touch-merged labels (K 2 s). LOMO vs merged truth 0.7563 (5/5) vs v6 0.7099; vs DFL truth 0.6517 vs v6 0.6860.
- Frozen and wired: `.\run_all.ps1 v8` / `--method v8`. **Use v8 only if the organiser confirms touches do not flip the owner;** else keep v6.
  `possession/results/v8_convention.md`. Issue to organiser: to be posted by Dihan.
- Human run of v8 on the competition tracking (aggregates): 199,319 rows, validator passed, rerun hash True; graded H 53,449, A 37,040,
  D 68,368; 326 possessions, mean 11.1 s (v6: 402, 9.0 s). v8 CSV ready next to v6; v6 stays the submission until the organiser answers.

## v9-v10 (6 Nov 2026): model at its limit; hedge option wired
- Failed gates (IDSSE, vs merged truth, v8 0.7563): v9 3000 steps 0.7493, longer context 0.7347; boundary refiner 0.7553; MBR 0.7545;
  10-seed ensemble 0.7495. `results/v9.md`, `results/v10_refine.md`, `results/v10_hedge_mbr_seeds.md`.
- v10 = hedge: v6 + v8 nets in one ensemble. vs DFL 0.6686, vs merged 0.7449, worst-case regret 0.017 (v6 0.047, v8 0.034).
  Wired: `.\run_all.ps1 v10`. Deterministic (J03WMX h0 SHA d93596b0... at 1 and 4 threads).
- Submit rule: organiser says touches ignored -> v8; touches count -> v6; no answer -> v10.
- Human run of v10 on the competition tracking (aggregates): 199,319 rows, validator passed, rerun hash True; graded H 53,223, A 37,302,
  D 68,332; 347 possessions, mean 10.4 s (v6 402 / 9.0 s, v8 326 / 11.1 s). All three CSVs ready.

## Real-data sanity checks (11 Nov 2026, label-free, aggregates run by the human on the v10 CSV)
- `tag_truth_check`: clear-control proxy (ball seen, own player <= 1.5 m, no opponent within 3 m, >= 0.6 s; = truth 0.967 on IDSSE).
  Our labels = proxy 0.970 (IDSSE v8 0.990): the TCN path behaves on real data. Smart Tagging team = proxy only 0.60-0.76 on
  final-third / box codes while our labels = proxy 0.95-0.98 in every code: tags are coarse, v5 conclusion confirmed.
- `dead_check`: 13.0% of clear-control frames labelled D (IDSSE v8 5.0%, truth 4.7%). 71% of them are in D runs > 10 s and they are
  2.7x enriched inside Smart Tagging set-piece intervals; short same-team D runs (< 3 s) hold only 4.8% (9 runs of 1-3 s in the match).
  Reading: stoppages where the taker stands on a visible ball, not split possessions. No fix warranted.

## Convention check on the public window (11 Nov 2026, human run, aggregates only; rule fixed before the count)
- `gt_convention_check`: 17 possessions, mean 14.2 s, shortest 0.76 s; possessions < 2 s: 2; short X-Y-X touches: 2
  (IDSSE DFL 5-min windows: < 2 s 5-20, X-Y-X 0-18, median 6). Rule -> **v10**.
- Reading: the truth does flip the owner on some brief touches (so v8's blanket merge is too strong), but far less often than DFL
  (so v6 over-splits): an in-between convention, which is what the v10 hedge covers. No parameter is chosen from this count.
- **Final choice: v10.** Fallback: v6. Used only to understand the definition (organiser email; CHALLENGE.md), recorded here for the technical document.
