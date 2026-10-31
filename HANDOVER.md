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
