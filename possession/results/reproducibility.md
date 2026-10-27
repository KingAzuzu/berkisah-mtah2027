# Reproducibility of the v2 path

Scope: the v2 labelling path (`possession.pipeline.predict`: WS2 rules evidence + players-only dead-ball HGB `models/dead_v2.joblib` + Viterbi).
IDSSE tuning match J03WMX only (degraded half 0, 70,708 frames, from `run_idsse.load`), passed as ball / xy / team only. Locked matches and competition data never loaded; `bar_ath.py` never run.

## Files added

- `requirements-submission.txt`: exact pins (numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, threadpoolctl 3.7.0, python-dateutil 3.0.0, six 1.17.0, tzdata 2026.5, cloudpickle 3.1.2, narwhals 2.28.0). Found by tracing imports (pipeline, dead, rules, safety, mbr, data, run_idsse, idsse, possession_metrics, starter/baseline, games); the last two came in as dependencies when the fresh venv was built, and are pinned to what pip resolved. No torch. The versions equal those stored in `dead_v2.joblib`. The organiser's `requirements.txt` is untouched.
- `possession/repro_check.py`: forces OMP/OPENBLAS/MKL thread vars before numpy loads, predicts the half, prints SHA-256 of the int8 label bytes and of the letter string. Refuses the locked matches.
- `docs/technical_document.md`.
- `bar_ath.py` not edited (see below).

## Thread invariance

`dead.py` already does `os.environ.setdefault(..., "4")` before its numpy import (only matters if nothing imported numpy earlier). Empirically the output does not depend on the thread count: 1, 4 and 16 threads give the same hash, so no `bar_ath.py` change was needed.

## Results

| run | python / numpy / sklearn | threads | sha256 of label bytes (int8 codes) |
|---|---|---|---|
| working copy (`idsse-tests\.venv`) | 3.12.10 / 2.5.4 / 1.9.1 | 1 | 4dd0d8de111077f9e5b06bc1ca2eb69feb71016d85d9c30a67ec36cb3d2bacfa |
| working copy | same | 4 | 4dd0d8de111077f9e5b06bc1ca2eb69feb71016d85d9c30a67ec36cb3d2bacfa |
| working copy | same | 16 | 4dd0d8de111077f9e5b06bc1ca2eb69feb71016d85d9c30a67ec36cb3d2bacfa |
| fresh clone + fresh venv | 3.12.10 / 2.5.4 / 1.9.1 | 1 | 4dd0d8de111077f9e5b06bc1ca2eb69feb71016d85d9c30a67ec36cb3d2bacfa |
| fresh clone + fresh venv | same | 4 | 4dd0d8de111077f9e5b06bc1ca2eb69feb71016d85d9c30a67ec36cb3d2bacfa |

SHA-256 of the letters string, every run: 177342422de9893a29180d525257f0c77a5f936262d5ac9f7eb3efad5ec1354f. Counts H 19187, A 27915, D 23606.

## Exact commands

Working copy (from `D:\barca`):

    idsse-tests\.venv\Scripts\python -m possession.repro_check --threads 1
    idsse-tests\.venv\Scripts\python -m possession.repro_check --threads 4

Fresh clone (done in `C:\Users\MUHAMM~1\AppData\Local\Temp\claude\D--barca\04b492b8-31f8-4130-865a-eb43fd8cc395\scratchpad\clone`):

    git clone --branch possession-dev D:\barca clone
    cd clone
    py -3.12 -m venv venv
    venv\Scripts\python -m pip install -r requirements-submission.txt
    venv\Scripts\python -m possession.repro_check --threads 1 --idsse-data D:\barca\idsse-tests\data
    venv\Scripts\python -m possession.repro_check --threads 4 --idsse-data D:\barca\idsse-tests\data

The IDSSE data was read in place via `--idsse-data` (or env var `IDSSE_DATA_DIR`), not copied. It uses the cached degraded half `cache/J03WMX_degraded0.pkl`.

## Caveats (must be handled before submission)

- The clone only contains committed files. `requirements-submission.txt` and `possession/repro_check.py` were not committed (instruction), so I copied those two files into the clone. The committed `models/dead_v2.joblib` and `params_v2.json` were used from the clone's own checkout. Commit these new files before the real clone test.
- Verified: same machine, same Windows/CPU, same pinned wheels. Not verified: another machine or CPU architecture (HGB prediction sums floats per sample over trees in fixed order, so I expect identity, but it is untested), and the full BAR-ATH run.
- The test covers `pipeline.predict` on one half. The remaining steps in `bar_ath.py` (ATD parsing, forced-D block 110,808-113,741, CSV writing) are not exercised here. For the real file the human should run `bar_ath` twice in the clean venv and `cmp` the two CSVs, then run `evaluation\validate_submission.py`.
- The IDSSE degraded half comes from a pickle cache; `degrade.py` seeds with crc32 (`zlib`), so regeneration should be deterministic, but cache regeneration was not retested here.
