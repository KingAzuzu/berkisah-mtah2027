# IDSSE tests A and B

Two preregistered analyses (see `PREREGISTRATION.md`) on the public IDSSE dataset. Results in `RESULTS.md` and `results.json`.

## Rerun

Python 3.12, about 3 GB disk for the data, 8+ GB RAM.

```
./run_all.sh
```

which does: create `.venv` and install `requirements.txt`, `download.py` (7 matches into `data/`, gitignored),
`test_a.py` (corners), `test_b.py` (possession losses), `make_results.py` (thresholds -> `results.json`, `plots.png`).
Seeds are fixed, results are deterministic. Parsed matches are cached in `data/cache/`.

## Files

- `idsse.py` loader, events, coordinate normalisation. `corners.py` corner kick alignment.
- `test_a.py`, `test_b.py`, `make_results.py`, `download.py`.

## Data and attribution

Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event data in elite soccer."
Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y. Released by DFL / IDSSE under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Files are fetched from the `pysport/idsse-data` mirror of the release.
