#!/usr/bin/env bash
# Rerun everything. Python 3.12. Downloads the 7 IDSSE matches into data/ on first run.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
PY=.venv/bin/python
$PY download.py
$PY test_a.py
$PY test_b.py
$PY make_results.py
