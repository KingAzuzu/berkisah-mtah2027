# v21 on IDSSE only (public data, CC BY 4.0): LOMO screening, gates, freeze, locked check, determinism.
# Safe anywhere (never reads competition data). Fast on a GPU (train21 uses CUDA + bf16 when available), ~hours on CPU.
# Afterwards: commit possession/models/tcn_v21_s*.pt, dead_v21.joblib, params_v21.json, logs/v21, results/v21.md,
# then run the competition file in your own terminal:  .\run_all.ps1 v21
$ErrorActionPreference = "Stop"
$py = "idsse-tests\.venv\Scripts\python"
& $py -m possession.v21_lomo data
foreach ($s in 20, 21, 22) { foreach ($r in "r2", "r3", "r3x", "r3xl") { & $py -m possession.v21_lomo nets $r $s } }
& $py -m possession.v21_lomo dead
& $py -m possession.v21_select
& $py -m possession.tcn_v21 freeze
& $py -m possession.tcn_v21 locked
& $py -m possession.repro_check --method v21 --threads 1
& $py -m possession.repro_check --method v21 --threads 4
