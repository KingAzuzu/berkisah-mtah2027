# v21 on IDSSE only (public data, CC BY 4.0): LOMO screening, gates, freeze, locked check, determinism.
# Safe anywhere (never reads competition data). Fast on a GPU (train21 uses CUDA + bf16 when torch sees one), many hours on CPU.
# Optional: $env:V21_PY = path of a python with CUDA torch (default idsse-tests\.venv); $env:V21_JOBS = parallel matches in
# the data step (default 2, ~2-3 GB RAM each). Rerunning skips finished work (cached data, finished LOMO folds).
# Afterwards commit possession/models/tcn_v21_s*.pt, models/dead_v21.joblib, params_v21.json, logs/v21, logs/locked_v21.json,
# results/v21.md, then run the competition file in your own terminal:  .\run_all.ps1 v21
$py = if ($env:V21_PY) { $env:V21_PY } else { "idsse-tests\.venv\Scripts\python" }
function Step { & $py @args; if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $args"; exit 1 } }
Step -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Step -m possession.v21_lomo data
foreach ($s in 20, 21, 22) { foreach ($r in "r2", "r3", "r3x", "r3xl") { Step -m possession.v21_lomo nets $r $s } }
Step -m possession.v21_lomo dead
Step -m possession.v21_select
Step -m possession.tcn_v21 freeze
Step -m possession.tcn_v21 locked
Step -m possession.repro_check --method v21 --threads 1
Step -m possession.repro_check --method v21 --threads 4
Write-Host "== done: paste results/v21.md, the locked line and the two sha256 lines =="
