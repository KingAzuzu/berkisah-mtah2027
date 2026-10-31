# One-shot run on the competition data. Human only, in your own terminal (not inside Claude Code):
#     cd D:\barca ; .\run_all.ps1
# Prints aggregate lines only; paste the whole output back. Writes the submission CSVs to
# evaluation\results\ (gitignored).
$ErrorActionPreference = "Continue"
$env:MTAH_DATA_DIR = "D:\barca\train"
$py = "idsse-tests\.venv\Scripts\python"
$r = "evaluation\results"

Write-Host "== 1. submission v2 + backup v1 =="
& $py -m possession.bar_ath --method v2 --out "$r\v2.csv"
& $py -m possession.bar_ath --method v1 --out "$r\v1_backup.csv"
Write-Host "== 2. validator =="
& $py evaluation\validate_submission.py "$r\v2.csv" 2>&1 | Select-Object -Last 3
& $py evaluation\validate_submission.py "$r\v1_backup.csv" 2>&1 | Select-Object -Last 3
Write-Host "== 3. rerun identical? =="
& $py -m possession.bar_ath --method v2 --out "$r\v2_run2.csv" | Out-Null
Write-Host ((Get-FileHash "$r\v2.csv").Hash -eq (Get-FileHash "$r\v2_run2.csv").Hash)
Remove-Item "$r\v2_run2.csv"
Write-Host "== 4. input statistics =="
& $py -m possession.atd_stats
& $py -m possession.camera_centre
Write-Host "== done: paste everything above =="
