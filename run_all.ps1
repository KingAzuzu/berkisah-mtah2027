# One-shot run on the competition data. Human only, in your own terminal (not inside Claude Code):
#     cd D:\barca ; .\run_all.ps1
# Prints aggregate lines only; paste the whole output back. Writes the submission CSVs to
# evaluation\results\ (gitignored).
$ErrorActionPreference = "Continue"
$root = "D:\barca\train"
$py = "idsse-tests\.venv\Scripts\python"
$r = "evaluation\results"
$want = "BAR-ATH_tactical_atd.txt"

Write-Host "== 0. locate the bundle =="
# Google Drive splits a download into several zips (...-1-001.zip, -002, ...), each holding different
# files. Unzip them all into one folder so the parts merge.
$dest = Join-Path $root "bundle"
$zips = @(Get-ChildItem $root -Recurse -Filter *.zip)
Write-Host "zip parts found: $($zips.Count)"
if (-not (Test-Path $dest)) {
    foreach ($z in $zips) { Write-Host "unzipping $($z.Name)"; Expand-Archive $z.FullName -DestinationPath $dest -Force }
}
$hit = Get-ChildItem $dest -Recurse -Filter $want -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $hit) { $hit = Get-ChildItem $root -Recurse -Filter $want -ErrorAction SilentlyContinue | Select-Object -First 1 }
if (-not $hit) {
    Write-Host "NOT FOUND: $want. File types under train (count per extension):"
    Get-ChildItem $root -Recurse -File | Group-Object Extension | ForEach-Object { "  $($_.Name)  $($_.Count)" }
    Write-Host "Top-level folders:"; Get-ChildItem $root -Directory | ForEach-Object { "  $($_.Name)" }
    exit 1
}
if ($hit.Directory.Name -ne "tactical") { Write-Host "WARNING: $want is not inside a 'tactical' folder" }
$env:MTAH_DATA_DIR = $hit.Directory.Parent.FullName
Write-Host "data dir: $env:MTAH_DATA_DIR"

Write-Host "== 1. submission v2 + backup v1 =="
& $py -m possession.bar_ath --method v2 --out "$r\v2.csv"
if (-not (Test-Path "$r\v2.csv")) { Write-Host "STOP: v2 failed"; exit 1 }
& $py -m possession.bar_ath --method v1 --out "$r\v1_backup.csv"
Write-Host "== 2. validator =="
& $py evaluation\validate_submission.py "$r\v2.csv" 2>&1 | Select-Object -Last 3
& $py evaluation\validate_submission.py "$r\v1_backup.csv" 2>&1 | Select-Object -Last 3
Write-Host "== 3. rerun identical? =="
& $py -m possession.bar_ath --method v2 --out "$r\v2_run2.csv" | Out-Null
Write-Host ((Get-FileHash "$r\v2.csv").Hash -eq (Get-FileHash "$r\v2_run2.csv").Hash)
Remove-Item "$r\v2_run2.csv" -ErrorAction SilentlyContinue
Write-Host "== 4. input statistics =="
& $py -m possession.atd_stats
& $py -m possession.camera_centre
Write-Host "== done: paste everything above =="
