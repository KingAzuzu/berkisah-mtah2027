# Live view of a running experiment: progress line + GPU load, refreshed every 5 s. Ctrl+C to stop.
#     .\watch.ps1            # v4 experiment
#     .\watch.ps1 tcn_x      # any possession\logs\<name>_progress.txt
param([string]$name = "v4")
$f = "possession\logs\${name}_progress.txt"
while ($true) {
    Clear-Host
    Write-Host "== $name  $(Get-Date -Format HH:mm:ss) =="
    if (Test-Path $f) { Get-Content $f -Tail 1 } else { Write-Host "(no progress file yet)" }
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader |
        ForEach-Object { Write-Host "GPU: $_" }
    Start-Sleep 5
}
