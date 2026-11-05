#!/bin/bash
# confirmation LOMO, fresh seeds 10-14, configs cA and cB: bash possession/v4c_queue.sh
cd "$(dirname "$0")/.."
python3 -c "import time;open('possession/logs/v4c_t0.txt','w').write(str(time.time()))" 2>/dev/null || date +%s > possession/logs/v4c_t0.txt
for c in cB cA; do for s in 10 11 12 13 14; do echo "$c $s"; done; done |   xargs -P 7 -L 1 sh -c 'idsse-tests/.venv/Scripts/python -W ignore -m possession.v4_run $0 $1 >> possession/logs/v4c_run_$0_$1.log 2>&1'
echo "100% | DONE | jobs 50/50 | ETA 0 min" > possession/logs/v4c_progress.txt
echo DONE >> possession/logs/v4c_progress.txt
