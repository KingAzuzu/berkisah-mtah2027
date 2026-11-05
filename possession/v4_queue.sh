#!/bin/bash
# runs all v4 ablation folds, 7 processes at a time: bash possession/v4_queue.sh
cd "$(dirname "$0")/.."
printf "full 0\nfull 1\nfull 2\nnomet 0\nnomet 1\nnomet 2\nnosc 0\nnosc 1\nnosc 2\nnobd 0\nnobd 1\nnobd 2\nfull 3\nfull 4\n" | \
  xargs -P 7 -L 1 sh -c 'idsse-tests/.venv/Scripts/python -W ignore -m possession.v4_run $0 $1 >> possession/logs/v4_run_$0_$1.log 2>&1'
echo ALLDONE > possession/logs/v4_queue_done.txt
