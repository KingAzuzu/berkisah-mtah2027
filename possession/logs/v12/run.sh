cd /d/barca; PY=idsse-tests/.venv/Scripts/python
for s in 20 21 22 23 24; do $PY -m possession.v12_phase v8 $s || { echo "A: ERROR seed $s" > possession/logs/v12/progress.txt; exit 1; }; done
$PY -m possession.v12_eval v8 > possession/logs/v12/eval_v8.log 2>&1
echo "A: done $(tail -1 possession/logs/v12/eval_v8.log)" > possession/logs/v12/progress.txt
