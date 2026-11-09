cd /d/barca
while [ ! -f possession/logs/v11/e1_done ]; do sleep 20; done
PY=idsse-tests/.venv/Scripts/python
$PY -m possession.v11_distill teachers >> possession/logs/v11/e2.log 2>&1 || { echo "E: ERROR teachers" > possession/logs/v11/progress.txt; exit 1; }
for s in 20 21 22 23 24; do
 $PY -m possession.v11_distill student $s >> possession/logs/v11/e2.log 2>&1 || { echo "E: ERROR student $s" > possession/logs/v11/progress.txt; exit 1; }
done
echo "E2 done" > possession/logs/v11/progress.txt
