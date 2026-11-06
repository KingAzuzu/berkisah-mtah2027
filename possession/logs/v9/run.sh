cd /d/barca; PY=idsse-tests/.venv/Scripts/python; P=possession/logs/v9/progress.txt
for V in A B; do for s in 20 21 22 23 24; do echo "training $V seed $s ($(ls possession/logs/v9/*.npz 2>/dev/null | wc -l)/50 folds done)" > $P; $PY -m possession.v9_run $V $s || exit 1; done; done
echo "evaluating" > $P; $PY -m possession.v9_eval; echo "done" > $P
