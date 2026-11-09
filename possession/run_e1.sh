cd /d/barca
for V in V1 V2; do for s in 20 21 22 23 24; do
 idsse-tests/.venv/Scripts/python -m possession.v11_ext $V $s >> possession/logs/v11/e1.log 2>&1 || exit 1
 echo "E1-$V seed $s done (n=$(ls possession/logs/v11/${V}_s*.npz | wc -l)/25)" > possession/logs/v11/progress.txt
done; done
touch possession/logs/v11/e1_done
