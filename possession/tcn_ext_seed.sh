s=$1
for c in base mir ext extmir; do
  [ -f possession/logs/tcn_ext_${c}_s$s.json ] || idsse-tests/.venv/Scripts/python -m possession.tcn_ext_run $c $s
done