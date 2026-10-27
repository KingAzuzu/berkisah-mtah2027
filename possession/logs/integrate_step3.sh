PY=D:/barca/idsse-tests/.venv/Scripts/python
cd /d/barca
$PY -m possession.integrate_run --cond seed0 --ref > possession/logs/integrate_step3_seed0ref.txt 2>&1
for c in seed1 seed2; do
  $PY -m possession.integrate_run --cond $c --ref --runs s1_rules s1_players s1_ball > possession/logs/integrate_step3_$c.txt 2>&1
done
for c in noise0.3 noise1.0; do
  $PY -m possession.integrate_run --cond $c --ref --runs s1_rules s1_players s1_ball --mismatch s1_players s1_ball > possession/logs/integrate_step3_$c.txt 2>&1
done
echo done > possession/logs/integrate_step3_done.txt
