# Two diagnostics for the human to run on BAR-ATH

Both scripts load the competition data, so they refuse under Claude Code (`CLAUDECODE`). Run them in
your own terminal. Neither prints a per-frame row. The JSON files go to `data/stats/` (gitignored).
Neither script was run on real data; they were tested on synthetic files only
(`tests/test_diagnostics.py`, 8 checks passed).

## Commands (PowerShell, from D:\barca, outside Claude Code)

```powershell
cd D:\barca
$env:MTAH_DATA_DIR = "D:\mtah-data"
idsse-tests\.venv\Scripts\python -m possession.atd_stats
idsse-tests\.venv\Scripts\python -m possession.camera_centre
```

## What to paste back

Only the printed text blocks (about 15 lines and 5 lines). Do not paste the JSON if you have edited
it, and do not paste any rows of the input files.

1. `atd_stats`: the whole block, starting "ATD input statistics over graded frames".
2. `camera_centre`: the whole block, ending in the PASS or FAIL line.

## Camera centre (HANDOVER section 6, idea 1)

Pixel (960, 540) goes through the inverse homography to normalised pitch coordinates, then to metres
with the pitch from `games.GAMES` (105.3 x 68.0), and the distance to the ATD ball is taken on graded
frames with a valid homography and a seen ball. Pre-set bar, unchanged: median distance under 10 m
means "use it". The script prints PASS or FAIL against that bar.

## atd_stats: what each number is for

- Run lengths (ball seen, ball missing), objects per frame per team and slot tracklet lengths check
  the assumptions in `possession/degrade.py`: `BALL_GAP_S` = 2.0, `PLAYER_GAP_S` = 1.0,
  `TRACKLET_S` = 20.0, `CROWD_GAIN` = 0.5 are guesses, and the 46 slots / 20.5 objects / 59.3% seen
  come from DATA.md. Pasting these lets us compare, not tune on labels (there are none).
- Ball step and speed, share above 40 m/s, share seen more than 1 m outside the pitch and speed when
  nearly stationary describe jitter and detection jumps, which `degrade.py` does not model.
- Conventions in the script, all printed or recorded in the JSON: runs are counted inside each
  graded half separately, so none spans halftime; runs cut by a half boundary are truncated; speeds
  use only pairs of consecutive frames where the ball is seen in both; "nearly stationary" means
  the ball moves less than 0.5 m over 25 frames (both ends seen). The 40 m/s and 1 m thresholds are
  our choices, not organiser values.
