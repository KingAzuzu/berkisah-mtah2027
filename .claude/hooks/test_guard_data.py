"""Check the data guard. Run: python .claude/hooks/test_guard_data.py  (every line must say ok)."""
import json
import os
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).with_name("guard_data.py")
CASES = [  # (expected exit code, tool name, tool input, MTAH_DATA_DIR)
    (2, "Read", {"file_path": r"D:\mtah-data\tactical\x.txt"}, None),
    (2, "Bash", {"command": "cat /d/mtah-data/tactical/BAR-ATH_tactical_atd.txt | head"}, None),
    (2, "PowerShell", {"command": r"Get-Content D:\MTAH-DATA\x"}, None),
    (2, "Bash", {"command": r"dir E:\comp\tactical"}, r"E:\comp"),
    (2, "Glob", {"pattern": "**/*.mp4"}, None),
    (2, "Grep", {"pattern": "H", "path": "evaluation/results"}, None),
    (2, "Read", {"file_path": "evaluation/results/team.csv"}, None),
    (0, "Bash", {"command": "python -m pytest tests -q"}, None),
    (0, "Write", {"file_path": "possession/io.py", "content": "BAR-ATH loader"}, None),
    (0, "Read", {"file_path": "D:/repo/games.py"}, None),
    (0, "Read", {"file_path": "idsse-tests/data/DFL_04_03_positions_raw_observed.xml"}, None),
]

failed = 0
for want, tool, tool_input, data_dir in CASES:
    env = dict(os.environ)
    env.pop("MTAH_DATA_DIR", None)
    if data_dir:
        env["MTAH_DATA_DIR"] = data_dir
    stdin = json.dumps({"tool_name": tool, "tool_input": tool_input})
    got = subprocess.run([sys.executable, str(HOOK)], input=stdin, text=True, capture_output=True, env=env).returncode
    failed += got != want
    print("ok  " if got == want else "FAIL", want, got, tool, tool_input)
bad = subprocess.run([sys.executable, str(HOOK)], input="not json", text=True, capture_output=True).returncode
failed += bad != 2
print("ok  " if bad == 2 else "FAIL", "unparsable input blocked:", bad)
sys.exit(1 if failed else 0)
