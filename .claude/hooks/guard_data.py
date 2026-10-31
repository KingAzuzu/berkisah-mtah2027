"""PreToolUse hook: stop Claude Code from touching the competition data (T&C section 8).

Claude Code passes the tool call as JSON on stdin. Exit code 2 blocks the call and shows stderr to
Claude. Only paths and commands are checked, not file contents, so code that mentions BAR-ATH can
still be written. Input it cannot parse is blocked too: a guard that fails open is no guard.
"""
import json
import os
import re
import sys

PATTERNS = [r"bar-ath", r"_atd\.(txt|xml)", r"_homography\.csv", r"smart_tagging\.xml", r"\.mp4\b",
            r"evaluation/results\b"]
FIELDS = {"Bash": ["command"], "PowerShell": ["command"], "Read": ["file_path"], "Write": ["file_path"],
          "Edit": ["file_path"], "MultiEdit": ["file_path"], "NotebookEdit": ["notebook_path"],
          "Grep": ["path", "glob"], "Glob": ["path", "pattern"]}


def norm(s):
    s = re.sub(r"/+", "/", s.replace("\\", "/")).lower()
    return re.sub(r"(^|[\s'\"=])/([a-z])/", r"\1\2:/", s)  # /d/x (Git Bash) -> d:/x


def data_dirs():
    dirs = [os.environ.get("MTAH_DATA_DIR", ""), "D:/mtah-data", "D:/barca/train"]
    return [norm(d).rstrip("/") for d in dirs if d]


def block(reason):
    print("Blocked by guard_data.py: " + reason + " T&C section 8 forbids sending the competition data "
          "to a hosted LLM. Ask the human to run it in their own terminal and paste back aggregate "
          "numbers only.", file=sys.stderr)
    return 2


def main():
    try:
        call = json.load(sys.stdin)
    except ValueError:
        return block("could not parse the tool call, so it cannot be checked.")
    tool_input = call.get("tool_input") or {}
    text = "\n".join(norm(str(tool_input.get(f) or "")) for f in FIELDS.get(call.get("tool_name"), []))
    hits = [d for d in data_dirs() if d in text] + [p for p in PATTERNS if re.search(p, text)]
    if hits:
        return block("this call touches the competition data (" + ", ".join(hits) + ").")
    return 0


if __name__ == "__main__":
    sys.exit(main())
