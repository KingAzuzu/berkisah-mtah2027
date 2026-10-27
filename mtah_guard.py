"""Call refuse_under_claude() at the top of every function that loads the competition data (T&C section 8)."""
import os


def refuse_under_claude():
    if os.environ.get("CLAUDECODE"):
        raise SystemExit("Refusing to load the competition data inside a Claude Code session (T&C section 8). "
                         "Run this command in your own terminal.")
