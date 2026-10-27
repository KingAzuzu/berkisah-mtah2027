"""WS2 control: the plain safety net with a tighter R_CLOSE, to separate 'smaller radius' from the rules.

    python -m possession.rules_ablate_safety
"""
import json
from pathlib import Path

from .lomo import _reference_predict, evaluate_grid

if __name__ == "__main__":
    base = json.loads(Path(__file__).with_name("params.json").read_text())
    grid = {**{k: [v] for k, v in base.items()}, "R_CLOSE": [1.0, 1.5, 3.0]}
    evaluate_grid(_reference_predict, grid, name="rules_ablate_safety")
