"""Live OpenRouter check: Mercury on the town-board sample transcript."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.town_board_eval import run_engine_sample

if __name__ == "__main__":
    raise SystemExit(run_engine_sample("mercury"))
