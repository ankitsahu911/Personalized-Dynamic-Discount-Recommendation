"""
Golden file regression test.
Verifies that 20 fixed benchmark recommendations are bit-for-bit identical.
"""

import json
from pathlib import Path
import pytest

from app.recommender import recommend

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_outputs.json"


def test_golden_file_parity():
    assert GOLDEN_PATH.exists(), f"Missing golden file at {GOLDEN_PATH}"

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    assert len(golden_cases) == 20

    for idx, item in enumerate(golden_cases):
        inp = item["inputs"]
        rec = recommend(inp)

        assert rec.chosen_discount == item["chosen_discount"], f"Case {idx} chosen discount mismatch"
        assert rec.eligible_discounts == item["eligible_discounts"], f"Case {idx} eligible discounts mismatch"
        assert rec.discounted_price == item["discounted_price"], f"Case {idx} discounted price mismatch"
        assert rec.unit_margin_after_discount == item["unit_margin"], f"Case {idx} unit margin mismatch"
        assert rec.why_pick_text == item["why_pick_text"], f"Case {idx} narrative mismatch"
