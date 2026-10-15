"""
Milestone 16 - Margin-Floor Guardrail
Retail Discount-Recommendation Bandit Project

This one is mostly self-contained (per the project doc, Milestone 16
doesn't need your full simulator.py -- just your real column names for
price and cost).

ASSUMPTION FLAGGED:
I'm assuming your context has fields named "Price" and "CostPrice" (in Rs),
matching the doc's description: CostPrice = Price x (1 - ASSUMED_MARGIN_RATE),
already engineered in an earlier feature-engineering step. If your real
column names differ (e.g. "SellingPrice", "unit_cost", different casing),
tell me the actual names and I'll adjust -- the logic underneath is
unaffected either way.

NOTE FOR MILESTONE 17: milestone15_round_generator.py's sample_context()
does NOT currently produce a "CostPrice" field, so you can't chain the two
files together yet as-is. That's expected -- Milestone 17 is where they
get combined, and I'll add CostPrice sampling then (or better: point it at
your real CostPrice column if you share it).

Run:
    python src/milestone16_margin_guardrail.py
"""

from typing import Dict, List, Tuple


def margin_floor_guardrail(
    context: Dict,
    candidate_discounts: List[float],
    floor: float,
) -> Tuple[List[float], bool]:
    """
    Filters candidate_discounts down to only those that keep per-unit
    margin at or above `floor`, after the discount is applied.

    Args:
        context: dict with at least "Price" and "CostPrice" (Rs).
        candidate_discounts: e.g. [0, 5, 10, 15, 20] (percent).
        floor: minimum acceptable per-unit margin in Rs. Required, no
            hardcoded default inside the function -- this is meant to be
            a live, adjustable setting (e.g. an app slider) supplied by
            the caller, per the milestone spec.

    Returns:
        (surviving_discounts, warning_flag)
        warning_flag is True only when NOTHING survives (even 0% fails
        the floor -- can happen on a genuinely thin-margin product). The
        caller decides what to do next (fall back to 0% anyway, or drop
        the product from this round) -- this function never makes that
        call silently.
    """
    price = context["Price"]
    cost_price = context["CostPrice"]

    surviving = []
    for discount in candidate_discounts:
        discount_fraction = discount / 100.0
        selling_price = price * (1 - discount_fraction)
        unit_margin_after_discount = selling_price - cost_price
        if unit_margin_after_discount >= floor:
            surviving.append(discount)

    warning_flag = len(surviving) == 0
    if warning_flag:
        print(
            f"[margin_floor_guardrail] WARNING: no discount survives "
            f"floor={floor} for Price={price}, CostPrice={cost_price}. "
            f"Even 0% discount fails."
        )

    return surviving, warning_flag


# ---------------------------------------------------------------------------
# TEST -- the 3 example rows the milestone spec asks for
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    candidate_discounts = [0, 5, 10, 15, 20]

    # (a) Healthy-margin product, low floor -> all 5 should survive
    healthy = {"Price": 1000.0, "CostPrice": 400.0}
    low_floor = 50.0
    result_a, warn_a = margin_floor_guardrail(healthy, candidate_discounts, low_floor)
    print(f"(a) Healthy margin (Price=1000, Cost=400), floor={low_floor}")
    print(f"    survives={result_a}  warning={warn_a}\n")

    # (b) Thin-margin product, moderate floor -> only 0% and 5% should survive
    thin = {"Price": 500.0, "CostPrice": 440.0}
    moderate_floor = 30.0
    result_b, warn_b = margin_floor_guardrail(thin, candidate_discounts, moderate_floor)
    print(f"(b) Thin margin (Price=500, Cost=440), floor={moderate_floor}")
    print(f"    survives={result_b}  warning={warn_b}\n")

    # (c) Razor-thin-margin product, strict floor -> even 0% should fail
    razor_thin = {"Price": 300.0, "CostPrice": 290.0}
    strict_floor = 20.0
    result_c, warn_c = margin_floor_guardrail(razor_thin, candidate_discounts, strict_floor)
    print(f"(c) Razor-thin margin (Price=300, Cost=290), floor={strict_floor}")
    print(f"    survives={result_c}  warning={warn_c}")
