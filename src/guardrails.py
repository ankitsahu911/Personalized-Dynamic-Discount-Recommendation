"""
Milestone 17 � Combined get_eligible_actions(context, customer)
Retail Discount-Recommendation Bandit Project

This imports the real functions from Milestone 15 and Milestone 16 (not
reinvented copies) and chains them together. From this point on, nothing
downstream (baselines, bandits) should ever call get_tier(),
eligible_discounts(), or margin_floor_guardrail() directly -- everything
should go through get_eligible_actions().

ONE DELIBERATE DEVIATION FROM THE MILESTONE DOC, FLAGGED:
The doc's prompt template writes the signature as
    get_eligible_actions(context, tier_boundaries=None, margin_floor=None)
i.e. context only. But tier lookup needs PurchaseCount, which in our
Milestone 15 design lives on the Customer object, not inside the
situational context dict (Price/Category/etc.). Rather than silently
smuggling PurchaseCount into the context dict (which would blur what
"context" means for later milestones), I added customer as an explicit
second parameter:
    get_eligible_actions(context, customer, tier_boundaries=..., margin_floor=...)
If your real project keeps everything in one merged dict instead of a
separate Customer object, tell me and I'll collapse this back to a single
`context` argument.

Requires: milestone15_round_generator.py and milestone16_margin_guardrail.py
in the SAME folder (both importable as plain modules).

Run:
    python milestone17_get_eligible_actions.py
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


from round_generator import (
    TIER_BOUNDARIES,
    TIER_DISCOUNTS,
    Customer,
    get_tier,
    eligible_discounts,
)
from margin_guardrail import margin_floor_guardrail


# ---------------------------------------------------------------------------
# CONFIG
#    Sensible defaults, exposed as parameters below -- NOT buried inside
#    the function body -- so Phase 8's live app controls can override them
#    at runtime without editing this file.
# ---------------------------------------------------------------------------

DEFAULT_TIER_BOUNDARIES = TIER_BOUNDARIES
DEFAULT_MARGIN_FLOOR = 0.0  # >>> INVENTED - REPLACE WITH YOUR REAL BUSINESS FLOOR <<<
# 0.0 = "never sell at or below cost." No real floor value was given to me,
# so this is the most conservative placeholder that still makes business
# sense. Swap it for whatever your actual finance/margin requirement is.


def get_eligible_actions(
    context: Dict,
    customer: Customer,
    tier_boundaries: Dict = DEFAULT_TIER_BOUNDARIES,
    margin_floor: float = DEFAULT_MARGIN_FLOOR,
) -> List[int]:
    """
    THE single source of truth for "what discounts are allowed right now".
    Combines the tier guardrail (Milestone 15) and margin guardrail
    (Milestone 16) with AND logic: a discount only survives if it passes
    BOTH.

    Args:
        context: dict with at least "Price" and "CostPrice".
        customer: Customer with at least PurchaseCount.
        tier_boundaries: purchase-count cutoffs for New/Regular/Loyal.
            Defaults to Milestone 15's TIER_BOUNDARIES, but adjustable.
        margin_floor: minimum acceptable per-unit margin in currency
            units. Defaults to DEFAULT_MARGIN_FLOOR above, but adjustable.

    Returns:
        Final filtered list of eligible discount percentages. Empty list
        if nothing survives -- never raises.
    """
    tier = get_tier(customer.PurchaseCount, tier_boundaries)
    tier_eligible = eligible_discounts(tier, TIER_DISCOUNTS)

    surviving, warning_flag = margin_floor_guardrail(context, tier_eligible, margin_floor)

    if warning_flag:
        logger.warning(
            "[get_eligible_actions] CustomerID=%s (tier=%s) has NO eligible discounts "
            "this round — margin floor=%.2f eliminated all of %s given Price=%s, CostPrice=%s.",
            customer.CustomerID, tier, margin_floor, tier_eligible,
            context.get('Price'), context.get('CostPrice'),
        )

    return surviving


# ---------------------------------------------------------------------------
# TEST -- the 2 scenarios the milestone spec asks for
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario 1: brand-new customer, healthy-margin product
    # Expect: [0, 5], both surviving (tier caps it at [0,5]; margin is fine)
    new_customer = Customer(CustomerID=101, PurchaseCount=0, RecencyDays=2)
    healthy_context = {"Price": 1000.0, "CostPrice": 400.0}
    result_1 = get_eligible_actions(healthy_context, new_customer, margin_floor=50.0)
    print(f"Scenario 1 -- New customer, healthy margin:")
    print(f"  tier eligible would be [0, 5] -> after margin filter: {result_1}\n")

    # Scenario 2: Loyal customer, thin-margin product
    # Tier alone allows [10, 15, 20]; margin floor should narrow that down.
    loyal_customer = Customer(CustomerID=202, PurchaseCount=15, RecencyDays=10)
    thin_context = {"Price": 1000.0, "CostPrice": 820.0}
    result_2 = get_eligible_actions(thin_context, loyal_customer, margin_floor=50.0)
    print(f"Scenario 2 -- Loyal customer, thin margin:")
    print(f"  tier eligible would be [10, 15, 20] -> after margin filter: {result_2}")
