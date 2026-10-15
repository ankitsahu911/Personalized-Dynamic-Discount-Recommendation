"""
Unit and property tests for business guardrails and eligibility filtering.
"""

import random
import pytest
import numpy as np

from app.recommender import Rules, evaluate_guardrails, recommend, CustomerSnapshot
from src.guardrails import get_eligible_actions
from src.round_generator import Customer


def test_guardrails_property_500_random_checks():
    """Property test: evaluate_guardrails must match get_eligible_actions over 500 random inputs."""
    random.seed(42)
    for _ in range(500):
        p = round(random.uniform(15.0, 140.0), 2)
        cp = round(p * random.uniform(0.40, 0.85), 2)
        pc = random.randint(0, 25)
        floor = round(random.uniform(0.0, 15.0), 2)

        rules = Rules(margin_floor=floor)
        app_elig, _, _, _ = evaluate_guardrails(p, cp, pc, rules)

        ctx = {"Price": p, "CostPrice": cp}
        cust = Customer(CustomerID=1, PurchaseCount=pc, RecencyDays=10)
        src_elig = get_eligible_actions(ctx, cust, margin_floor=floor)

        assert app_elig == src_elig, f"Mismatch at Price={p}, CostPrice={cp}, PC={pc}, Floor={floor}: {app_elig} vs {src_elig}"


def test_new_customer_boundary():
    """Customer with PurchaseCount 0, 1, 2 must never get discount above 5% under default rules."""
    for pc in [0, 1, 2]:
        rec = recommend({"Price": 100.0, "PurchaseCount": pc})
        assert rec.chosen_discount in (0, 5)
        for arm in rec.arms_table:
            if arm.discount in (10, 15, 20):
                assert arm.status.startswith("blocked")


def test_boundary_transitions():
    """Check boundaries at 2 (New), 3 (Regular), 9 (Regular), 10 (Loyal)."""
    r2 = recommend({"Price": 100.0, "PurchaseCount": 2})
    assert r2.tier_for_rules == "New"
    assert set(r2.eligible_discounts).issubset({0, 5})

    r3 = recommend({"Price": 100.0, "PurchaseCount": 3})
    assert r3.tier_for_rules == "Regular"
    assert set(r3.eligible_discounts).issubset({5, 10, 15})

    r9 = recommend({"Price": 100.0, "PurchaseCount": 9})
    assert r9.tier_for_rules == "Regular"
    assert set(r9.eligible_discounts).issubset({5, 10, 15})

    r10 = recommend({"Price": 100.0, "PurchaseCount": 10})
    assert r10.tier_for_rules == "Loyal"
    assert set(r10.eligible_discounts).issubset({10, 15, 20})


def test_high_margin_floor_empty_set():
    """When margin floor is above product margin at 0% discount, chosen_discount must be None."""
    # Price = 50, Cost = 30. Max margin is 20 Rs. Floor = 25 Rs.
    rec = recommend({"Price": 50.0, "PurchaseCount": 5}, rules=Rules(margin_floor=25.0))
    assert rec.chosen_discount is None
    assert len(rec.eligible_discounts) == 0


def test_extrapolation_flag_on_custom_rules():
    """Changing Regular boundary to 1 for PC=2 makes 10% eligible and flags it as extrapolated."""
    # Customer has 2 purchases (normally New: 0%, 5%).
    # Custom rule sets Regular starting at 1 -> now Regular: eligible 5%, 10%, 15%.
    # Discounts 10% and 15% are extrapolated!
    custom_rules = Rules(regular_cutoff=1, loyal_cutoff=10)
    rec = recommend({"Price": 100.0, "PurchaseCount": 2}, rules=custom_rules)

    assert rec.tier_for_rules == "Regular"
    assert 10 in rec.eligible_discounts

    arm_10 = next(a for a in rec.arms_table if a.discount == 10)
    assert arm_10.extrapolated is True
