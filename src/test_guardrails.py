"""
Milestone 18 � Unit Tests for get_eligible_actions()
Retail Discount-Recommendation Bandit Project

These test the ACTUAL get_eligible_actions() from
milestone17_get_eligible_actions.py -- not a mock, not a reimplementation.
If a test fails, the bug is either in Milestone 17's code, or in an
assumption baked into the test itself -- each docstring explains exactly
what that test is trying to prove, so you can tell which.

Requires milestone15_round_generator.py, milestone16_margin_guardrail.py,
and milestone17_get_eligible_actions.py all in the SAME folder as this file.

Run:
    pytest milestone18_test_guardrails.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from round_generator import Customer
from guardrails import get_eligible_actions


def test_brand_new_customer():
    """A brand-new customer (PurchaseCount=0) must NEVER be offered
    10/15/20, no matter how healthy the product's margin is -- the tier
    guardrail alone should cap them at [0, 5]."""
    customer = Customer(CustomerID=1, PurchaseCount=0, RecencyDays=5)
    healthy_context = {"Price": 1000.0, "CostPrice": 100.0}  # very healthy margin

    result = get_eligible_actions(healthy_context, customer, margin_floor=0.0)

    assert set(result).issubset({0, 5}), f"New customer got out-of-tier discounts: {result}"
    assert 10 not in result and 15 not in result and 20 not in result


def test_loyal_customer_healthy_margin():
    """A Loyal customer (PurchaseCount>=10) on a healthy-margin product with
    a generous floor should get the FULL [10, 15, 20] set -- neither
    guardrail should wrongly narrow it when it shouldn't."""
    customer = Customer(CustomerID=2, PurchaseCount=15, RecencyDays=5)
    healthy_context = {"Price": 1000.0, "CostPrice": 100.0}

    result = get_eligible_actions(healthy_context, customer, margin_floor=0.0)

    assert set(result) == {10, 15, 20}, f"Expected full Loyal set, got {result}"


def test_loyal_customer_thin_margin():
    """Proves the two guardrails combine with AND logic, NOT that margin is
    ignored once tier says yes. Tier alone would allow [10, 15, 20] here --
    but a near-zero-margin product with a strict floor should let the
    margin guardrail override the tier guardrail and eliminate ALL of
    them."""
    customer = Customer(CustomerID=3, PurchaseCount=12, RecencyDays=5)
    thin_context = {"Price": 300.0, "CostPrice": 295.0}  # only 5 margin at full price

    tier_only_would_allow = {10, 15, 20}
    result = get_eligible_actions(thin_context, customer, margin_floor=10.0)

    assert result == [], f"Expected margin guardrail to override tier and return [], got {result}"
    assert set(result) != tier_only_would_allow, (
        "Margin guardrail was not applied -- the raw tier set passed through unfiltered"
    )


def test_boundary_purchase_counts():
    """Off-by-one errors are the most common bug in tier logic.
    PurchaseCount exactly 3 must be Regular (NOT New), and exactly 10 must
    be Loyal (NOT Regular). Uses a generous margin floor so only tier logic
    is on trial here."""
    healthy_context = {"Price": 1000.0, "CostPrice": 100.0}

    regular_boundary_customer = Customer(CustomerID=4, PurchaseCount=3, RecencyDays=5)
    result_regular = get_eligible_actions(healthy_context, regular_boundary_customer, margin_floor=0.0)
    assert set(result_regular) == {5, 10, 15}, f"PurchaseCount=3 should be Regular [5,10,15], got {result_regular}"
    assert 0 not in result_regular, "PurchaseCount=3 was wrongly treated as New"

    loyal_boundary_customer = Customer(CustomerID=5, PurchaseCount=10, RecencyDays=5)
    result_loyal = get_eligible_actions(healthy_context, loyal_boundary_customer, margin_floor=0.0)
    assert set(result_loyal) == {10, 15, 20}, f"PurchaseCount=10 should be Loyal [10,15,20], got {result_loyal}"
    assert 5 not in result_loyal, "PurchaseCount=10 was wrongly treated as Regular"


def test_impossible_margin_floor(capsys):
    """A floor so strict that even the smallest available discount fails
    should return an empty list WITHOUT raising an exception, and should
    log a warning (per Milestone 17's design) rather than failing
    silently."""
    customer = Customer(CustomerID=6, PurchaseCount=0, RecencyDays=5)  # New -> includes 0%
    context = {"Price": 1000.0, "CostPrice": 100.0}
    impossible_floor = 999999.0  # nothing can clear this

    result = get_eligible_actions(context, customer, margin_floor=impossible_floor)
    captured = capsys.readouterr()

    assert result == [], f"Expected empty list, got {result}"
    assert "WARNING" in captured.out, "Expected a warning to be printed/logged when nothing survives"
