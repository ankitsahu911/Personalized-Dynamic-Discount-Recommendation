"""
Unit tests for pure recommender logic engine.
"""

import hashlib
import pickle
import pytest
import numpy as np

from app.recommender import (
    Rules,
    load_model,
    make_context,
    recommend,
    recommend_scenarios,
    compute_response_curve,
)


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_make_context_mapping():
    inputs = {
        "Price": 120.0,
        "DemandLevel": "High",
        "Promotion": True,
        "PurchaseCount": 5,
        "CurrentBasketValue": 250.0,
    }
    ctx, cust = make_context(inputs)
    assert ctx["Price"] == 120.0
    assert ctx["CostPrice"] == 72.0  # 120 * 0.60
    assert ctx["DemandIndex"] == 0.80
    assert ctx["demand_index"] == 2
    assert ctx["Promotion"] == 1
    assert cust.PurchaseCount == 5
    assert cust.LoyaltyTier == "Regular"


def test_score_breakdown_properties(model):
    inputs = {"Price": 80.0, "PurchaseCount": 4}
    rec = recommend(inputs, model=model)
    assert rec.chosen_discount in rec.eligible_discounts

    for arm in rec.arms_table:
        if arm.score is not None:
            # Score must equal estimate + bonus
            assert pytest.approx(arm.score, rel=1e-3) == (arm.point_estimate + arm.uncertainty_bonus)
            assert arm.uncertainty_bonus >= 0.0


def test_no_mutation(model):
    """Verify that calling recommend does not mutate weights."""
    hash_A = hashlib.sha256(model.A.tobytes()).hexdigest()
    hash_b = hashlib.sha256(model.b.tobytes()).hexdigest()

    for i in range(200):
        recommend({"Price": 30.0 + i, "PurchaseCount": i % 15}, model=model)

    assert hashlib.sha256(model.A.tobytes()).hexdigest() == hash_A
    assert hashlib.sha256(model.b.tobytes()).hexdigest() == hash_b


def test_whatif_zero_adjustment_identity(model):
    """Zero adjustment what-if must give identical recommendation."""
    base_in = {"Price": 75.0, "CompetitorPricing": 70.0, "InventoryLevel": 200}
    res = recommend_scenarios(
        base_inputs=base_in,
        whatif_deltas={
            "competitor_price_pct_change": 0.0,
            "inventory_multiplier": 1.0,
            "seasonality": "Same as base",
        },
        model=model,
    )
    assert res["base_recommendation"].chosen_discount == res["whatif_recommendation"].chosen_discount


def test_response_curve_generation(model):
    base_in = {"Price": 60.0, "CompetitorPricing": 55.0, "InventoryLevel": 150}
    curve = compute_response_curve(base_in, "CompetitorPricing", model=model)
    assert "curves" in curve
    assert len(curve["points"]) > 10
