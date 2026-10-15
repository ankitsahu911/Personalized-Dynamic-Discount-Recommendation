"""
Milestone 31 - Verification Suite for Saved Models
Retail Discount-Recommendation Bandit Project

Verifies:
    1. Round-trip equality (np.array_equal on weights)
    2. Cross-process test (100% deterministic decision match in a clean subprocess)
    3. Thompson Sampling reproducibility check
    4. Continue-learning test (update() works on loaded models)
    5. Fresh-context standalone recommendation test (no training code/dataset needed)
"""

import json
import os
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_io import load_linucb, load_thompson
from linucb import build_context_vector, DisjointLinUCB
from thompson_sampling import ThompsonSampling
from round_generator import Customer
from training_rounds import load_training_rounds

LINUCB_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "linucb_v1")
THOMPSON_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "thompson_v1")


def run_verification():
    print("=" * 70)
    print("Milestone 31 -- Verification Suite for Saved Models")
    print("=" * 70)

    # 1. Round-Trip Equality
    print("\n--- Check 1: Round-Trip Equality ---")
    linucb = load_linucb(LINUCB_PREFIX)
    thompson = load_thompson(THOMPSON_PREFIX)

    raw_linucb = np.load(f"{LINUCB_PREFIX}.npz")
    raw_thompson = np.load(f"{THOMPSON_PREFIX}.npz")

    arms = [0, 5, 10, 15, 20]
    linucb_equal = True
    for i, a in enumerate(arms):
        if not np.array_equal(linucb.A[a], raw_linucb["A"][i]):
            linucb_equal = False
        if not np.array_equal(linucb.b[a], raw_linucb["b"][i]):
            linucb_equal = False

    ts_equal = True
    for i, a in enumerate(arms):
        if not np.array_equal(thompson.mu[a], raw_thompson["mu"][i]):
            ts_equal = False
        if not np.array_equal(thompson.Sigma[a], raw_thompson["Sigma"][i]):
            ts_equal = False

    print(f"  LinUCB A & b exact array match : {linucb_equal} (must be True)")
    print(f"  Thompson mu & Sigma exact match : {ts_equal} (must be True)")
    assert linucb_equal and ts_equal, "Round-trip equality failed!"

    # 2. Cross-Process Test on 100 contexts
    print("\n--- Check 2: Cross-Process Reproducibility (Subprocess Test) ---")
    rounds = load_training_rounds()[:100]

    # In-memory LinUCB choices
    in_memory_choices = []
    for r in rounds:
        ch = linucb.select_action(r["context"], r["customer"], r["eligible_discounts"])
        in_memory_choices.append(ch)

    # Subprocess evaluation script
    eval_code = f"""
import sys, os, pickle
sys.path.insert(0, r'{os.path.dirname(os.path.abspath(__file__))}')
from model_io import load_linucb
from training_rounds import load_training_rounds

model = load_linucb(r'{LINUCB_PREFIX}')
rounds = load_training_rounds()[:100]
choices = [model.select_action(r['context'], r['customer'], r['eligible_discounts']) for r in rounds]
print(','.join(map(str, choices)))
"""
    result = subprocess.run([sys.executable, "-c", eval_code], capture_output=True, text=True, check=True)
    subprocess_choices = list(map(int, result.stdout.strip().split(",")))

    match_rate = sum(a == b for a, b in zip(in_memory_choices, subprocess_choices)) / len(in_memory_choices)
    print(f"  LinUCB Cross-process decision agreement: {match_rate*100:.1f}% ({match_rate == 1.0})")
    assert match_rate == 1.0, "Subprocess LinUCB choices did not match in-memory model!"

    # 3. Thompson Reproducibility Check
    print("\n--- Check 3: Thompson Sampling Reproducibility ---")
    np.random.seed(999)
    ts_choices_1 = [thompson.select_action(r["context"], r["customer"], r["eligible_discounts"]) for r in rounds]

    thompson_reloaded = load_thompson(THOMPSON_PREFIX)
    np.random.seed(999)
    ts_choices_2 = [thompson_reloaded.select_action(r["context"], r["customer"], r["eligible_discounts"]) for r in rounds]

    ts_match = sum(a == b for a, b in zip(ts_choices_1, ts_choices_2)) / len(ts_choices_1)
    print(f"  Thompson Sampling with fixed seed agreement: {ts_match*100:.1f}% ({ts_match == 1.0})")
    assert ts_match == 1.0, "Thompson reproducibility check failed!"

    # 4. Continue-Learning Check
    print("\n--- Check 4: Continue-Learning Test (update() on loaded models) ---")
    sample_r = rounds[0]
    try:
        linucb.update(arm=10, raw_context=sample_r["context"], customer=sample_r["customer"], reward=5.50)
        thompson.update(arm=10, raw_context=sample_r["context"], customer=sample_r["customer"], reward=5.50)
        print("  Successfully called update() on both loaded models without error.")
    except Exception as e:
        print(f"  Error updating loaded models: {e}")
        raise e

    # 5. Fresh-Context Standalone Test
    print("\n--- Check 5: Fresh-Context Standalone Recommendation ---")
    # Freshly typed realistic context with NO training data
    fresh_context = {
        "Price": 89.99,
        "CostPrice": 49.50,
        "Category": "Electronics",
        "Region": "West",
        "InventoryLevel": 140,
        "CompetitorPricing": 92.00,
        "Seasonality": "Winter",
        "Promotion": 1,
        "DemandIndex": 0.45,
        "CustomerPriceSensitivity": 1.25,
    }
    fresh_customer = Customer(
        CustomerID=9999,
        PurchaseCount=15,
        RecencyDays=3,
        LoyaltyTier="Loyal",
        CurrentBasketValue=180.0,
    )
    eligible = [10, 15, 20]  # Loyal tier eligible set

    rec_linucb = linucb.select_action(fresh_context, fresh_customer, eligible)
    rec_ts = thompson.select_action(fresh_context, fresh_customer, eligible)

    print(f"  Input Context : Price=Rs {fresh_context['Price']}, Category={fresh_context['Category']}, LoyaltyTier={fresh_customer.LoyaltyTier}")
    print(f"  Eligible Arms : {eligible}")
    print(f"  Loaded LinUCB Recommendation        : {rec_linucb}% discount")
    print(f"  Loaded Thompson Recommendation     : {rec_ts}% discount")
    assert rec_linucb in eligible and rec_ts in eligible

    print("\n" + "=" * 70)
    print("ALL MILESTONE 31 VERIFICATION CHECKS PASSED (5 / 5)")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
