"""
Milestone 19 � Live Infinite Round Generator (CALIBRATED WITH REAL DATASET)
Retail Discount-Recommendation Bandit Project

Calibrated against your real 73,100-row dataset:
  - Price: mean=52.76, std=27.48, min=11.13, max=124.18
  - Categories: Electronics (29.3%), Groceries (24.6%), Furniture (20.0%), Toys (15.4%), Clothing (10.8%)
  - Promotion rate: 20.57%
  - Seasonality: Spring (25.2%), Summer (25.2%), Autumn (25.0%), Winter (24.6%)
  - Regions: West (34.6%), South (22.0%), North (21.9%), East (21.5%)

Run:
    python src/milestone19_live_generator.py
"""

import itertools
import random
import statistics
from collections import Counter
from typing import Dict, Generator, List, Tuple

from round_generator import (
    ASSUMED_MARGIN_RATE,
    Customer,
    build_demo_registry,
)
from guardrails import get_eligible_actions


# ---------------------------------------------------------------------------
# 1. REAL CALIBRATED DISTRIBUTIONS FROM 73,100-ROW DATASET
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS = {
    "Electronics": 0.2928,
    "Groceries": 0.2459,
    "Furniture": 0.1995,
    "Toys": 0.1540,
    "Clothing": 0.1078,
}

CATEGORY_PRICE_STATS = {
    "Electronics": {"mean": 66.40, "std": 24.15, "min": 11.66},
    "Groceries": {"mean": 32.45, "std": 17.11, "min": 11.13},
    "Furniture": {"mean": 69.84, "std": 29.12, "min": 11.73},
    "Toys": {"mean": 49.53, "std": 21.09, "min": 11.35},
    "Clothing": {"mean": 35.02, "std": 15.30, "min": 11.70},
}

REGION_WEIGHTS = {
    "West": 0.3465,
    "South": 0.2199,
    "North": 0.2186,
    "East": 0.2150,
}

SEASONALITY_WEIGHTS = {
    "Spring": 0.2517,
    "Summer": 0.2517,
    "Autumn": 0.2503,
    "Winter": 0.2462,
}

PROMOTION_RATE = 0.2057  # Bernoulli p for Promotion=1 (Real dataset: 20.57%)


def _weighted_choice(weights: Dict[str, float]) -> str:
    labels = list(weights.keys())
    probs = list(weights.values())
    return random.choices(labels, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# 2. FRESH CONTEXT SAMPLER
# ---------------------------------------------------------------------------

def sample_live_context() -> Dict:
    category = _weighted_choice(CATEGORY_WEIGHTS)
    stats = CATEGORY_PRICE_STATS[category]

    price = random.gauss(stats["mean"], stats["std"])
    price = max(stats["min"], price)  # never let category price go below its real floor
    price = round(price, 2)

    cost_price = round(price * (1 - ASSUMED_MARGIN_RATE), 2)

    # Competitor pricing from real data: tightly correlated with own price, mean ratio approx 1.0
    competitor_pricing = round(price * random.uniform(0.92, 1.08), 2)

    return {
        "Price": price,
        "CostPrice": cost_price,
        "Category": category,
        "Region": _weighted_choice(REGION_WEIGHTS),
        "InventoryLevel": random.randint(10, 300),
        "CompetitorPricing": competitor_pricing,
        "Seasonality": _weighted_choice(SEASONALITY_WEIGHTS),
        "Promotion": 1 if random.random() < PROMOTION_RATE else 0,
        "DemandIndex": round(random.uniform(0, 1), 3),
        # CustomerPriceSensitivity: from real dataset (mean=1.0025, std=0.2138, range [0.55,1.55])
        # Higher = customer more responsive to discounts (bandit uses this to calibrate offer)
        "CustomerPriceSensitivity": round(max(0.55, min(1.55, random.gauss(1.0025, 0.2138))), 4),
    }


# ---------------------------------------------------------------------------
# 3. REGISTRY EVOLUTION
# ---------------------------------------------------------------------------

def _advance_or_create_customer(
    registry: List[Customer],
    next_id: "itertools.count",
    p_new_customer: float,
    p_purchase_on_visit: float,
) -> Customer:
    if random.random() < p_new_customer:
        customer = Customer(
            CustomerID=next(next_id),
            PurchaseCount=0,
            RecencyDays=0,
            CurrentBasketValue=round(random.uniform(50, 500), 2),
        )
        registry.append(customer)
        return customer

    customer = random.choice(registry)
    if random.random() < p_purchase_on_visit:
        customer.PurchaseCount += 1
        customer.RecencyDays = 0
    else:
        customer.RecencyDays += random.randint(1, 14)
    return customer


# ---------------------------------------------------------------------------
# 4. LIVE INFINITE ROUND GENERATOR
# ---------------------------------------------------------------------------

def generate_live_rounds(
    registry: List[Customer],
    p_new_customer: float = 0.05,
    p_purchase_on_visit: float = 0.30,
    margin_floor: float = None,
) -> Generator[Tuple[Dict, Customer, List[int]], None, None]:
    start_id = max((c.CustomerID for c in registry), default=0) + 1
    next_id = itertools.count(start=start_id)

    while True:
        customer = _advance_or_create_customer(registry, next_id, p_new_customer, p_purchase_on_visit)
        context = sample_live_context()

        if margin_floor is None:
            eligible = get_eligible_actions(context, customer)
        else:
            eligible = get_eligible_actions(context, customer, margin_floor=margin_floor)

        yield context, customer, eligible


# ---------------------------------------------------------------------------
# 5. VALIDATION CHECK -- 10,000 fresh rounds vs REAL HISTORICAL TARGETS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    registry = build_demo_registry(n=50)

    N = 10_000
    gen = generate_live_rounds(registry)

    prices, categories, promotions = [], [], []
    for _ in range(N):
        context, customer, eligible = next(gen)
        prices.append(context["Price"])
        categories.append(context["Category"])
        promotions.append(context["Promotion"])

    mean_price_generated = statistics.mean(prices)
    promotion_rate_generated = sum(promotions) / N
    category_counts = Counter(categories)
    category_dist_generated = {c: category_counts[c] / N for c in CATEGORY_WEIGHTS}
    target_mean_price = sum(
        CATEGORY_WEIGHTS[c] * CATEGORY_PRICE_STATS[c]["mean"] for c in CATEGORY_WEIGHTS
    )

    print(f"Generated {N:,} fresh rounds. Registry grew from 50 to {len(registry)} customers.\n")

    print(f"{'Metric':<22}{'Generated':<15}{'Target (Real Dataset)'}")
    print("-" * 55)
    print(f"{'Mean Price':<22}{mean_price_generated:<15.2f}{target_mean_price:.2f}")
    print(f"{'Promotion rate':<22}{promotion_rate_generated:<15.3f}{PROMOTION_RATE:.3f}")
    print()

    print("Category distribution (generated vs Real Dataset):")
    for c in CATEGORY_WEIGHTS:
        print(f"  {c:<13} generated={category_dist_generated[c]:.3f}   target={CATEGORY_WEIGHTS[c]:.3f}")

    print(f"\nMin price generated: {min(prices):.2f}  (must be >= 0, and >= that category's floor)")
    print(f"Max price generated: {max(prices):.2f}")
    assert min(prices) >= 0, "Negative price generated -- something is broken"
