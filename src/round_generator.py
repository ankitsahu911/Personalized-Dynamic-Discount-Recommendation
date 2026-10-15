"""
Milestone 15 - Live Round-Generation Loop
Retail Discount-Recommendation Bandit Project

READ THIS FIRST
----------------
This is a STANDALONE reference version. It does NOT know about your real
simulator.py from Milestones 1-14, because that file wasn't shared with me.

Every block marked:
    # >>> INVENTED - REPLACE WITH YOUR REAL VERSION <<<
is a placeholder that satisfies the milestone spec, but is NOT guaranteed to
match your actual project. Swap those for your real code before treating
Milestone 15 as done -- otherwise you're testing my simulator, not yours.

The one thing that IS the actual deliverable, and should stay structurally
the same even after you swap things in, is generate_rounds() and its
output shape: (context, customer, eligible_discounts).

Run:
    python milestone15_round_generator.py
"""

import random
from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple


# ---------------------------------------------------------------------------
# 1. CONFIG
#    Tier thresholds & discount menus live here as data, not hardcoded
#    inside functions -- Milestones 17/18 need to import/override these.
# ---------------------------------------------------------------------------

TIER_BOUNDARIES = {
    "New": (0, 3),        # PurchaseCount < 3
    "Regular": (3, 10),   # 3 <= PurchaseCount < 10
    "Loyal": (10, None),  # PurchaseCount >= 10
}

TIER_DISCOUNTS = {
    "New": [0, 5],
    "Regular": [5, 10, 15],
    "Loyal": [10, 15, 20],
}


# ---------------------------------------------------------------------------
# 2. CUSTOMER REGISTRY
#    >>> INVENTED - REPLACE WITH YOUR REAL VERSION <<<
#    Your real simulator.py almost certainly already builds a registry with
#    these fields. This is a minimal stand-in so the file runs on its own.
# ---------------------------------------------------------------------------

@dataclass
class Customer:
    CustomerID: int
    PurchaseCount: int
    RecencyDays: int
    LoyaltyTier: str = ""
    CurrentBasketValue: float = 0.0


def build_demo_registry(n: int = 50) -> List[Customer]:
    registry = []
    for cid in range(1, n + 1):
        purchase_count = random.choices(
            population=[0, 1, 2, 4, 7, 9, 10, 15, 25],
            weights=[10, 8, 8, 10, 10, 8, 10, 8, 8],
            k=1,
        )[0]
        registry.append(
            Customer(
                CustomerID=cid,
                PurchaseCount=purchase_count,
                RecencyDays=random.randint(0, 120),
                CurrentBasketValue=round(random.uniform(200, 3000), 2),
            )
        )
    return registry


# ---------------------------------------------------------------------------
# 3. TIER LOOKUP + ELIGIBLE DISCOUNTS
#    THIS is the actual function Milestone 15 needs to call. If your real
#    simulator.py already has one, delete this block and import yours --
#    generate_rounds() below only needs a callable with this signature.
# ---------------------------------------------------------------------------

def get_tier(purchase_count: int, tier_boundaries: Dict = TIER_BOUNDARIES) -> str:
    for tier, (low, high) in tier_boundaries.items():
        if high is None:
            if purchase_count >= low:
                return tier
        elif low <= purchase_count < high:
            return tier
    raise ValueError(f"PurchaseCount {purchase_count} did not match any tier")


def eligible_discounts(tier: str, tier_discounts: Dict = TIER_DISCOUNTS) -> List[int]:
    """Tier -> eligible-discount-set. Referenced by name in Milestones 17/18 --
    keep this name stable once you've wired in your real version."""
    return list(tier_discounts[tier])


# ---------------------------------------------------------------------------
# 4. CONTEXT SAMPLER
#    >>> INVENTED - REPLACE WITH YOUR REAL VERSION <<<
#    No real column distributions were provided, so these ranges are
#    reasonable guesses, NOT calibrated to your actual dataset.
# ---------------------------------------------------------------------------

CATEGORIES = ["Electronics", "Apparel", "Grocery", "Home", "Beauty"]
REGIONS = ["North", "South", "East", "West"]
SEASONALITY = ["Festive", "Regular", "Clearance"]

# >>> INVENTED - REPLACE WITH YOUR REAL VERSION <<<
# Added for Milestone 17: margin_floor_guardrail needs a CostPrice field.
# Milestone 16's doc says CostPrice = Price x (1 - ASSUMED_MARGIN_RATE) is
# already engineered in your real dataset from an earlier step. I don't
# have your real ASSUMED_MARGIN_RATE or CostPrice column, so this is a
# placeholder rate -- swap it for your real value/column.
ASSUMED_MARGIN_RATE = 0.40  # i.e. 40% margin baked into CostPrice


def sample_context() -> Dict:
    price = round(random.uniform(199, 15000), 2)
    return {
        "Price": price,
        "CostPrice": round(price * (1 - ASSUMED_MARGIN_RATE), 2),
        "Category": random.choice(CATEGORIES),
        "Region": random.choice(REGIONS),
        "InventoryLevel": random.randint(0, 500),
        "CompetitorPricing": round(random.uniform(150, 15500), 2),
        "Seasonality": random.choice(SEASONALITY),
        "Promotion": random.choice([0, 1]),
        "DemandIndex": round(random.uniform(0, 1), 3),
    }


# ---------------------------------------------------------------------------
# 5. LIVE ROUND-GENERATION LOOP  <-- the actual Milestone 15 deliverable
#    Output shape: (context: dict, customer: Customer, eligible: List[int])
#    Keep this shape stable -- Milestone 19 must yield the identical shape.
# ---------------------------------------------------------------------------

def generate_rounds(
    registry: List[Customer],
    tier_boundaries: Dict = TIER_BOUNDARIES,
    tier_discounts: Dict = TIER_DISCOUNTS,
) -> Generator[Tuple[Dict, Customer, List[int]], None, None]:
    while True:
        customer = random.choice(registry)
        customer.LoyaltyTier = get_tier(customer.PurchaseCount, tier_boundaries)
        elig = eligible_discounts(customer.LoyaltyTier, tier_discounts)
        context = sample_context()
        yield context, customer, elig


# ---------------------------------------------------------------------------
# 6. 20-ROUND SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    registry = build_demo_registry()
    gen = generate_rounds(registry)

    print(f"{'Round':<6}{'CustomerID':<12}{'PurchaseCount':<15}{'Tier':<10}{'Eligible Discounts'}")
    print("-" * 70)
    for i in range(1, 21):
        context, customer, elig = next(gen)
        print(
            f"{i:<6}{customer.CustomerID:<12}{customer.PurchaseCount:<15}"
            f"{customer.LoyaltyTier:<10}{elig}"
        )

    # Sanity check matching the milestone's own acceptance criteria
    print("\nSanity check over 2,000 rounds:")
    gen2 = generate_rounds(registry)
    new_saw_20 = False
    loyal_saw_20 = False
    for _ in range(2000):
        _, customer, elig = next(gen2)
        if customer.LoyaltyTier == "New" and 20 in elig:
            new_saw_20 = True
        if customer.LoyaltyTier == "Loyal" and 20 in elig:
            loyal_saw_20 = True
    print(f"  New customer ever offered 20%?   {new_saw_20}  (must be False)")
    print(f"  Loyal customer ever offered 20%? {loyal_saw_20}  (must be True)")
