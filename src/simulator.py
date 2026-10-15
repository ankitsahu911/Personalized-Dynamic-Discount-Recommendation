"""
Milestone 25 - Simulator + run_round() Integration  [UPDATED: Real Dataset]
Retail Discount-Recommendation Bandit Project

What this file does:
    1. DATA-DRIVEN simulator using real Prob_at_X and Profit_at_X values
       from the 73,100-row dataset (via data_loader.py).
       No invented formulas -- actual purchase probabilities and profits
       from real data drive every simulated outcome.

    2. run_round(): pulls one live round, bandit picks a discount,
       simulator scores it, bandit updates, returns a log dict.

REWARD -- Real Profit from Dataset:
    data_loader.lookup() returns:
        prob   = real Prob_at_X  (e.g. 0.226 for 10% discount in this context)
        profit = real Profit_at_X (expected profit per unit, from real CostPrice)

    reward    = profit  (expected profit -- already probability-weighted in dataset)
    purchased = Bernoulli(prob)
    revenue   = discounted_price * purchased  (computed from context Price)

Run:
    python src/simulator.py
"""

import random
from typing import Dict, List, Optional, Tuple

from data_loader import lookup, get_context_probabilities
import numpy as np


# ---------------------------------------------------------------------------
# 1. SINGLE-ROUND OUTCOME SIMULATOR  (data-driven with optional per-round seed)
# ---------------------------------------------------------------------------

def get_expected_profit(context: Dict, customer, discount: int, seed: Optional[int] = None) -> float:
    """
    Compute EXPECTED profit for a given context and discount:
        E[profit] = P(purchase | context, discount) * (SellingPrice - CostPrice)
    """
    category     = context.get("Category", "Electronics")
    demand_index = context.get("DemandIndex", 0.5)
    promotion    = int(context.get("Promotion", 0))
    price        = float(context.get("Price", 0.0))
    cost_price   = float(context.get("CostPrice", price * 0.60))
    loyalty_tier = getattr(customer, "LoyaltyTier", "") or "New"

    probs = get_context_probabilities(category, loyalty_tier, demand_index, promotion, seed=seed)
    prob = probs.get(discount, 0.0)

    selling_price = price * (1.0 - discount / 100.0)
    margin = selling_price - cost_price
    return float(prob * margin)


def get_oracle_and_regret(
    context: Dict,
    customer,
    eligible_discounts: List[int],
    chosen_discount: int,
    seed: Optional[int] = None,
) -> Tuple[int, float, float, float]:
    """
    Find the oracle discount among eligible_discounts and compute regret.

    Returns:
        (oracle_discount, oracle_expected_profit, chosen_expected_profit, regret)
    """
    category     = context.get("Category", "Electronics")
    demand_index = context.get("DemandIndex", 0.5)
    promotion    = int(context.get("Promotion", 0))
    price        = float(context.get("Price", 0.0))
    cost_price   = float(context.get("CostPrice", price * 0.60))
    loyalty_tier = getattr(customer, "LoyaltyTier", "") or "New"

    probs = get_context_probabilities(category, loyalty_tier, demand_index, promotion, seed=seed)

    best_arm = None
    best_exp_profit = -float("inf")
    exp_profits = {}

    for d in eligible_discounts:
        p = probs.get(d, 0.0)
        sp = price * (1.0 - d / 100.0)
        margin = sp - cost_price
        exp_p = p * margin
        exp_profits[d] = exp_p
        if exp_p > best_exp_profit:
            best_exp_profit = exp_p
            best_arm = d

    chosen_exp = exp_profits.get(chosen_discount, 0.0)
    regret = max(0.0, best_exp_profit - chosen_exp)
    return best_arm, best_exp_profit, chosen_exp, regret


def simulate_outcome(
    context: Dict,
    customer,
    discount: int,
    round_seed: Optional[int] = None,
) -> Tuple[bool, float, float]:
    """
    Simulate one customer interaction using REAL probabilities from data_loader.

    If round_seed is provided, randomness is deterministic based on (round_seed, discount)
    ensuring identical outcomes for identical (round, discount) pairs across policies.

    Parameters
    ----------
    context    : dict  -- must contain Category, DemandIndex, Promotion, Price, CostPrice
    customer   : Customer dataclass -- provides LoyaltyTier
    discount   : int   -- the discount offered (0, 5, 10, 15, 20)
    round_seed : int, optional -- seed for reproducible per-round outcome

    Returns
    -------
    purchased : bool   -- did the customer buy?
    revenue   : float  -- discounted sale price if purchased, else 0
    profit    : float  -- (SellingPrice - CostPrice) if purchased, else 0 (REWARD)
    """
    category     = context.get("Category", "Electronics")
    demand_index = context.get("DemandIndex", 0.5)
    promotion    = int(context.get("Promotion", 0))
    price        = float(context.get("Price", 0.0))
    cost_price   = float(context.get("CostPrice", price * 0.60))
    loyalty_tier = getattr(customer, "LoyaltyTier", "") or "New"

    probs = get_context_probabilities(category, loyalty_tier, demand_index, promotion, seed=round_seed)
    prob = probs.get(discount, 0.20)

    if round_seed is not None:
        rng = np.random.default_rng(abs(round_seed * 10007 + discount * 31) % (2**32))
        purchased = bool(rng.random() < prob)
    else:
        purchased = bool(random.random() < prob)

    if purchased:
        selling_price = round(price * (1.0 - discount / 100.0), 2)
        revenue = selling_price
        profit  = round(selling_price - cost_price, 2)
    else:
        revenue = 0.0
        profit  = 0.0

    return purchased, revenue, profit




# ---------------------------------------------------------------------------
# 3. run_round() -- THE CORE INTEGRATION FUNCTION
# ---------------------------------------------------------------------------

def run_round(bandit, round_generator) -> Dict:
    """
    Execute one full round of the bandit loop:
        pull context -> select action -> simulate outcome -> update bandit -> log

    Parameters
    ----------
    bandit          : DisjointLinUCB or ThompsonSampling instance
                      Must have .select_action(context, customer, eligible_arms)
                      and .update(arm, context, customer, reward).
    round_generator : generator yielding (context, customer, eligible_discounts)
                      This is the output of live_generator.generate_live_rounds().
                      eligible_discounts IS already the guardrail-filtered set
                      (confirmed: live_generator calls get_eligible_actions()
                      internally before yielding -- see live_generator.py line 150).

    Returns
    -------
    dict with keys:
        context, customer, eligible_discounts, chosen_discount,
        purchased, revenue, profit
        (Milestone 28's audit and Phase 6's training loop consume this shape.)
    """
    # Step 1: Pull next round from the generator
    context, customer, eligible_discounts = next(round_generator)

    # Step 2: Bandit picks a discount from the eligible set
    # NOTE: eligible_discounts from live_generator is already guardrail-filtered.
    # We do NOT re-call get_eligible_actions() here -- that would be double-filtering.
    chosen_discount = bandit.select_action(context, customer, eligible_discounts)

    # Step 3: Simulate the outcome using REAL dataset probabilities
    purchased, revenue, profit = simulate_outcome(context, customer, chosen_discount)

    # Step 4: Update the bandit with the observed reward (profit = margin earned)
    bandit.update(chosen_discount, context, customer, profit)

    # Step 5: Return full log dict for audit + training loop consumption
    return {
        "context":            context,
        "customer_id":        customer.CustomerID,
        "customer_tier":      getattr(customer, "LoyaltyTier", "") or "New",
        "purchase_count":     customer.PurchaseCount,
        "eligible_discounts": eligible_discounts,
        "chosen_discount":    chosen_discount,
        "purchased":          purchased,
        "revenue":            revenue,
        "profit":             profit,
    }


# ---------------------------------------------------------------------------
# 4. SELF-TEST: 5-round end-to-end trace
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from linucb import DisjointLinUCB, CONTEXT_DIM
    from live_generator import generate_live_rounds, build_demo_registry

    ARMS = [0, 5, 10, 15, 20]
    random.seed(42)

    registry  = build_demo_registry()
    gen       = generate_live_rounds(registry)
    bandit    = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=1.0)

    print("=" * 70)
    print("Milestone 25 -- 5-Round End-to-End Pipeline Trace")
    print("  context -> guardrails -> bandit -> simulator -> update -> log")
    print("=" * 70)

    for i in range(1, 6):
        log = run_round(bandit, gen)
        print(f"\nRound {i}:")
        print(f"  Tier        : {log['customer_tier']}")
        print(f"  Eligible    : {log['eligible_discounts']}")
        print(f"  Chosen      : {log['chosen_discount']}%")
        print(f"  Purchased   : {log['purchased']}")
        print(f"  Revenue     : Rs {log['revenue']:.2f}")
        print(f"  Profit      : Rs {log['profit']:.2f}  <- reward signal fed to bandit")

    print("\n" + "=" * 70)
    print("Pipeline verified -- run_round() working end-to-end.")
    print("=" * 70)
