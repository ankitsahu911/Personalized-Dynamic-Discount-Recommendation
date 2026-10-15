"""
Upgrade: Warm-Start LinUCB from Historical Data Priors
Retail Discount-Recommendation Bandit Project

Standard LinUCB begins with b_a = 0 (theta_0 = 0), meaning it flies blind
during early rounds and accumulates high cold-start regret.

Warm-start initializes theta_0,a for each discount arm using a ridge-regression
prior fit on historical context features, giving the bandit prior knowledge of
how price, category, demand, and margin interact.

This script:
    1. Fits prior weight vectors theta_0,a for each arm [0, 5, 10, 15, 20].
    2. Runs an apples-to-apples comparison on early rounds (Rounds 1 - 2,000)
       for Cold-Start LinUCB vs Warm-Start LinUCB.
    3. Saves comparison plot: outputs/plots/warm_start_comparison.png.
    4. Computes early-round regret reduction percentage.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linucb import DisjointLinUCB, CONTEXT_DIM, build_context_vector
from simulator import simulate_outcome, get_oracle_and_regret
from training_rounds import load_training_rounds
from data_loader import get_table, DISCOUNT_LEVELS

PLOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots", "warm_start_comparison.png")
N_EVAL_ROUNDS = 2000


def estimate_prior_thetas(n_samples: int = 1000, l2_reg: float = 10.0) -> dict:
    """
    Fits a ridge regression prior for each discount arm using sample contexts.
    """
    from round_generator import Customer

    # Draw diverse sample contexts from live generator / demo registry
    table = get_table()
    categories = ["Electronics", "Groceries", "Furniture", "Toys", "Clothing"]
    regions = ["West", "South", "North", "East"]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    tiers = ["New", "Regular", "Loyal"]

    X_list = []
    y_dict = {a: [] for a in DISCOUNT_LEVELS}

    np.random.seed(42)
    for _ in range(n_samples):
        cat = np.random.choice(categories)
        reg = np.random.choice(regions)
        seas = np.random.choice(seasons)
        tier = np.random.choice(tiers)
        price = float(np.random.uniform(20, 120))
        cost = round(price * 0.60, 2)
        promo = int(np.random.choice([0, 1]))
        demand = float(np.random.uniform(0, 1))
        cps = float(np.random.uniform(0.7, 1.3))

        ctx = {
            "Price": price, "CostPrice": cost, "Category": cat, "Region": reg,
            "InventoryLevel": np.random.randint(20, 250),
            "CompetitorPricing": round(price * np.random.uniform(0.95, 1.05), 2),
            "Seasonality": seas, "Promotion": promo, "DemandIndex": demand,
            "CustomerPriceSensitivity": cps,
        }
        cust = Customer(1, np.random.randint(0, 20), np.random.randint(0, 60), tier, 100.0)

        x_vec = build_context_vector(ctx, cust)
        X_list.append(x_vec)

        # Expected profit for each arm under this context
        demand_level = "High" if demand < 0.34 else ("Medium" if demand < 0.67 else "Low")
        key = (cat, tier, demand_level, promo)
        bucket = table.get(key)
        if not bucket:
            bucket = [e for k, v in table.items() if k[0] == cat for e in v]
        entry = bucket[0]

        for a in DISCOUNT_LEVELS:
            prob = entry.get(a, (0.2, 2.0))[0]
            sp = price * (1.0 - a / 100.0)
            margin = sp - cost
            exp_p = prob * margin
            y_dict[a].append(exp_p)

    X = np.array(X_list)  # (n_samples, d)
    d = X.shape[1]

    prior_thetas = {}
    for a in DISCOUNT_LEVELS:
        y = np.array(y_dict[a])
        # Ridge regression: theta = (X^T X + lambda I)^(-1) X^T y
        theta_a = np.linalg.solve(X.T @ X + l2_reg * np.eye(d), X.T @ y)
        prior_thetas[a] = theta_a

    return prior_thetas


def run_warm_start_experiment():
    print("=" * 70)
    print("UPGRADE EXPERIMENT: Cold-Start vs. Warm-Start LinUCB")
    print(f"Evaluating across early {N_EVAL_ROUNDS:,} customer rounds...")
    print("=" * 70)

    # 1. Estimate prior weights
    print("\n[Step 1] Estimating prior weight vectors theta_0 from historical data...")
    prior_thetas = estimate_prior_thetas(n_samples=1500)
    for a in DISCOUNT_LEVELS:
        norm = np.linalg.norm(prior_thetas[a])
        print(f"  Arm {a:>2}% discount: prior ||theta_0|| = {norm:.4f}")

    # 2. Instantiate Cold and Warm bandits
    cold_bandit = DisjointLinUCB(arms=DISCOUNT_LEVELS, context_dim=CONTEXT_DIM, alpha=1.0)
    warm_bandit = DisjointLinUCB(
        arms=DISCOUNT_LEVELS,
        context_dim=CONTEXT_DIM,
        alpha=1.0,
        warm_start_thetas=prior_thetas,
    )

    rounds = load_training_rounds()[:N_EVAL_ROUNDS]

    cold_regrets = []
    warm_regrets = []
    cold_cum = 0.0
    warm_cum = 0.0

    print(f"\n[Step 2] Simulating {N_EVAL_ROUNDS:,} rounds on both models...")
    for idx, r in enumerate(rounds):
        ctx = r["context"]
        cust = r["customer"]
        elig = r["eligible_discounts"]
        seed = r["round_seed"]

        # Cold model decision & update
        ch_cold = cold_bandit.select_action(ctx, cust, elig)
        _, _, p_cold = simulate_outcome(ctx, cust, ch_cold, round_seed=seed)
        cold_bandit.update(ch_cold, ctx, cust, reward=p_cold)
        _, _, _, reg_cold = get_oracle_and_regret(ctx, cust, elig, ch_cold, seed=seed)
        cold_cum += reg_cold
        cold_regrets.append(cold_cum)

        # Warm model decision & update
        ch_warm = warm_bandit.select_action(ctx, cust, elig)
        _, _, p_warm = simulate_outcome(ctx, cust, ch_warm, round_seed=seed)
        warm_bandit.update(ch_warm, ctx, cust, reward=p_warm)
        _, _, _, reg_warm = get_oracle_and_regret(ctx, cust, elig, ch_warm, seed=seed)
        warm_cum += reg_warm
        warm_regrets.append(warm_cum)

    regret_saved = cold_cum - warm_cum
    pct_reduction = (regret_saved / cold_cum) * 100.0

    print("\n" + "=" * 70)
    print("WARM-START EXPERIMENTAL RESULTS (Rounds 1 - 2,000)")
    print("=" * 70)
    print(f"  Cold-Start Cumulative Regret : Rs {cold_cum:>10.2f}")
    print(f"  Warm-Start Cumulative Regret : Rs {warm_cum:>10.2f}")
    print(f"  Early-Round Regret Reduction : Rs {regret_saved:>10.2f} ({pct_reduction:+.2f}%)")

    # Plot comparison
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(9, 5))
    x_axis = np.arange(1, N_EVAL_ROUNDS + 1)
    plt.plot(x_axis, cold_regrets, label="Cold-Start LinUCB (Zero Prior)", color="#d62728", linestyle="--", linewidth=2)
    plt.plot(x_axis, warm_regrets, label=f"Warm-Start LinUCB (Regret -{pct_reduction:.1f}%)", color="#1f77b4", linewidth=2)
    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Cumulative Regret (Rs)", fontsize=11)
    plt.title("Impact of Warm-Start Initialization on Cold-Start Regret (First 2,000 Rounds)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"  Saved comparison plot to {PLOT_PATH}")

    return {
        "cold_regret": cold_cum,
        "warm_regret": warm_cum,
        "pct_reduction": pct_reduction,
        "prior_thetas": prior_thetas,
    }


if __name__ == "__main__":
    run_warm_start_experiment()
