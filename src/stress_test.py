"""
Upgrade: Non-Stationary Market Shock Stress Test
Retail Discount-Recommendation Bandit Project

Simulates a sudden market shock at Round 5,000:
    Competitor crashes prices by 30% (CompetitorPricing *= 0.70)
    Customers become significantly more price-sensitive.

Compares 4 policies:
    1. Fixed 10% Baseline (Static)
    2. Rule-Based Policy (Static logic)
    3. Standard LinUCB (Stationary, gamma = 1.0)
    4. Discounted LinUCB (Non-stationary, gamma = 0.999)

Outputs:
    outputs/plots/stress_test_competitor_shock.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linucb import DisjointLinUCB, CONTEXT_DIM
from fixed_baseline import fixed_baseline
from rule_based_baseline import rule_based_baseline
from simulator import simulate_outcome, get_oracle_and_regret
from training_rounds import load_training_rounds

PLOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots", "stress_test_competitor_shock.png")
N_ROUNDS = 10000
SHOCK_ROUND = 5000
ARMS = [0, 5, 10, 15, 20]


def run_stress_test():
    print("=" * 70)
    print("STRESS TEST EXPERIMENT: Non-Stationary Competitor Price Shock")
    print(f"Total Rounds: {N_ROUNDS:,} | Shock Point: Round {SHOCK_ROUND:,}")
    print("=" * 70)

    rounds = load_training_rounds()[:N_ROUNDS]

    # Instantiate the 4 policies
    linucb_standard = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=1.0, gamma=1.0)
    linucb_discounted = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=1.0, gamma=0.999)

    profits = {
        "Fixed 10%": [],
        "Rule-Based": [],
        "Standard LinUCB": [],
        "Discounted LinUCB (gamma=0.999)": [],
    }

    regrets = {
        "Standard LinUCB": [],
        "Discounted LinUCB (gamma=0.999)": [],
    }

    cum_regret_std = 0.0
    cum_regret_d = 0.0

    print("\nSimulating rounds across shock boundary...")
    for idx, r in enumerate(rounds):
        round_idx = r["round_idx"]
        context = dict(r["context"])
        customer = r["customer"]
        eligible = r["eligible_discounts"]
        seed = r["round_seed"]

        # Inject shock at round SHOCK_ROUND
        is_shocked = round_idx >= SHOCK_ROUND
        if is_shocked:
            # Competitor drops price by 30%
            context["CompetitorPricing"] = round(float(context["CompetitorPricing"]) * 0.70, 2)
            # Customer price sensitivity surges
            context["CustomerPriceSensitivity"] = min(1.55, float(context.get("CustomerPriceSensitivity", 1.0)) * 1.35)

        # 1. Fixed 10% Baseline
        ch_fixed = fixed_baseline(eligible, target=10)
        _, _, p_fixed = simulate_outcome(context, customer, ch_fixed, round_seed=seed)
        profits["Fixed 10%"].append(p_fixed)

        # 2. Rule-Based Baseline
        raw_demand = context.get("DemandIndex", 0.5)
        d_idx = 2 if raw_demand >= 0.67 else (1 if raw_demand >= 0.34 else 0)
        ch_rule = rule_based_baseline({"demand_index": d_idx}, eligible)
        _, _, p_rule = simulate_outcome(context, customer, ch_rule, round_seed=seed)
        profits["Rule-Based"].append(p_rule)

        # 3. Standard LinUCB
        ch_std = linucb_standard.select_action(context, customer, eligible)
        _, _, p_std = simulate_outcome(context, customer, ch_std, round_seed=seed)
        linucb_standard.update(ch_std, context, customer, reward=p_std)
        profits["Standard LinUCB"].append(p_std)
        _, _, _, r_std = get_oracle_and_regret(context, customer, eligible, ch_std, seed=seed)
        cum_regret_std += r_std
        regrets["Standard LinUCB"].append(cum_regret_std)

        # 4. Discounted LinUCB
        ch_d = linucb_discounted.select_action(context, customer, eligible)
        _, _, p_d = simulate_outcome(context, customer, ch_d, round_seed=seed)
        linucb_discounted.update(ch_d, context, customer, reward=p_d)
        profits["Discounted LinUCB (gamma=0.999)"].append(p_d)
        _, _, _, r_d = get_oracle_and_regret(context, customer, eligible, ch_d, seed=seed)
        cum_regret_d += r_d
        regrets["Discounted LinUCB (gamma=0.999)"].append(cum_regret_d)

        if round_idx % 2000 == 0:
            print(f"  Processed {round_idx:,} / {N_ROUNDS:,} rounds...")

    # Calculate post-shock performance (Rounds 5,000 - 10,000)
    print("\n" + "=" * 70)
    print(f"POST-SHOCK RESULTS (Rounds {SHOCK_ROUND:,} - {N_ROUNDS:,})")
    print("=" * 70)
    for name, p_list in profits.items():
        pre_mean = np.mean(p_list[:SHOCK_ROUND])
        post_mean = np.mean(p_list[SHOCK_ROUND:])
        drop_pct = (post_mean - pre_mean) / pre_mean * 100.0
        print(f"  {name:<32} : Pre-Shock Rs {pre_mean:.2f} | Post-Shock Rs {post_mean:.2f} ({drop_pct:+.1f}%)")

    print(f"\nFinal Cumulative Regret across Market Shock:")
    print(f"  Standard LinUCB (gamma=1.0)   : Rs {cum_regret_std:.2f}")
    print(f"  Discounted LinUCB (gamma=0.999): Rs {cum_regret_d:.2f} (-{(cum_regret_std - cum_regret_d)/cum_regret_std*100:.1f}%)")

    # Plot rolling profit & cumulative regret
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    window = 300
    rounds_x = np.arange(1, N_ROUNDS + 1)

    # Subplot 1: Rolling average profit
    for name, p_list in profits.items():
        rolling_p = pd.Series(p_list).rolling(window=window, min_periods=50).mean()
        style = "--" if "Fixed" in name or "Rule" in name else "-"
        width = 2.0 if "Discounted" in name else (1.5 if "Standard" in name else 1.2)
        ax1.plot(rounds_x, rolling_p, label=name, linestyle=style, linewidth=width)

    ax1.axvline(x=SHOCK_ROUND, color="black", linestyle=":", linewidth=2, label="Shock: -30% Competitor Price")
    ax1.set_ylabel(f"Rolling Avg Profit (window={window}) [Rs]", fontsize=10)
    ax1.set_title("Market Shock Adaptation: Rolling Profit Before & After Competitor Price Crash", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Subplot 2: Cumulative regret comparison
    ax2.plot(rounds_x, regrets["Standard LinUCB"], label="Standard LinUCB (Stationary)", color="#1f77b4", linewidth=2)
    ax2.plot(rounds_x, regrets["Discounted LinUCB (gamma=0.999)"], label="Discounted LinUCB (Adaptive)", color="#2ca02c", linewidth=2)
    ax2.axvline(x=SHOCK_ROUND, color="black", linestyle=":", linewidth=2)
    ax2.set_xlabel("Customer Round", fontsize=10)
    ax2.set_ylabel("Cumulative Regret [Rs]", fontsize=10)
    ax2.set_title("Cumulative Regret Across Market Shock", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"\nSaved stress-test plot to {PLOT_PATH}")


if __name__ == "__main__":
    run_stress_test()
