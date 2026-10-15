"""
Milestone 30 - Full Thompson Sampling Training on an Identical Round Sequence
Retail Discount-Recommendation Bandit Project

Loads the exact 20,000-round sequence file from Milestone 29, trains
Bayesian Linear Thompson Sampling, verifies identical contexts and guardrail
adherence, saves plots and logs, and produces a side-by-side comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thompson_sampling import ThompsonSampling
from linucb import CONTEXT_DIM
from simulator import simulate_outcome, get_oracle_and_regret
from training_rounds import load_training_rounds
from model_io import save_thompson

N_ROUNDS = 20000
TS_SEED = 7
SIGMA = 1.0
ARMS = [0, 5, 10, 15, 20]
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "thompson_training_log.csv")
LINUCB_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "linucb_training_log.csv")
PLOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots", "thompson_cumulative_regret.png")
MODEL_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "thompson_v1")


def run_thompson_training():
    print("=" * 70)
    print("Milestone 30 -- Running Full Thompson Sampling Training Loop")
    print(f"Rounds: {N_ROUNDS:,} | Sigma: {SIGMA} | TS_SEED: {TS_SEED} | Context Dim: {CONTEXT_DIM}")
    print("=" * 70)

    # 1. Load exact round sequence from Milestone 29
    rounds = load_training_rounds()
    print(f"\n[Sequence Verification]")
    print(f"  Loaded {len(rounds):,} rounds from training_rounds_seed42_n20000.pkl.")
    first_round = rounds[0]
    print(f"  First round context keys: {list(first_round['context'].keys())}")
    print(f"  First round price: Rs {first_round['context']['Price']}, Category: {first_round['context']['Category']}")

    # Check against LinUCB log if available
    has_linucb_log = os.path.exists(LINUCB_LOG_PATH)
    if has_linucb_log:
        df_linucb = pd.read_csv(LINUCB_LOG_PATH)
        assert len(df_linucb) == len(rounds), "Round sequence length mismatch with LinUCB log!"
        print("  Verified: Round count matches LinUCB log exactly.")

    # 2. Instantiate Thompson Sampling with fixed RNG
    np.random.seed(TS_SEED)
    bandit = ThompsonSampling(arms=ARMS, context_dim=CONTEXT_DIM, sigma=SIGMA)

    records = []
    cum_regret = 0.0
    cum_profit = 0.0
    arm_pull_counts = {a: 0 for a in ARMS}

    print("\nStarting 20,000 training rounds...")
    for idx, r in enumerate(rounds):
        round_idx = r["round_idx"]
        context = r["context"]
        customer = r["customer"]
        eligible_discounts = r["eligible_discounts"]
        round_seed = r["round_seed"]

        # Verification: compare eligible_discounts with LinUCB log
        if has_linucb_log:
            linucb_elig = list(map(int, df_linucb.iloc[idx]["eligible_discounts"].split(",")))
            assert eligible_discounts == linucb_elig, f"Eligible discounts mismatch at round {round_idx}!"

        # Action selection
        chosen_discount = bandit.select_action(context, customer, eligible_discounts)
        arm_pull_counts[chosen_discount] += 1

        # Verification 4(c): Chosen discount must be within eligible discounts
        assert chosen_discount in eligible_discounts, (
            f"Violation at round {round_idx}: chosen {chosen_discount} not in {eligible_discounts}!"
        )

        # Simulate outcome using exact per-round seed
        purchased, revenue, profit = simulate_outcome(
            context, customer, chosen_discount, round_seed=round_seed
        )

        # Update posterior
        bandit.update(chosen_discount, context, customer, reward=profit)

        # Compute oracle & regret
        oracle_disc, oracle_exp, chosen_exp, regret = get_oracle_and_regret(
            context, customer, eligible_discounts, chosen_discount, seed=round_seed
        )

        # Verification 4(b): Regret is never meaningfully negative
        assert regret >= -1e-5, f"Bug: negative regret {regret} at round {round_idx}!"

        cum_regret += regret
        cum_profit += profit
        price = float(context["Price"])
        disc_amount = round(price * (chosen_discount / 100.0), 2)

        records.append({
            "round_idx": round_idx,
            "customer_tier": getattr(customer, "LoyaltyTier", "") or "New",
            "eligible_discounts": ",".join(map(str, eligible_discounts)),
            "chosen_discount": chosen_discount,
            "purchased": int(purchased),
            "revenue": revenue,
            "profit": profit,
            "discount_amount_rupees": disc_amount,
            "oracle_discount": oracle_disc,
            "oracle_expected_profit": round(oracle_exp, 4),
            "chosen_expected_profit": round(chosen_exp, 4),
            "regret": round(regret, 4),
            "cumulative_regret": round(cum_regret, 4),
            "cumulative_profit": round(cum_profit, 2),
        })

        if round_idx % 2000 == 0:
            print(f"  Round {round_idx:>5,d} / {N_ROUNDS:,} | Cum Regret: Rs {cum_regret:>9.2f} | Cum Profit: Rs {cum_profit:>10.2f} | Arm counts: {arm_pull_counts}")

    # Save log
    df_log = pd.DataFrame(records)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df_log.to_csv(LOG_PATH, index=False)
    print(f"\n[Logging] Saved complete training log to {LOG_PATH}")

    # -----------------------------------------------------------------------
    # Convergence Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONVERGENCE REPORT -- Thompson Sampling")
    print("=" * 70)

    print("\nAverage Per-Round Regret in Consecutive 1,000-Round Windows:")
    print(f"{'Window':<15} {'Rounds':<18} {'Avg Regret / Round':>20} {'Avg Profit / Round':>20}")
    print("-" * 75)
    for w in range(20):
        start_r = w * 1000
        end_r = (w + 1) * 1000
        sub = df_log.iloc[start_r:end_r]
        avg_r = sub["regret"].mean()
        avg_p = sub["profit"].mean()
        print(f"Window {w+1:>2}        {start_r+1:>5} - {end_r:<5}        Rs {avg_r:>14.4f}        Rs {avg_p:>14.4f}")

    total_conv = df_log["purchased"].sum()
    total_prof = df_log["profit"].sum()
    avg_disc = df_log["chosen_discount"].mean()

    print(f"\nOverall Summary (Thompson Sampling):")
    print(f"  Total conversions     : {total_conv:,} / {N_ROUNDS:,} ({total_conv/N_ROUNDS*100:.2f}%)")
    print(f"  Total profit earned   : Rs {total_prof:,.2f}")
    print(f"  Final cumulative regret: Rs {cum_regret:,.2f}")
    print(f"  Average discount chosen: {avg_disc:.2f}%")

    print(f"\nChosen Discount Distribution (Overall):")
    dist = df_log["chosen_discount"].value_counts().sort_index()
    for d, c in dist.items():
        print(f"  {d:>2}% discount : {c:>5} rounds ({c/N_ROUNDS*100:>5.1f}%)")

    # Plot cumulative regret
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(df_log["round_idx"], df_log["cumulative_regret"], color="#2ca02c", linewidth=2, label="Thompson Sampling (sigma=1.0)")
    if has_linucb_log:
        plt.plot(df_linucb["round_idx"], df_linucb["cumulative_regret"], color="#1f77b4", linewidth=1.5, linestyle="--", label="LinUCB (alpha=1.0)")
    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Cumulative Regret (Rs)", fontsize=11)
    plt.title("Cumulative Regret Over 20,000 Rounds", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"  Saved cumulative regret plot to {PLOT_PATH}")

    # -----------------------------------------------------------------------
    # Step 6: Side-by-Side Preview Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: PREVIEW COMPARISON -- Single Seed Run (seed=42)")
    print("Note: This is an initial single-seed preview. The comprehensive multi-seed")
    print("statistical benchmark and significance testing will be conducted in Phase 7.")
    print("=" * 70)

    if has_linucb_log:
        linucb_conv = df_linucb["purchased"].sum()
        linucb_prof = df_linucb["profit"].sum()
        linucb_reg = df_linucb["cumulative_regret"].iloc[-1]
        linucb_disc = df_linucb["chosen_discount"].mean()

        print(f"{'Metric':<30} {'LinUCB (alpha=1.0)':>20} {'Thompson Sampling (sigma=1.0)':>30}")
        print("-" * 82)
        print(f"{'Final Cumulative Regret':<30} Rs {linucb_reg:>17.2f} Rs {cum_regret:>27.2f}")
        print(f"{'Total Profit Earned':<30} Rs {linucb_prof:>17.2f} Rs {total_prof:>27.2f}")
        print(f"{'Conversion Count':<30} {linucb_conv:>16,d} / 20k {total_conv:>26,d} / 20k")
        print(f"{'Conversion Rate':<30} {linucb_conv/N_ROUNDS*100:>19.2f}% {total_conv/N_ROUNDS*100:>29.2f}%")
        print(f"{'Average Discount Chosen':<30} {linucb_disc:>19.2f}% {avg_disc:>29.2f}%")

    # -----------------------------------------------------------------------
    # Save Model Parameters (Milestone 31)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING TRAINED MODEL (Milestone 31 integration)")
    print("=" * 70)
    save_thompson(
        model=bandit,
        path_prefix=MODEL_PREFIX,
        n_rounds=N_ROUNDS,
        ts_seed=TS_SEED,
        sequence_file="training_rounds_seed42_n20000.pkl",
    )

    print("\nMilestone 30 Training Complete!")
    return df_log, bandit


if __name__ == "__main__":
    run_thompson_training()
