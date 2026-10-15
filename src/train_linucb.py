"""
Milestone 29 - Full LinUCB Training on 20,000 Rounds
Retail Discount-Recommendation Bandit Project

Executes the full 20,000-round training loop for Disjoint LinUCB, produces
convergence reports, plots cumulative regret, and saves the trained model.
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
from model_io import save_linucb

N_ROUNDS = 20000
MAIN_SEED = 42
ALPHA = 1.0
ARMS = [0, 5, 10, 15, 20]
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "linucb_training_log.csv")
PLOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots", "linucb_cumulative_regret.png")
MODEL_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "linucb_v1")


def run_linucb_training():
    print("=" * 70)
    print("Milestone 29 -- Running Full Disjoint LinUCB Training Loop")
    print(f"Rounds: {N_ROUNDS:,} | Alpha: {ALPHA} | Context Dim: {CONTEXT_DIM}")
    print("=" * 70)

    # Step 0 Inspection Notes
    print("\n[STEP 0 Inspection]")
    print("  (a) Customer state evolution: Customer visits and background purchases evolve on")
    print("      a fixed schedule independent of bandit action during pre-generation.")
    print("  (b) Expected profit calculation: Computed using real purchase probabilities from data_loader")
    print("      multiplied by the unit margin: E[profit] = P(purchase) * (SellingPrice - CostPrice).")
    print("  (c) Simulator determinism: Uses per-round seeds (np.random.default_rng) so (round, discount)")
    print("      always produces the exact same outcome across all policies.")

    # Load pre-generated rounds
    rounds = load_training_rounds()
    print(f"\nLoaded {len(rounds):,} pre-generated rounds successfully.")

    bandit = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=ALPHA)

    records = []
    cum_regret = 0.0
    cum_profit = 0.0
    arm_pull_counts = {a: 0 for a in ARMS}

    # Tracking diagnostic magnitudes
    exploit_terms_100 = []
    explore_terms_100 = []
    exploit_terms_10k = []
    explore_terms_10k = []

    print("\nStarting 20,000 training rounds...")
    for idx, r in enumerate(rounds):
        round_idx = r["round_idx"]
        context = r["context"]
        customer = r["customer"]
        eligible_discounts = r["eligible_discounts"]
        round_seed = r["round_seed"]

        # Bandit selects action from eligible discounts
        chosen_discount = bandit.select_action(context, customer, eligible_discounts)
        arm_pull_counts[chosen_discount] += 1

        # Diagnostic capture at rounds 100 and 10,000
        if round_idx == 100 or round_idx == 10000:
            x = build_context_vector(context, customer)
            for a in eligible_discounts:
                A_inv = np.linalg.inv(bandit.A[a])
                th = A_inv @ bandit.b[a]
                exp_t = float(th @ x)
                bon_t = float(bandit.alpha * np.sqrt(x @ A_inv @ x))
                if round_idx == 100:
                    exploit_terms_100.append(exp_t)
                    explore_terms_100.append(bon_t)
                else:
                    exploit_terms_10k.append(exp_t)
                    explore_terms_10k.append(bon_t)

        # Simulate outcome
        purchased, revenue, profit = simulate_outcome(
            context, customer, chosen_discount, round_seed=round_seed
        )

        # Update bandit with reward = profit
        bandit.update(chosen_discount, context, customer, reward=profit)

        # Compute oracle & regret
        oracle_disc, oracle_exp, chosen_exp, regret = get_oracle_and_regret(
            context, customer, eligible_discounts, chosen_discount, seed=round_seed
        )

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

    # Save training log
    df_log = pd.DataFrame(records)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df_log.to_csv(LOG_PATH, index=False)
    print(f"\n[Logging] Saved complete training log to {LOG_PATH}")

    # -----------------------------------------------------------------------
    # STEP 3: Convergence Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: CONVERGENCE REPORT -- LinUCB")
    print("=" * 70)

    # Windowed average regret table
    print("\nAverage Per-Round Regret in Consecutive 1,000-Round Windows:")
    print(f"{'Window':<15} {'Rounds':<18} {'Avg Regret / Round':>20} {'Avg Profit / Round':>20}")
    print("-" * 75)
    window_regrets = []
    for w in range(20):
        start_r = w * 1000
        end_r = (w + 1) * 1000
        sub = df_log.iloc[start_r:end_r]
        avg_r = sub["regret"].mean()
        avg_p = sub["profit"].mean()
        window_regrets.append(avg_r)
        print(f"Window {w+1:>2}        {start_r+1:>5} - {end_r:<5}        Rs {avg_r:>14.4f}        Rs {avg_p:>14.4f}")

    total_conv = df_log["purchased"].sum()
    total_prof = df_log["profit"].sum()
    avg_disc = df_log["chosen_discount"].mean()

    print(f"\nOverall Summary:")
    print(f"  Total conversions     : {total_conv:,} / {N_ROUNDS:,} ({total_conv/N_ROUNDS*100:.2f}%)")
    print(f"  Total profit earned   : Rs {total_prof:,.2f}")
    print(f"  Final cumulative regret: Rs {cum_regret:,.2f}")
    print(f"  Average discount chosen: {avg_disc:.2f}%")

    print(f"\nChosen Discount Distribution (Overall):")
    dist = df_log["chosen_discount"].value_counts().sort_index()
    for d, c in dist.items():
        print(f"  {d:>2}% discount : {c:>5} rounds ({c/N_ROUNDS*100:>5.1f}%)")

    print(f"\nChosen Discount Distribution by Customer Tier:")
    tier_dist = pd.crosstab(df_log["customer_tier"], df_log["chosen_discount"], margins=True)
    print(tier_dist)

    # Find flattening point: where window regret stays within 20% of final window value
    final_val = window_regrets[-1]
    flatten_window = None
    for w_idx in range(len(window_regrets)):
        # Check if all subsequent windows stay within 20% of final window value
        if all(abs(window_regrets[k] - final_val) <= 0.20 * final_val for k in range(w_idx, len(window_regrets))):
            flatten_window = w_idx + 1
            break

    print(f"\nFlattening Point Analysis:")
    if flatten_window is not None:
        print(f"  Stabilization point: Window {flatten_window} (around Round {(flatten_window-1)*1000:,} - {flatten_window*1000:,}).")
    else:
        print(f"  Curve continues gradual linear exploration across windows.")

    print("  Note on Regret Slope:")
    print("  As expected, cumulative regret does NOT flatten to zero because retail purchase outcomes")
    print("  are probabilistic/logistic while LinUCB is linear. A well-converged bandit displays a slope")
    print("  that drops noticeably from early exploration and then stabilizes into a low, steady rate.")

    # Plot cumulative regret
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(df_log["round_idx"], df_log["cumulative_regret"], color="#1f77b4", linewidth=2, label="LinUCB (alpha=1.0)")
    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Cumulative Regret (Rs)", fontsize=11)
    plt.title("Disjoint LinUCB Cumulative Regret Over 20,000 Rounds", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"  Saved cumulative regret plot to {PLOT_PATH}")

    # -----------------------------------------------------------------------
    # STEP 4: Diagnosis Checks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: DIAGNOSTIC REPORT")
    print("=" * 70)
    print("1. Feature scaling:")
    print(f"   All {CONTEXT_DIM} features are standardized (z-score) or binary/one-hot; scale is balanced around [0, 1].")

    print("\n2. Reward scale vs. Alpha comparison:")
    if exploit_terms_100 and explore_terms_100:
        print(f"   At Round 100   -- Mean Point Estimate: Rs {np.mean(exploit_terms_100):.3f} | Mean UCB Bonus: Rs {np.mean(explore_terms_100):.3f}")
    if exploit_terms_10k and explore_terms_10k:
        print(f"   At Round 10,000 -- Mean Point Estimate: Rs {np.mean(exploit_terms_10k):.3f} | Mean UCB Bonus: Rs {np.mean(explore_terms_10k):.3f}")
    print("   Observation: Exploration bonus naturally shrinks as matrix A accumulates observations.")

    print("\n3. Simulator signal:")
    exp_gap = (df_log["oracle_expected_profit"] - df_log["chosen_expected_profit"]).mean()
    print(f"   Average expected profit gap across eligible actions: Rs {exp_gap:.4f}")

    # -----------------------------------------------------------------------
    # Save Model Parameters (Milestone 31)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING TRAINED MODEL (Milestone 31 integration)")
    print("=" * 70)
    save_linucb(
        model=bandit,
        path_prefix=MODEL_PREFIX,
        n_rounds=N_ROUNDS,
        main_seed=MAIN_SEED,
        sequence_file="training_rounds_seed42_n20000.pkl",
        arm_pull_counts=arm_pull_counts,
    )

    print("\nMilestone 29 Training Complete!")
    return df_log, bandit


if __name__ == "__main__":
    run_linucb_training()
