"""
Helper script to generate the complete, accurate multiseed_online_by_tier.csv
with true customer tiers (New, Regular, Loyal) derived from customer purchase history.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import make_sequence, PolicyWrapper, run_policy, EVAL_SEEDS, N_ROUNDS, POLICY_ORDER
from round_generator import get_tier
from data_loader import DISCOUNT_LEVELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TIER_CSV = os.path.join(RESULTS_DIR, "multiseed_online_by_tier.csv")


def generate_accurate_tier_breakdown():
    print("=" * 80)
    print("GENERATING ACCURATE MULTI-SEED TIER BREAKDOWN (New, Regular, Loyal)")
    print("=" * 80)

    t0 = time.time()
    tier_records = []

    for s_idx, seed in enumerate(EVAL_SEEDS):
        print(f"Processing Seed {seed} ({s_idx+1}/{len(EVAL_SEEDS)})...")
        seq = make_sequence(seed=seed, n_rounds=N_ROUNDS)

        policies = [
            PolicyWrapper("Random", "random", seed=seed + 101),
            PolicyWrapper("Fixed 10%", "fixed"),
            PolicyWrapper("Rule-based", "rule_based"),
            PolicyWrapper("LinUCB", "linucb", alpha=1.0),
            PolicyWrapper("Thompson Sampling", "thompson", seed=seed + 707, sigma=1.0),
        ]

        for pol in policies:
            df = run_policy(pol, seq, learning=True)

            # Map true tier from eligible discounts or customer purchase count
            def get_true_tier(elig_str):
                if elig_str == "0,5":
                    return "New"
                elif elig_str == "5,10,15":
                    return "Regular"
                elif elig_str == "10,15,20":
                    return "Loyal"
                return "Unknown"

            df["customer_tier"] = df["eligible_discounts"].apply(get_true_tier)

            for tier, df_t in df.groupby("customer_tier"):
                t_rounds = len(df_t)
                t_conv = int(df_t["purchased"].sum())

                rec = {
                    "seed": seed,
                    "policy": pol.name,
                    "customer_tier": tier,
                    "tier_rounds": t_rounds,
                    "tier_total_profit": float(df_t["profit"].sum()),
                    "tier_avg_profit_per_round": float(df_t["profit"].mean()),
                    "tier_conversion_rate": float(t_conv / t_rounds * 100.0),
                    "tier_avg_discount_offered": float(df_t["chosen_discount"].mean()),
                    "tier_avg_regret_per_round": float(df_t["regret"].mean()),
                    "tier_pct_chose_oracle": float((df_t["chosen_discount"] == df_t["oracle_discount"]).mean() * 100.0),
                }

                for d in DISCOUNT_LEVELS:
                    rec[f"tier_share_discount_{d}pct"] = float((df_t["chosen_discount"] == d).mean() * 100.0)

                tier_records.append(rec)

    df_tiers = pd.DataFrame(tier_records)
    df_tiers.to_csv(TIER_CSV, index=False)
    elapsed = time.time() - t0
    print(f"\nSaved accurate tier breakdown to {TIER_CSV} in {elapsed:.2f} seconds.")

    # Summary table
    print("\nSummary of Tier Metrics Across 10 Seeds (Mean values):")
    summary = df_tiers.groupby(["customer_tier", "policy"]).agg({
        "tier_rounds": "mean",
        "tier_avg_profit_per_round": "mean",
        "tier_conversion_rate": "mean",
        "tier_avg_discount_offered": "mean",
        "tier_avg_regret_per_round": "mean",
        "tier_pct_chose_oracle": "mean",
        "tier_share_discount_0pct": "mean",
        "tier_share_discount_5pct": "mean",
        "tier_share_discount_10pct": "mean",
        "tier_share_discount_15pct": "mean",
        "tier_share_discount_20pct": "mean",
    }).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    generate_accurate_tier_breakdown()
