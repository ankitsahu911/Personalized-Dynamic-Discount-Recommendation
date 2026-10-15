"""
Upgrade: Doubly Robust (DR) Off-Policy Evaluation
Retail Discount-Recommendation Bandit Project

Implements Doubly Robust (DR) estimation (Dudik, Langford, & Li, 2011) to benchmark
policies offline against historical interaction logs without bias or excessive variance.

The DR estimator:
    V_DR(pi) = (1/N) * sum_{i=1}^N [ r_hat(x_i, pi(x_i))
               + (I(pi(x_i) == a_i) / p(a_i | x_i)) * (r_i - r_hat(x_i, a_i)) ]

Properties:
    - Combines direct model (DM) and inverse propensity scoring (IPS / Replay).
    - Unbiased if either the reward model OR the propensities are accurate.
    - Yields tighter confidence intervals than plain replay sampling.

Evaluates 5 Policies:
    1. LinUCB (alpha=1.0)
    2. Thompson Sampling (sigma=1.0)
    3. Rule-Based Baseline
    4. Fixed 10% Baseline
    5. Uniform Random Baseline
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linucb import DisjointLinUCB, CONTEXT_DIM
from thompson_sampling import ThompsonSampling
from fixed_baseline import fixed_baseline
from rule_based_baseline import rule_based_baseline
from random_baseline import random_baseline
from simulator import get_expected_profit
from training_rounds import load_training_rounds
from model_io import load_linucb, load_thompson

LINUCB_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "linucb_v1")
THOMPSON_PREFIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "thompson_v1")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "doubly_robust_evaluation.csv")


def evaluate_doubly_robust(n_rounds: int = 5000):
    print("=" * 70)
    print("DOUBLY ROBUST (DR) OFF-POLICY EVALUATION")
    print(f"Evaluating across {n_rounds:,} customer interaction rounds...")
    print("=" * 70)

    rounds = load_training_rounds()[:n_rounds]

    # Load trained models
    linucb = load_linucb(LINUCB_PREFIX)
    thompson = load_thompson(THOMPSON_PREFIX)

    policies = {
        "Thompson Sampling": lambda ctx, cust, elig: thompson.select_action(ctx, cust, elig),
        "Disjoint LinUCB":   lambda ctx, cust, elig: linucb.select_action(ctx, cust, elig),
        "Rule-Based":        lambda ctx, cust, elig: rule_based_baseline({"demand_index": 2 if ctx.get("DemandIndex", 0.5)>=0.67 else (1 if ctx.get("DemandIndex", 0.5)>=0.34 else 0)}, elig),
        "Fixed 10%":         lambda ctx, cust, elig: fixed_baseline(elig, target=10),
        "Uniform Random":    lambda ctx, cust, elig: random_baseline(elig),
    }

    # Step 1: Pre-calculate historical logging policy and outcomes
    # Logging policy in real retail is typically random exploration over eligible actions
    history = []
    np.random.seed(42)

    for r in rounds:
        ctx = r["context"]
        cust = r["customer"]
        elig = r["eligible_discounts"]
        seed = r["round_seed"]

        # Logging policy: uniform random pick over eligible actions
        k = len(elig)
        propensity = 1.0 / k
        logged_arm = int(np.random.choice(elig))

        # Direct model estimate r_hat(x, a) for all eligible arms
        r_hat = {a: get_expected_profit(ctx, cust, a, seed=seed) for a in elig}

        # Simulated outcome for logged arm
        from simulator import simulate_outcome
        _, _, realized_reward = simulate_outcome(ctx, cust, logged_arm, round_seed=seed)

        history.append({
            "context": ctx,
            "customer": cust,
            "eligible": elig,
            "logged_arm": logged_arm,
            "propensity": propensity,
            "reward": realized_reward,
            "r_hat": r_hat,
        })

    # Step 2: Compute Direct Method (DM), Replay/IPS, and Doubly Robust (DR) estimates
    results = []
    for pol_name, pol_fn in policies.items():
        dr_values = []
        dm_values = []
        ips_values = []
        matches = 0

        for h in history:
            ctx = h["context"]
            cust = h["customer"]
            elig = h["eligible"]
            a_logged = h["logged_arm"]
            p_logged = h["propensity"]
            r_realized = h["reward"]
            r_hat = h["r_hat"]

            # Candidate policy decision
            a_pi = pol_fn(ctx, cust, elig)
            if a_pi is None:
                a_pi = elig[0]

            # Direct method estimate for policy's choice
            r_hat_pi = r_hat.get(a_pi, 0.0)
            dm_values.append(r_hat_pi)

            # Inverse Propensity Scoring (Replay) term
            is_match = (a_pi == a_logged)
            if is_match:
                matches += 1
                ips_term = r_realized / p_logged
                residual_term = (r_realized - r_hat.get(a_logged, 0.0)) / p_logged
            else:
                ips_term = 0.0
                residual_term = 0.0

            ips_values.append(ips_term)
            dr_values.append(r_hat_pi + residual_term)

        dr_arr = np.array(dr_values)
        dr_mean = float(np.mean(dr_arr))
        dr_sem = float(np.std(dr_arr, ddof=1) / np.sqrt(len(dr_arr)))
        ci_95 = 1.96 * dr_sem

        results.append({
            "Policy": pol_name,
            "Doubly Robust Value (Rs/round)": dr_mean,
            "DR 95% CI (+/- Rs)": ci_95,
            "Direct Method (Rs/round)": float(np.mean(dm_values)),
            "Replay Matches": matches,
            "Replay Value (Rs/round)": float(np.mean(ips_values)),
        })

    df_res = pd.DataFrame(results).sort_values("Doubly Robust Value (Rs/round)", ascending=False)

    print("\n" + "=" * 80)
    print("DOUBLY ROBUST (DR) BENCHMARKING SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<22} {'DR Value (Rs)':>15} {'95% CI':>12} {'Direct Method':>15} {'Replay Matches':>15}")
    print("-" * 80)
    for _, row in df_res.iterrows():
        print(f"{row['Policy']:<22} Rs {row['Doubly Robust Value (Rs/round)']:>12.4f}  +/- {row['DR 95% CI (+/- Rs)']:>7.4f} Rs {row['Direct Method (Rs/round)']:>12.4f} {row['Replay Matches']:>13,d}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    df_res.to_csv(REPORT_PATH, index=False)
    print(f"\nSaved Doubly Robust evaluation report to {REPORT_PATH}")
    return df_res


if __name__ == "__main__":
    evaluate_doubly_robust()
