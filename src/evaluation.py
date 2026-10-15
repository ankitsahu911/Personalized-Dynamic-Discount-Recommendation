"""
Milestone 33 - Reusable Evaluation Harness & Multi-Seed Experiment
Retail Discount-Recommendation Bandit Project

Implements:
    - make_sequence(seed, n_rounds): deterministic, cached round sequences.
    - Policy wrappers for the 5 policies with Online / Frozen support.
    - run_policy(policy, rounds, learning=True): standard log schema execution.
    - Multi-seed Online (10 seeds x 20,000 rounds) & Frozen (10 seeds x 5,000 rounds) benchmarking.
    - Strict integrity checks and NPZ cumulative array exports.

Outputs:
    results/multiseed_online_metrics.csv
    results/multiseed_online_by_tier.csv
    results/cumulative_by_seed.npz
    results/multiseed_frozen_metrics.csv
"""

import os
import sys
import time
import pickle
import random
import hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from round_generator import Customer, build_demo_registry
from live_generator import generate_live_rounds
from fixed_baseline import fixed_baseline
from rule_based_baseline import rule_based_baseline
from random_baseline import random_baseline
from linucb import DisjointLinUCB, CONTEXT_DIM
from thompson_sampling import ThompsonSampling
from data_loader import get_context_probabilities, DISCOUNT_LEVELS

N_ROUNDS = 20000
N_HELDOUT = 5000
MAIN_SEED = 42
EVAL_SEEDS = list(range(10))  # [0, 1, ..., 9]
TUNING_SEEDS = [100, 101, 102, 103, 104]
POLICY_ORDER = ["Random", "Fixed 10%", "Rule-based", "LinUCB", "Thompson Sampling"]

ROUNDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rounds")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# 1. SEQUENCE GENERATOR (Cached)
# ---------------------------------------------------------------------------

def make_sequence(seed: int, n_rounds: int = N_ROUNDS) -> List[Dict]:
    """
    Produce a deterministic, cached round sequence.
    Each round is a dict with: round_idx, context, customer, eligible_discounts, round_seed.
    """
    cache_path = os.path.join(ROUNDS_DIR, f"eval_rounds_seed{seed}_n{n_rounds}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    os.makedirs(ROUNDS_DIR, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    registry = build_demo_registry(n=50)
    gen = generate_live_rounds(registry)

    rounds = []
    for i in range(n_rounds):
        ctx, cust, elig = next(gen)
        # Snapshot copy of customer state
        cust_copy = Customer(
            CustomerID=cust.CustomerID,
            PurchaseCount=cust.PurchaseCount,
            RecencyDays=cust.RecencyDays,
            LoyaltyTier=cust.LoyaltyTier,
            CurrentBasketValue=cust.CurrentBasketValue,
        )
        round_seed = int(abs(seed * 100003 + i * 37) % (2**31 - 1))
        rounds.append({
            "round_idx": i,
            "context": dict(ctx),
            "customer": cust_copy,
            "eligible_discounts": list(elig),
            "round_seed": round_seed,
        })

    with open(cache_path, "wb") as f:
        pickle.dump(rounds, f, protocol=pickle.HIGHEST_PROTOCOL)

    return rounds


# ---------------------------------------------------------------------------
# 2. POLICY WRAPPER
# ---------------------------------------------------------------------------

class PolicyWrapper:
    """
    Common interface for all 5 policies supporting both ONLINE and FROZEN modes.
    """
    def __init__(self, name: str, policy_type: str, seed: Optional[int] = None, alpha: float = 1.0, sigma: float = 1.0):
        self.name = name
        self.policy_type = policy_type
        self.frozen = False
        self.seed = seed

        if policy_type == "random":
            self.rng = random.Random(seed if seed is not None else 42)
        elif policy_type == "fixed":
            pass
        elif policy_type == "rule_based":
            pass
        elif policy_type == "linucb":
            self.model = DisjointLinUCB(arms=DISCOUNT_LEVELS, context_dim=CONTEXT_DIM, alpha=alpha, use_sherman_morrison=True)
        elif policy_type == "thompson":
            # Use per-instance rng_seed instead of global np.random.seed() to prevent
            # cross-contamination between Thompson's sampling and the outcome RNG.
            self.model = ThompsonSampling(arms=DISCOUNT_LEVELS, context_dim=CONTEXT_DIM, sigma=sigma, rng_seed=seed)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    def select(self, context: Dict, customer: Customer, eligible: List[int]) -> int:
        if self.policy_type == "random":
            return self.rng.choice(eligible)
        elif self.policy_type == "fixed":
            return fixed_baseline(eligible, target=10)
        elif self.policy_type == "rule_based":
            raw_demand = context.get("DemandIndex", 0.5)
            d_idx = 2 if raw_demand >= 0.67 else (1 if raw_demand >= 0.34 else 0)
            return rule_based_baseline({"demand_index": d_idx}, eligible)
        elif self.policy_type in ("linucb", "thompson"):
            return self.model.select_action(context, customer, eligible)

    def update(self, chosen_discount: int, context: Dict, customer: Customer, reward: float):
        if self.frozen:
            return
        if self.policy_type in ("linucb", "thompson"):
            self.model.update(chosen_discount, context, customer, reward)


# ---------------------------------------------------------------------------
# 3. RUN POLICY (Standard Log Schema)
# ---------------------------------------------------------------------------

def run_policy(policy: PolicyWrapper, rounds: List[Dict], learning: bool = True) -> pd.DataFrame:
    """
    Execute policy on round sequence and return a DataFrame in the Standard Log Schema.
    """
    policy.frozen = not learning
    rows = []

    cum_rev = 0.0
    cum_prof = 0.0
    cum_disc_given = 0.0
    cum_reg = 0.0

    for r in rounds:
        round_idx = r["round_idx"]
        ctx = r["context"]
        cust = r["customer"]
        elig = r["eligible_discounts"]
        round_seed = r["round_seed"]

        price = float(ctx["Price"])
        cost_price = float(ctx.get("CostPrice", price * 0.60))
        tier = getattr(cust, "LoyaltyTier", "") or "New"
        demand_idx = float(ctx.get("DemandIndex", 0.5))
        promo = int(ctx.get("Promotion", 0))
        cat = ctx.get("Category", "Electronics")

        # 1. Action selection
        chosen_disc = policy.select(ctx, cust, elig)
        assert chosen_disc in elig, f"Policy {policy.name} chose {chosen_disc} outside {elig} at round {round_idx}!"

        # 2. Outcome simulation
        probs = get_context_probabilities(cat, tier, demand_idx, promo, seed=round_seed)
        prob = probs.get(chosen_disc, 0.20)

        # Deterministic outcome draw
        rng = np.random.default_rng(abs(round_seed * 10007 + chosen_disc * 31) % (2**32))
        purchased = int(rng.random() < prob)
        units_sold = 1 if purchased else 0

        selling_price = round(price * (1.0 - chosen_disc / 100.0), 2)
        revenue = round(selling_price * units_sold, 2)
        profit = round((selling_price - cost_price) * units_sold, 2)
        disc_amount_rs = round(price * (chosen_disc / 100.0), 2)
        disc_given_rs = round(disc_amount_rs * units_sold, 2)

        # 3. Policy update
        policy.update(chosen_disc, ctx, cust, reward=profit)

        # 4. Oracle & regret calculation
        exp_profits = {}
        for a in elig:
            p_a = probs.get(a, 0.0)
            sp_a = price * (1.0 - a / 100.0)
            exp_profits[a] = p_a * (sp_a - cost_price)

        oracle_disc = max(elig, key=lambda a: exp_profits[a])
        oracle_exp = exp_profits[oracle_disc]
        chosen_exp = exp_profits[chosen_disc]
        regret = max(0.0, oracle_exp - chosen_exp)

        # 5. Accumulate running metrics
        cum_rev += revenue
        cum_prof += profit
        cum_disc_given += disc_given_rs
        cum_reg += regret

        rows.append({
            "round_idx": round_idx,
            "customer_tier": tier,
            "price": price,
            "cost_price": cost_price,
            "eligible_discounts": ",".join(map(str, elig)),
            "chosen_discount": chosen_disc,
            "purchased": purchased,
            "units_sold": units_sold,
            "revenue": revenue,
            "profit": profit,
            "discount_amount_rupees": disc_amount_rs,
            "discount_given_total_rupees": disc_given_rs,
            "oracle_discount": oracle_disc,
            "oracle_expected_profit": round(oracle_exp, 4),
            "chosen_expected_profit": round(chosen_exp, 4),
            "regret": round(regret, 4),
            "cum_revenue": round(cum_rev, 2),
            "cum_profit": round(cum_prof, 2),
            "cum_discount_given": round(cum_disc_given, 2),
            "cum_regret": round(cum_reg, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. MULTI-SEED EXPERIMENT RUNNER
# ---------------------------------------------------------------------------

def run_multiseed_experiment(seeds: List[int] = EVAL_SEEDS, n_rounds: int = N_ROUNDS, n_heldout: int = N_HELDOUT):
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("MILESTONE 33: MULTI-SEED BENCHMARKING (ONLINE & FROZEN)")
    print(f"Seeds: {seeds} | Rounds: {n_rounds:,} | Held-out: {n_heldout:,}")
    print("=" * 80)

    online_metrics_list = []
    online_tier_list = []
    frozen_metrics_list = []

    # NPZ arrays: (n_seeds, n_policies, n_rounds)
    n_seeds = len(seeds)
    n_policies = len(POLICY_ORDER)
    cum_profit_arr = np.zeros((n_seeds, n_policies, n_rounds), dtype=np.float64)
    cum_rev_arr = np.zeros((n_seeds, n_policies, n_rounds), dtype=np.float64)
    cum_disc_arr = np.zeros((n_seeds, n_policies, n_rounds), dtype=np.float64)
    cum_reg_arr = np.zeros((n_seeds, n_policies, n_rounds), dtype=np.float64)

    policy_timing = {p: 0.0 for p in POLICY_ORDER}

    for s_idx, seed in enumerate(seeds):
        t_seed_start = time.time()
        print(f"\n>>> Running Seed {seed} ({s_idx + 1} / {n_seeds})...")

        # 1. Build identical sequence for this seed
        seq_online = make_sequence(seed=seed, n_rounds=n_rounds)
        seq_heldout = make_sequence(seed=1000 + seed, n_rounds=n_heldout)

        # Integrity hash of online sequence
        seq_hash = hashlib.md5(str([(r["context"]["Price"], r["eligible_discounts"]) for r in seq_online]).encode()).hexdigest()

        # Instantiate fresh policies for this seed
        policies_online = [
            PolicyWrapper("Random", "random", seed=seed + 101),
            PolicyWrapper("Fixed 10%", "fixed"),
            PolicyWrapper("Rule-based", "rule_based"),
            PolicyWrapper("LinUCB", "linucb", alpha=1.0),
            PolicyWrapper("Thompson Sampling", "thompson", seed=seed + 707, sigma=1.0),
        ]

        # STEP 2: ONLINE RUN
        online_dfs = {}
        for p_idx, pol in enumerate(policies_online):
            t_pol_start = time.time()
            df_log = run_policy(pol, seq_online, learning=True)
            elapsed = time.time() - t_pol_start
            policy_timing[pol.name] += elapsed
            online_dfs[pol.name] = df_log

            # Store in NPZ arrays
            cum_profit_arr[s_idx, p_idx, :] = df_log["cum_profit"].values
            cum_rev_arr[s_idx, p_idx, :] = df_log["cum_revenue"].values
            cum_disc_arr[s_idx, p_idx, :] = df_log["cum_discount_given"].values
            cum_reg_arr[s_idx, p_idx, :] = df_log["cum_regret"].values

            # Summary metrics
            conv = int(df_log["purchased"].sum())
            total_prof = float(df_log["profit"].sum())
            total_rev = float(df_log["revenue"].sum())
            total_disc_rs = float(df_log["discount_given_total_rupees"].sum())
            cum_regret = float(df_log["cum_regret"].iloc[-1])
            avg_reg_last2k = float(df_log["regret"].iloc[-2000:].mean())
            pct_oracle = float((df_log["chosen_discount"] == df_log["oracle_discount"]).mean() * 100.0)

            online_metrics_list.append({
                "seed": seed,
                "policy": pol.name,
                "total_revenue": total_rev,
                "total_profit": total_prof,
                "conversions": conv,
                "conversion_rate": conv / n_rounds * 100.0,
                "avg_discount_pct_offered": float(df_log["chosen_discount"].mean()),
                "avg_discount_amount_rupees_offered": float(df_log["discount_amount_rupees"].mean()),
                "total_discount_given_rupees": total_disc_rs,
                "cumulative_regret": cum_regret,
                "avg_regret_last_2000_rounds": avg_reg_last2k,
                "pct_rounds_chose_oracle": pct_oracle,
                "seq_hash": seq_hash,
            })

            # By Tier Breakdown
            for tier, df_tier in df_log.groupby("customer_tier"):
                t_rounds = len(df_tier)
                t_conv = int(df_tier["purchased"].sum())
                online_tier_list.append({
                    "seed": seed,
                    "policy": pol.name,
                    "customer_tier": tier,
                    "tier_rounds": t_rounds,
                    "tier_total_profit": float(df_tier["profit"].sum()),
                    "tier_avg_profit_per_round": float(df_tier["profit"].mean()),
                    "tier_conversion_rate": t_conv / t_rounds * 100.0,
                    "tier_avg_discount_offered": float(df_tier["chosen_discount"].mean()),
                    "tier_avg_regret_per_round": float(df_tier["regret"].mean()),
                    "tier_pct_chose_oracle": float((df_tier["chosen_discount"] == df_tier["oracle_discount"]).mean() * 100.0),
                })

        # STEP 4: FROZEN RUN on Held-out Sequence
        # Freeze trained LinUCB & Thompson Sampling from online run
        policies_frozen = [
            PolicyWrapper("Random", "random", seed=seed + 202),
            PolicyWrapper("Fixed 10%", "fixed"),
            PolicyWrapper("Rule-based", "rule_based"),
            policies_online[3],  # trained LinUCB
            policies_online[4],  # trained Thompson Sampling
        ]

        for pol in policies_frozen:
            df_frozen = run_policy(pol, seq_heldout, learning=False)
            f_conv = int(df_frozen["purchased"].sum())
            f_prof = float(df_frozen["profit"].sum())
            f_rev = float(df_frozen["revenue"].sum())
            f_disc_rs = float(df_frozen["discount_given_total_rupees"].sum())
            f_reg = float(df_frozen["cum_regret"].iloc[-1])

            frozen_metrics_list.append({
                "seed": seed,
                "policy": pol.name,
                "heldout_rounds": n_heldout,
                "total_revenue": f_rev,
                "total_profit": f_prof,
                "profit_per_round": f_prof / n_heldout,
                "discount_given_per_round": f_disc_rs / n_heldout,
                "conversions": f_conv,
                "conversion_rate": f_conv / n_heldout * 100.0,
                "avg_discount_pct_offered": float(df_frozen["chosen_discount"].mean()),
                "total_discount_given_rupees": f_disc_rs,
                "cumulative_regret": f_reg,
            })

        t_seed_elapsed = time.time() - t_seed_start
        print(f"  Finished Seed {seed} in {t_seed_elapsed:.1f}s.")

    # Save CSVs & NPZ
    df_online_metrics = pd.DataFrame(online_metrics_list)
    df_online_tier = pd.DataFrame(online_tier_list)
    df_frozen_metrics = pd.DataFrame(frozen_metrics_list)

    online_path = os.path.join(RESULTS_DIR, "multiseed_online_metrics.csv")
    tier_path = os.path.join(RESULTS_DIR, "multiseed_online_by_tier.csv")
    frozen_path = os.path.join(RESULTS_DIR, "multiseed_frozen_metrics.csv")
    npz_path = os.path.join(RESULTS_DIR, "cumulative_by_seed.npz")

    df_online_metrics.to_csv(online_path, index=False)
    df_online_tier.to_csv(tier_path, index=False)
    df_frozen_metrics.to_csv(frozen_path, index=False)

    np.savez_compressed(
        npz_path,
        cum_profit=cum_profit_arr,
        cum_revenue=cum_rev_arr,
        cum_discount_given=cum_disc_arr,
        cum_regret=cum_reg_arr,
        policy_order=np.array(POLICY_ORDER),
        seeds=np.array(seeds),
    )

    print(f"\n[Artifacts Saved]:")
    print(f"  - {online_path}")
    print(f"  - {tier_path}")
    print(f"  - {frozen_path}")
    print(f"  - {npz_path}")

    # -----------------------------------------------------------------------
    # STEP 3: SUMMARY TABLES
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: MULTI-SEED SUMMARY (ONLINE MODE -- 10 SEEDS, MEAN +/- STD)")
    print("=" * 80)

    summary_rows = []
    for pol_name in POLICY_ORDER:
        sub = df_online_metrics[df_online_metrics["policy"] == pol_name]
        p_m, p_s = sub["total_profit"].mean(), sub["total_profit"].std()
        r_m, r_s = sub["total_revenue"].mean(), sub["total_revenue"].std()
        c_m, c_s = sub["conversion_rate"].mean(), sub["conversion_rate"].std()
        d_m, d_s = sub["avg_discount_pct_offered"].mean(), sub["avg_discount_pct_offered"].std()
        dg_m, dg_s = sub["total_discount_given_rupees"].mean(), sub["total_discount_given_rupees"].std()
        reg_m, reg_s = sub["cumulative_regret"].mean(), sub["cumulative_regret"].std()

        summary_rows.append({
            "Policy": pol_name,
            "Total Profit (Rs)": f"Rs {p_m:,.2f} +/- {p_s:,.2f}",
            "Total Revenue (Rs)": f"Rs {r_m:,.2f} +/- {r_s:,.2f}",
            "Conversion Rate": f"{c_m:.2f}% +/- {c_s:.2f}%",
            "Avg Discount Offered": f"{d_m:.2f}% +/- {d_s:.2f}%",
            "Total Discount Given (Rs)": f"Rs {dg_m:,.2f} +/- {dg_s:,.2f}",
            "Cumulative Regret (Rs)": f"Rs {reg_m:,.2f} +/- {reg_s:,.2f}",
        })

    print(pd.DataFrame(summary_rows).to_string(index=False))

    # PAIRED COMPARISONS
    print("\n" + "-" * 80)
    print("PAIRED TOTAL-PROFIT COMPARISONS (Per-Seed Differences in Simulated Rs)")
    print("-" * 80)
    p_fixed = df_online_metrics[df_online_metrics["policy"] == "Fixed 10%"].set_index("seed")["total_profit"]
    p_rule = df_online_metrics[df_online_metrics["policy"] == "Rule-based"].set_index("seed")["total_profit"]
    p_lin = df_online_metrics[df_online_metrics["policy"] == "LinUCB"].set_index("seed")["total_profit"]
    p_ts = df_online_metrics[df_online_metrics["policy"] == "Thompson Sampling"].set_index("seed")["total_profit"]

    for b_name, b_s in [("LinUCB", p_lin), ("Thompson Sampling", p_ts)]:
        # vs Fixed 10%
        diff_f = b_s - p_fixed
        win_f = int((diff_f > 0).sum())
        print(f"  {b_name} vs. Fixed 10%  : Diff = Rs {diff_f.mean():+,.2f} +/- {diff_f.std():,.2f} (Won in {win_f} of {n_seeds} seeds)")

        # vs Rule-based
        diff_r = b_s - p_rule
        win_r = int((diff_r > 0).sum())
        print(f"  {b_name} vs. Rule-based : Diff = Rs {diff_r.mean():+,.2f} +/- {diff_r.std():,.2f} (Won in {win_r} of {n_seeds} seeds)")

    # STEP 4: FROZEN SUMMARY
    print("\n" + "=" * 80)
    print("STEP 4: MULTI-SEED SUMMARY (FROZEN MODE -- 5,000 HELDOUT ROUNDS)")
    print("=" * 80)
    frozen_rows = []
    for pol_name in POLICY_ORDER:
        sub = df_frozen_metrics[df_frozen_metrics["policy"] == pol_name]
        p_m, p_s = sub["total_profit"].mean(), sub["total_profit"].std()
        pp_m, pp_s = sub["profit_per_round"].mean(), sub["profit_per_round"].std()
        c_m, c_s = sub["conversion_rate"].mean(), sub["conversion_rate"].std()
        reg_m, reg_s = sub["cumulative_regret"].mean(), sub["cumulative_regret"].std()

        frozen_rows.append({
            "Policy": pol_name,
            "Total Profit (Rs)": f"Rs {p_m:,.2f} +/- {p_s:,.2f}",
            "Profit/Round (Rs)": f"Rs {pp_m:.4f} +/- {pp_s:.4f}",
            "Conversion Rate": f"{c_m:.2f}% +/- {c_s:.2f}%",
            "Cumulative Regret (Rs)": f"Rs {reg_m:,.2f} +/- {reg_s:,.2f}",
        })
    print(pd.DataFrame(frozen_rows).to_string(index=False))

    # STEP 5: INTEGRITY CHECKS
    print("\n" + "=" * 80)
    print("STEP 5: INTEGRITY CHECKS")
    print("=" * 80)

    # 1. Regret never negative
    all_regrets = df_online_metrics["cumulative_regret"].values
    assert (all_regrets >= 0).all(), "Negative cumulative regret found!"
    print("  1. Cumulative regret >= 0 for all seeds and policies: PASS")

    # 2. Sequence hash identity within seed
    hash_match = True
    for s in seeds:
        hashes = df_online_metrics[df_online_metrics["seed"] == s]["seq_hash"].unique()
        if len(hashes) != 1:
            hash_match = False
    print(f"  2. Identical sequence hashes within every seed: {'PASS' if hash_match else 'FAIL'}")

    # 3. Final cumulative values in NPZ equal metrics
    npz_data = np.load(npz_path)
    npz_profit_match = np.allclose(
        npz_data["cum_profit"][:, :, -1].flatten(),
        df_online_metrics["total_profit"].values,
        atol=1e-2
    )
    print(f"  3. NPZ final values match CSV total profit exactly: {'PASS' if npz_profit_match else 'FAIL'}")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"STEP 6: RUNTIME -- Total Runtime: {total_time:.1f}s ({total_time/60:.2f} mins)")
    for p, t_sec in policy_timing.items():
        print(f"  {p:<20} : {t_sec:.1f}s total ({t_sec/n_seeds:.2f}s per 20k run)")
    print("=" * 80)


if __name__ == "__main__":
    run_multiseed_experiment()
