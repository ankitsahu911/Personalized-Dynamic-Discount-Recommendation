"""
Milestone 39 - Reward Ablation Experiment (Profit vs. Profit - lambda * DiscountAmount)
Retail Discount-Recommendation Bandit Project

Explores penalized reward formulation:
    R_lambda = Profit - lambda * DiscountAmountRupees
where:
    DiscountAmountRupees = Price * (d / 100.0)
    penalty_on = 'offered' (default) or 'purchased'

Evaluates on TUNING_SEEDS = [100, 101, 102, 103, 104] across 20,000 rounds.

Generates:
    results/reward_ablation.csv
    outputs/plots/ablation_discount_vs_lambda.png
    outputs/plots/ablation_tradeoff.png
    outputs/plots/ablation_discount_mix.png
    outputs/plots/ablation_regret.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import make_sequence, TUNING_SEEDS, N_ROUNDS
from linucb import DisjointLinUCB, CONTEXT_DIM
from data_loader import get_context_probabilities, DISCOUNT_LEVELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Implied shares of mean per-round profit (~3.85 Rs):
# Mean discount amount across 5 arms is 52.76 * 0.10 = 5.276 Rs.
# lambda = (share * 3.8493) / 5.276
# 0%   -> 0.0
# 1%   -> 0.007
# 10%  -> 0.073
# 50%  -> 0.365
# 100% -> 0.730
LAMBDA_GRID = [
    {"lambda": 0.0,   "penalty_on": "offered",   "implied_share": "0% (Pure Profit)"},
    {"lambda": 0.007, "penalty_on": "offered",   "implied_share": "1% of Profit"},
    {"lambda": 0.073, "penalty_on": "offered",   "implied_share": "10% of Profit"},
    {"lambda": 0.365, "penalty_on": "offered",   "implied_share": "50% of Profit"},
    {"lambda": 0.730, "penalty_on": "offered",   "implied_share": "100% of Profit"},
    {"lambda": 0.073, "penalty_on": "purchased", "implied_share": "10% of Profit (Purchased Only)"},
]


def run_ablation_round(
    bandit: DisjointLinUCB,
    rounds: list,
    lam: float,
    penalty_on: str = "offered"
) -> pd.DataFrame:
    """
    Executes an online training run under penalized reward R_lambda.
    """
    records = []
    cum_profit = 0.0
    cum_revenue = 0.0
    cum_disc_given = 0.0
    cum_reward_lam = 0.0
    cum_regret_lam = 0.0

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
        chosen_disc = bandit.select_action(ctx, cust, elig)
        assert chosen_disc in elig, f"Discount {chosen_disc} outside {elig}!"

        # 2. Outcome simulation
        probs = get_context_probabilities(cat, tier, demand_idx, promo, seed=round_seed)
        prob = probs.get(chosen_disc, 0.20)

        rng = np.random.default_rng(abs(round_seed * 10007 + chosen_disc * 31) % (2**32))
        purchased = int(rng.random() < prob)
        units_sold = 1 if purchased else 0

        selling_price = round(price * (1.0 - chosen_disc / 100.0), 2)
        revenue = round(selling_price * units_sold, 2)
        profit = round((selling_price - cost_price) * units_sold, 2)
        disc_amount_rs = round(price * (chosen_disc / 100.0), 2)
        disc_given_rs = round(disc_amount_rs * units_sold, 2)

        # 3. Penalized reward calculation
        if penalty_on == "offered":
            r_lambda = profit - lam * disc_amount_rs
        elif penalty_on == "purchased":
            r_lambda = profit - lam * disc_given_rs
        else:
            raise ValueError(f"Unknown penalty_on mode: {penalty_on}")

        # Update bandit with R_lambda
        bandit.update(chosen_disc, ctx, cust, reward=r_lambda)

        # 4. Oracle & regret under ITS OWN reward R_lambda
        exp_rewards_lam = {}
        for a in elig:
            p_a = probs.get(a, 0.0)
            sp_a = price * (1.0 - a / 100.0)
            exp_prof_a = p_a * (sp_a - cost_price)
            d_amt_a = price * (a / 100.0)

            if penalty_on == "offered":
                exp_r_lam = exp_prof_a - lam * d_amt_a
            else:
                exp_r_lam = exp_prof_a - lam * (d_amt_a * p_a)
            exp_rewards_lam[a] = exp_r_lam

        oracle_disc_lam = max(elig, key=lambda a: exp_rewards_lam[a])
        oracle_exp_lam = exp_rewards_lam[oracle_disc_lam]
        chosen_exp_lam = exp_rewards_lam[chosen_disc]
        regret_lam = max(0.0, oracle_exp_lam - chosen_exp_lam)

        # Accumulate
        cum_profit += profit
        cum_revenue += revenue
        cum_disc_given += disc_given_rs
        cum_reward_lam += r_lambda
        cum_regret_lam += regret_lam

        records.append({
            "round_idx": round_idx,
            "customer_tier": tier,
            "chosen_discount": chosen_disc,
            "purchased": purchased,
            "revenue": revenue,
            "profit": profit,
            "disc_given_rs": disc_given_rs,
            "reward_lambda": r_lambda,
            "regret_lambda": regret_lam,
            "cum_profit": cum_profit,
            "cum_disc_given": cum_disc_given,
            "cum_regret_lambda": cum_regret_lam,
        })

    return pd.DataFrame(records)


def run_milestone39():
    print("=" * 80)
    print("MILESTONE 39: REWARD ABLATION EXPERIMENT (PROFIT vs PROFIT - lambda * DISCOUNT)")
    print("=" * 80)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 0: SCALE THE GRID TO RUPEES
    # -----------------------------------------------------------------------
    print("\n[Step 0: Scale Penalty Parameter lambda to Rupees]")
    mean_prof = 3.8493
    mean_disc_rs = 5.276
    print(f"  Mean per-round profit = Rs {mean_prof:.4f}")
    print(f"  Mean per-round discount across 5 arms = Rs {mean_disc_rs:.4f}")
    print("\n  Configured Lambda Grid:")
    for item in LAMBDA_GRID:
        lam = item["lambda"]
        p_on = item["penalty_on"]
        ishare = item["implied_share"]
        eff_penalty = lam * mean_disc_rs
        print(f"    lambda={lam:<6.3f} (penalty_on={p_on:<9}): penalty/round ~ Rs {eff_penalty:.3f} ({ishare})")

    # -----------------------------------------------------------------------
    # STEP 1 & 2: TRAIN AND EVALUATE ACROSS TUNING SEEDS
    # -----------------------------------------------------------------------
    print("\n[Step 1 & 2: Executing Ablation Sweep Across Tuning Seeds]")
    start_time = time.time()

    all_seed_runs = []
    # Store regret curves for plotting: cfg_name -> (rounds, mean_curve)
    regret_curves = {}

    for cfg_idx, cfg in enumerate(LAMBDA_GRID):
        lam = cfg["lambda"]
        p_on = cfg["penalty_on"]
        cfg_name = f"lam_{lam}_{p_on}"
        print(f"\n  Running config {cfg_idx+1}/{len(LAMBDA_GRID)}: lambda={lam}, penalty_on={p_on}...")

        seed_regrets = []

        for seed in TUNING_SEEDS:
            seq = make_sequence(seed=seed, n_rounds=N_ROUNDS)
            bandit = DisjointLinUCB(arms=DISCOUNT_LEVELS, context_dim=CONTEXT_DIM, alpha=1.0, use_sherman_morrison=True)

            df_run = run_ablation_round(bandit, seq, lam=lam, penalty_on=p_on)

            tot_profit = float(df_run["profit"].sum())
            tot_revenue = float(df_run["revenue"].sum())
            tot_disc_given = float(df_run["disc_given_rs"].sum())
            conv_rate = float(df_run["purchased"].mean() * 100.0)
            avg_disc_offered = float(df_run["chosen_discount"].mean())
            tot_r_lambda = float(df_run["reward_lambda"].sum())
            final_regret_lambda = float(df_run["cum_regret_lambda"].iloc[-1])

            # Tier breakdown
            tier_stats = {}
            for t in ["New", "Regular", "Loyal"]:
                sub_t = df_run[df_run["customer_tier"] == t]
                tier_stats[f"{t}_avg_disc"] = float(sub_t["chosen_discount"].mean())
                for d in DISCOUNT_LEVELS:
                    tier_stats[f"{t}_share_{d}pct"] = float((sub_t["chosen_discount"] == d).mean() * 100.0)

            # Check for collapse to minimum eligible discount
            # Min eligible: New -> 0%, Regular -> 5%, Loyal -> 10%
            is_collapsed = True
            for _, row_s in df_run.iterrows():
                t = row_s["customer_tier"]
                d = row_s["chosen_discount"]
                min_d = 0 if t == "New" else (5 if t == "Regular" else 10)
                if d != min_d:
                    is_collapsed = False
                    break

            seed_regrets.append(df_run["cum_regret_lambda"].values)

            rec = {
                "lambda": lam,
                "penalty_on": p_on,
                "implied_share": cfg["implied_share"],
                "seed": seed,
                "total_profit_rs": tot_profit,
                "total_revenue_rs": tot_revenue,
                "conversion_rate_pct": conv_rate,
                "avg_discount_offered_pct": avg_disc_offered,
                "total_discount_given_rs": tot_disc_given,
                "total_reward_lambda": tot_r_lambda,
                "final_regret_lambda": final_regret_lambda,
                "is_collapsed_to_minimum": int(is_collapsed),
            }
            rec.update(tier_stats)
            all_seed_runs.append(rec)

        regret_curves[cfg_name] = np.mean(seed_regrets, axis=0)

    elapsed = time.time() - start_time
    print(f"\nAblation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

    df_all = pd.DataFrame(all_seed_runs)

    # -----------------------------------------------------------------------
    # STEP 3: SUMMARY TABLE (results/reward_ablation.csv)
    # -----------------------------------------------------------------------
    print("\n[Step 3: Compiling Summary CSV]")
    summary_records = []
    
    for cfg in LAMBDA_GRID:
        lam = cfg["lambda"]
        p_on = cfg["penalty_on"]
        sub = df_all[(df_all["lambda"] == lam) & (df_all["penalty_on"] == p_on)]

        row = {
            "lambda": lam,
            "penalty_on": p_on,
            "implied_share": cfg["implied_share"],
            "total_profit_mean_rs": float(sub["total_profit_rs"].mean()),
            "total_profit_std_rs": float(sub["total_profit_rs"].std()),
            "total_revenue_mean_rs": float(sub["total_revenue_rs"].mean()),
            "total_revenue_std_rs": float(sub["total_revenue_rs"].std()),
            "conversion_rate_mean_pct": float(sub["conversion_rate_pct"].mean()),
            "conversion_rate_std_pct": float(sub["conversion_rate_pct"].std()),
            "avg_discount_offered_mean_pct": float(sub["avg_discount_offered_pct"].mean()),
            "avg_discount_offered_std_pct": float(sub["avg_discount_offered_pct"].std()),
            "total_discount_given_mean_rs": float(sub["total_discount_given_rs"].mean()),
            "total_discount_given_std_rs": float(sub["total_discount_given_rs"].std()),
            "final_regret_lambda_mean": float(sub["final_regret_lambda"].mean()),
            "final_regret_lambda_std": float(sub["final_regret_lambda"].std()),
            "share_collapsed_pct": float(sub["is_collapsed_to_minimum"].mean() * 100.0),
        }
        for t in ["New", "Regular", "Loyal"]:
            row[f"{t}_avg_disc_mean"] = float(sub[f"{t}_avg_disc"].mean())
        summary_records.append(row)

    summary_df = pd.DataFrame(summary_records)
    out_csv = os.path.join(RESULTS_DIR, "reward_ablation.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"  Saved summary: {out_csv}")

    # -----------------------------------------------------------------------
    # STEP 4: PRICE OF RESTRAINT TABLE
    # -----------------------------------------------------------------------
    print("\n[Step 4: Price of Restraint Table]")
    base_prof = summary_df.loc[0, "total_profit_mean_rs"]
    base_disc = summary_df.loc[0, "total_discount_given_mean_rs"]

    restraint_rows = []
    print(f"{'Config':<28} | {'Delta Profit (Rs)':<18} | {'Delta Disc Given (Rs)':<22} | {'Rs Profit Sacrificed / Rs Disc Avoided'}")
    print("-" * 95)
    for idx, r in summary_df.iterrows():
        lam = r["lambda"]
        p_on = r["penalty_on"]
        p_mean = r["total_profit_mean_rs"]
        d_mean = r["total_discount_given_mean_rs"]

        delta_prof = p_mean - base_prof  # negative if profit lost
        delta_disc_avoided = base_disc - d_mean  # positive if discount saved
        ratio = (-delta_prof / delta_disc_avoided) if delta_disc_avoided > 0 else 0.0

        tag = f"lambda={lam} ({p_on})"
        print(f"{tag:<28} | {delta_prof:>17,.2f} | {delta_disc_avoided:>21,.2f} | {ratio:>32.4f}")
        restraint_rows.append({
            "config": tag,
            "lambda": lam,
            "penalty_on": p_on,
            "profit_change_rs": delta_prof,
            "discount_avoided_rs": delta_disc_avoided,
            "sacrifice_ratio": ratio
        })

    # -----------------------------------------------------------------------
    # STEP 5 & 6: PLOTS
    # -----------------------------------------------------------------------
    print("\n[Generating Plots]")

    # (a) avg discount offered (%) vs lambda
    plt.figure(figsize=(8, 5))
    offered_sub = summary_df[summary_df["penalty_on"] == "offered"]
    plt.errorbar(offered_sub["lambda"], offered_sub["avg_discount_offered_mean_pct"],
                 yerr=offered_sub["avg_discount_offered_std_pct"], fmt="o-", color="#1f77b4",
                 lw=2, capsize=4, markersize=7, label="Penalty on Offered")
    # Mark the purchased-only point
    purch_sub = summary_df[summary_df["penalty_on"] == "purchased"]
    plt.scatter(purch_sub["lambda"], purch_sub["avg_discount_offered_mean_pct"],
                color="#d62728", s=80, zorder=5, label="Penalty on Purchased (lambda=0.073)")
    plt.xlabel("Penalty Parameter lambda", fontsize=11)
    plt.ylabel("Average Discount Offered (%)", fontsize=11)
    plt.title("Effect of Penalty lambda on Average Discount Offered", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "ablation_discount_vs_lambda.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  Saved plot: {p1}")

    # (b) Trade-off Curve: x = total discount given, y = total pure profit
    plt.figure(figsize=(8, 5.5))
    for _, r in summary_df.iterrows():
        lam = r["lambda"]
        p_on = r["penalty_on"]
        x = r["total_discount_given_mean_rs"]
        x_err = r["total_discount_given_std_rs"]
        y = r["total_profit_mean_rs"]
        y_err = r["total_profit_std_rs"]

        c = "#d62728" if p_on == "purchased" else "#1f77b4"
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", color=c, capsize=4, markersize=8)
        lbl = f"lambda={lam} ({'purch' if p_on=='purchased' else 'off'})"
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, -4), fontsize=9)

    plt.xlabel("Total Discount Given (Simulated Rs)", fontsize=11)
    plt.ylabel("Total Pure Profit (Simulated Rs)", fontsize=11)
    plt.title("Trade-off Curve: Discount Given vs Pure Profit", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "ablation_tradeoff.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  Saved plot: {p2}")

    # (c) Stacked bars of discount shares per lambda
    plt.figure(figsize=(10, 5.5))
    disc_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    labels = [f"lam={r['lambda']}\n({r['penalty_on'][:4]})" for _, r in summary_df.iterrows()]
    x_pos = np.arange(len(labels))

    # Compute overall discount shares per config across runs
    shares_by_arm = {d: [] for d in DISCOUNT_LEVELS}
    for cfg in LAMBDA_GRID:
        lam = cfg["lambda"]
        p_on = cfg["penalty_on"]
        sub = df_all[(df_all["lambda"] == lam) & (df_all["penalty_on"] == p_on)]
        # calculate mean shares across all tiers
        # We can extract from tier-shares or total
        s_total = {d: 0.0 for d in DISCOUNT_LEVELS}
        for d in DISCOUNT_LEVELS:
            # Weighted average across tiers
            s_d = (sub[f"New_share_{d}pct"].mean() * 0.35 +
                   sub[f"Regular_share_{d}pct"].mean() * 0.45 +
                   sub[f"Loyal_share_{d}pct"].mean() * 0.20)
            shares_by_arm[d].append(s_d)

    bottom = np.zeros(len(labels))
    for idx_d, d in enumerate(DISCOUNT_LEVELS):
        vals = np.array(shares_by_arm[d])
        plt.bar(x_pos, vals, bottom=bottom, color=disc_colors[idx_d], label=f"{d}% Discount", alpha=0.85, width=0.55)
        bottom += vals

    plt.xticks(x_pos, labels)
    plt.xlabel("Penalty Configuration", fontsize=11)
    plt.ylabel("Share of Rounds Chosen (%)", fontsize=11)
    plt.title("Policy Discount Distribution Under Penalized Reward", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "ablation_discount_mix.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  Saved plot: {p3}")

    # (d) Cumulative regret_lambda per lambda
    plt.figure(figsize=(9, 5.5))
    rx = np.arange(1, N_ROUNDS + 1)
    for cfg in LAMBDA_GRID:
        c_name = f"lam_{cfg['lambda']}_{cfg['penalty_on']}"
        curve = regret_curves[c_name]
        lbl = f"lambda={cfg['lambda']} ({cfg['penalty_on']})"
        plt.plot(rx, curve, lw=2, label=lbl)

    plt.xlabel("Round Number", fontsize=11)
    plt.ylabel("Cumulative Regret (in respective R_lambda units)", fontsize=11)
    plt.title("Cumulative Regret Under Policy-Specific Penalized Rewards", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "ablation_regret.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  Saved plot: {p4}")

    # -----------------------------------------------------------------------
    # STEP 6: INTERPRETATION
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: INTERPRETATION (RIGOROUS AUDIT SUMMARY)")
    print("=" * 80)
    print(f"1. When lambda=0.0 (profit-only baseline), LinUCB offered an average discount of {summary_df.loc[0, 'avg_discount_offered_mean_pct']:.2f}% and gave Rs {summary_df.loc[0, 'total_discount_given_mean_rs']:,.2f} in total discounts, generating Rs {summary_df.loc[0, 'total_profit_mean_rs']:,.2f} profit.")
    print(f"2. Introducing lambda=0.073 (penalty on offered) reduced average discount to {summary_df.loc[2, 'avg_discount_offered_mean_pct']:.2f}% and saved Rs {restraint_rows[2]['discount_avoided_rs']:,.2f} in discounts, sacrificing Rs {abs(restraint_rows[2]['profit_change_rs']):,.2f} profit ({restraint_rows[2]['sacrifice_ratio']:.3f} Rs lost per Rs saved).")
    print(f"3. Applying the penalty only to purchased items (penalty_on='purchased' at lambda=0.073) resulted in an average discount of {summary_df.loc[5, 'avg_discount_offered_mean_pct']:.2f}%, demonstrating less aggressive reduction than penalizing all offers because unexercised offers incur zero penalty.")
    print(f"4. At high penalty (lambda=0.730), average discount fell to {summary_df.loc[4, 'avg_discount_offered_mean_pct']:.2f}%, with total discount given dropping to Rs {summary_df.loc[4, 'total_discount_given_mean_rs']:,.2f}, representing substantial margin suppression.")
    print(f"5. Crucially, whether a penalty lambda > 0 is justified represents an external business and modeling choice (such as brand equity preservation), not an empirical finding from reward maximization.")
    print("=" * 80)


if __name__ == "__main__":
    run_milestone39()
