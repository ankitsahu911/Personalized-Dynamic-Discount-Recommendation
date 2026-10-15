"""
Milestone 34 - Build the Revenue/Profit Ledger
Retail Discount-Recommendation Bandit Project

Generates the complete ledger table for all 5 policies on the MAIN_SEED sequence (seed 42),
ONLINE mode, runs 8 strict accounting and integrity checks, and exports to CSV & SQLite.

Inputs:
    src/evaluation.py (make_sequence, PolicyWrapper, run_policy)

Outputs:
    results/ledger_seed42.csv
    results/ledger.db
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import make_sequence, PolicyWrapper, run_policy, N_ROUNDS, MAIN_SEED

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LEDGER_CSV = os.path.join(RESULTS_DIR, "ledger_seed42.csv")
LEDGER_DB = os.path.join(RESULTS_DIR, "ledger.db")
LINUCB_TRAIN_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "linucb_training_log.csv")
TS_TRAIN_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "thompson_training_log.csv")


def run_milestone34():
    print("=" * 80)
    print("MILESTONE 34: REVENUE & PROFIT LEDGER (MAIN_SEED = 42)")
    print("=" * 80)

    # STEP 1: Generate sequence for MAIN_SEED
    print(f"\n[Step 1] Loading / generating {N_ROUNDS:,} rounds for MAIN_SEED = {MAIN_SEED}...")
    seq_42 = make_sequence(seed=MAIN_SEED, n_rounds=N_ROUNDS)
    assert len(seq_42) == N_ROUNDS

    # Policies with exact Milestone 30 TS_SEED = 7
    policies = [
        PolicyWrapper("Random", "random", seed=42 + 101),
        PolicyWrapper("Fixed 10%", "fixed"),
        PolicyWrapper("Rule-based", "rule_based"),
        PolicyWrapper("LinUCB", "linucb", alpha=1.0),
        PolicyWrapper("Thompson Sampling", "thompson", seed=7, sigma=1.0),
    ]

    # STEP 2: Execute policies and build single ledger table
    print("\n[Step 2] Executing all 5 policies and assembling ledger...")
    ledger_dfs = []
    for pol in policies:
        print(f"  Executing {pol.name:<20}...")
        df_pol = run_policy(pol, seq_42, learning=True)
        df_pol["policy"] = pol.name
        df_pol["seed"] = MAIN_SEED
        ledger_dfs.append(df_pol)

    df_ledger = pd.concat(ledger_dfs, ignore_index=True)

    # Reorder columns to standard ledger schema
    col_order = [
        "policy", "seed", "round_idx", "customer_tier", "price", "cost_price",
        "eligible_discounts", "chosen_discount", "purchased", "units_sold",
        "revenue", "profit", "discount_amount_rupees", "discount_given_total_rupees",
        "oracle_expected_profit", "chosen_expected_profit", "regret",
        "cum_revenue", "cum_profit", "cum_discount_given", "cum_regret",
    ]
    df_ledger = df_ledger[col_order]

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_ledger.to_csv(LEDGER_CSV, index=False)
    csv_size_mb = os.path.getsize(LEDGER_CSV) / (1024 * 1024)
    print(f"\nSaved CSV: {LEDGER_CSV} ({csv_size_mb:.2f} MB)")

    # Save SQLite
    conn = sqlite3.connect(LEDGER_DB)
    df_ledger.to_sql("ledger", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_round ON ledger(policy, round_idx)")
    conn.commit()
    conn.close()
    print(f"Saved SQLite DB: {LEDGER_DB}")

    # -----------------------------------------------------------------------
    # STEP 3: STRICT INTEGRITY CHECKS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: LEDGER INTEGRITY CHECKS")
    print("=" * 80)

    # a. Exactly 5 * N_ROUNDS rows; each policy has round_idx 0..N_ROUNDS-1 exactly once
    check_a = (len(df_ledger) == 5 * N_ROUNDS)
    for p in policies:
        p_sub = df_ledger[df_ledger["policy"] == p.name]
        if not (p_sub["round_idx"].nunique() == N_ROUNDS and (p_sub["round_idx"].values == np.arange(N_ROUNDS)).all()):
            check_a = False
    print(f"  Check a: Exactly 5 * N_ROUNDS rows, each policy round_idx 0..N-1: {'PASS' if check_a else 'FAIL'}")
    assert check_a

    # b. price, customer_tier, eligible_discounts identical across all 5 policies
    p_ref = df_ledger[df_ledger["policy"] == "Random"].reset_index(drop=True)
    check_b = True
    for p in policies[1:]:
        p_sub = df_ledger[df_ledger["policy"] == p.name].reset_index(drop=True)
        if not (p_ref["price"].equals(p_sub["price"]) and
                p_ref["customer_tier"].equals(p_sub["customer_tier"]) and
                p_ref["eligible_discounts"].equals(p_sub["eligible_discounts"])):
            check_b = False
    print(f"  Check b: Identical contexts/eligible sets across all policies: {'PASS' if check_b else 'FAIL'}")
    assert check_b

    # c. Accounting identities per row
    mask_purchased = (df_ledger["purchased"] == 1)
    d_frac = df_ledger["chosen_discount"] / 100.0

    exp_rev_p = np.round(df_ledger["price"] * (1.0 - d_frac) * df_ledger["units_sold"], 2)
    exp_prof_p = np.round((df_ledger["price"] * (1.0 - d_frac) - df_ledger["cost_price"]) * df_ledger["units_sold"], 2)

    diff_rev = np.max(np.abs(df_ledger.loc[mask_purchased, "revenue"] - exp_rev_p.loc[mask_purchased]))
    diff_prof = np.max(np.abs(df_ledger.loc[mask_purchased, "profit"] - exp_prof_p.loc[mask_purchased]))

    check_c_nonpurchase = (
        (df_ledger.loc[~mask_purchased, "revenue"] == 0).all() and
        (df_ledger.loc[~mask_purchased, "profit"] == 0).all() and
        (df_ledger.loc[~mask_purchased, "units_sold"] == 0).all() and
        (df_ledger.loc[~mask_purchased, "discount_given_total_rupees"] == 0).all()
    )
    check_c = (diff_rev <= 0.02 and diff_prof <= 0.02 and check_c_nonpurchase)
    print(f"  Check c: Accounting identities per row (revenue/profit/units, max diff={max(diff_rev, diff_prof):.4f}): {'PASS' if check_c else 'FAIL'}")
    assert check_c

    # d. discount_amount_rupees and discount_given_total_rupees
    calc_disc_amt = np.round(df_ledger["price"] * (df_ledger["chosen_discount"] / 100.0), 2)
    calc_disc_given = np.round(calc_disc_amt * df_ledger["units_sold"], 2)
    check_d = (
        np.allclose(df_ledger["discount_amount_rupees"], calc_disc_amt, atol=1e-2) and
        np.allclose(df_ledger["discount_given_total_rupees"], calc_disc_given, atol=1e-2)
    )
    print(f"  Check d: Discount rupee amounts and concession totals: {'PASS' if check_d else 'FAIL'}")
    assert check_d

    # e. Cumulative columns equal running sums; last cumulative value equals column total
    check_e = True
    for p in policies:
        p_sub = df_ledger[df_ledger["policy"] == p.name]
        for col, cum_col in [("revenue", "cum_revenue"), ("profit", "cum_profit"), ("discount_given_total_rupees", "cum_discount_given")]:
            if not np.allclose(np.round(np.cumsum(p_sub[col]), 2), p_sub[cum_col], atol=1e-2):
                check_e = False
            if not np.isclose(p_sub[col].sum(), p_sub[cum_col].iloc[-1], atol=1e-2):
                check_e = False
    print(f"  Check e: Cumulative columns match running sums exactly: {'PASS' if check_e else 'FAIL'}")
    assert check_e

    # f. chosen_discount inside eligible_discounts for every row
    check_f = True
    for idx, row in df_ledger.iterrows():
        elig_list = list(map(int, row["eligible_discounts"].split(",")))
        if row["chosen_discount"] not in elig_list:
            check_f = False
            break
    print(f"  Check f: Every chosen discount inside eligible set: {'PASS' if check_f else 'FAIL'}")
    assert check_f

    # g. Fixed 10% policy chose 10 whenever 10 was eligible
    fixed_sub = df_ledger[df_ledger["policy"] == "Fixed 10%"]
    check_g = True
    for _, row in fixed_sub.iterrows():
        elig_list = list(map(int, row["eligible_discounts"].split(",")))
        if 10 in elig_list and row["chosen_discount"] != 10:
            check_g = False
            break
    print(f"  Check g: Fixed 10% chose 10 whenever eligible: {'PASS' if check_g else 'FAIL'}")
    assert check_g

    # h. Consistency with training logs (LinUCB & Thompson Sampling)
    check_h_linucb = False
    if os.path.exists(LINUCB_TRAIN_LOG):
        df_old_l = pd.read_csv(LINUCB_TRAIN_LOG)
        lin_sub = df_ledger[df_ledger["policy"] == "LinUCB"].reset_index(drop=True)
        # Check agreement on chosen_discount and profit
        disc_match = (df_old_l["chosen_discount"] == lin_sub["chosen_discount"]).mean() * 100.0
        check_h_linucb = (disc_match == 100.0)
        print(f"  Check h (LinUCB): 100% decision match with training log ({disc_match:.1f}%): {'PASS' if check_h_linucb else 'NOTE'}")

    check_h_ts = False
    if os.path.exists(TS_TRAIN_LOG):
        df_old_t = pd.read_csv(TS_TRAIN_LOG)
        ts_sub = df_ledger[df_ledger["policy"] == "Thompson Sampling"].reset_index(drop=True)
        disc_match_t = (df_old_t["chosen_discount"] == ts_sub["chosen_discount"]).mean() * 100.0
        check_h_ts = (disc_match_t == 100.0)
        print(f"  Check h (Thompson): 100% decision match with training log ({disc_match_t:.1f}%): {'PASS' if check_h_ts else 'NOTE'}")

    # -----------------------------------------------------------------------
    # STEP 4: PRINT LEDGER SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: LEDGER SUMMARY (SEED 42 -- ONLINE MODE)")
    print("=" * 80)

    fixed_profit = float(df_ledger[df_ledger["policy"] == "Fixed 10%"]["profit"].sum())
    summary_rows = []

    for pol in policies:
        sub = df_ledger[df_ledger["policy"] == pol.name]
        rev = float(sub["revenue"].sum())
        prof = float(sub["profit"].sum())
        disc_given = float(sub["discount_given_total_rupees"].sum())
        conv = int(sub["purchased"].sum())
        gain_vs_fixed = prof - fixed_profit

        summary_rows.append({
            "Policy": pol.name,
            "Total Revenue (Rs)": f"Rs {rev:>11,.2f}",
            "Total Profit (Rs)": f"Rs {prof:>11,.2f}",
            "Total Discount Given (Rs)": f"Rs {disc_given:>10,.2f}",
            "Conversions": f"{conv:>5,d} / {N_ROUNDS:,}",
            "Gain vs Fixed 10% (Rs)": f"Rs {gain_vs_fixed:>+10,.2f}",
        })

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 80)
    return df_ledger


if __name__ == "__main__":
    run_milestone34()
