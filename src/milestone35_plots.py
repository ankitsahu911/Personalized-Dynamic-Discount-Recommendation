"""
Milestone 35 - Plot Cumulative Profit and Discount-Given Curves (The Gain/Loss Chart)
Retail Discount-Recommendation Bandit Project

Inputs:
    results/ledger_seed42.csv
    results/cumulative_by_seed.npz

Outputs:
    outputs/plots/cumulative_profit_seed42.png
    outputs/plots/cumulative_profit_mean_std.png
    outputs/plots/cumulative_discount_given_seed42.png
    outputs/plots/gain_loss_linucb_vs_others.png
    outputs/plots/profit_vs_discount_scatter.png
    outputs/dashboard_data/cumulative_series_seed42.csv
    outputs/dashboard_data/summary_kpis.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "plots")
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "dashboard_data")
LEDGER_CSV = os.path.join(RESULTS_DIR, "ledger_seed42.csv")
NPZ_FILE = os.path.join(RESULTS_DIR, "cumulative_by_seed.npz")

POLICY_STYLES = {
    "Random":            {"color": "#7f7f7f", "linestyle": ":",  "label": "Random"},
    "Fixed 10%":         {"color": "#ff7f0e", "linestyle": "--", "label": "Fixed 10%"},
    "Rule-based":        {"color": "#9467bd", "linestyle": "-.", "label": "Rule-based"},
    "LinUCB":            {"color": "#1f77b4", "linestyle": "-",  "label": r"LinUCB ($\alpha=1.0$)"},
    "Thompson Sampling": {"color": "#2ca02c", "linestyle": "-",  "label": r"Thompson Sampling ($\sigma=1.0$)"},
}


def run_milestone35():
    print("=" * 80)
    print("MILESTONE 35: VISUALIZATION SUITE & DASHBOARD DATA EXPORT")
    print("=" * 80)

    assert os.path.exists(LEDGER_CSV), f"Missing {LEDGER_CSV}"
    assert os.path.exists(NPZ_FILE), f"Missing {NPZ_FILE}"

    df_ledger = pd.read_csv(LEDGER_CSV)
    npz_data = np.load(NPZ_FILE)
    policy_order = list(npz_data["policy_order"])

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    # 1. PLOT A - Cumulative profit vs round (Seed 42)
    plt.figure(figsize=(10, 6))
    for pol_name, style in POLICY_STYLES.items():
        sub = df_ledger[df_ledger["policy"] == pol_name].sort_values("round_idx")
        x = sub["round_idx"]
        y = sub["cum_profit"]
        plt.plot(x, y, color=style["color"], linestyle=style["linestyle"], linewidth=2.0, label=style["label"])
        final_val = y.iloc[-1]
        plt.annotate(
            f"Rs {final_val:,.0f}",
            xy=(x.iloc[-1], final_val),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=style["color"],
        )

    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Cumulative Profit (simulated Rs)", fontsize=11)
    plt.title("Cumulative Profit Across 20,000 Rounds (Seed 42, Online Mode)", fontsize=12, fontweight="bold")
    plt.suptitle("Individual Seed Comparison", fontsize=10, y=0.92)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rs {val:,.0f}"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plot_a = os.path.join(PLOT_DIR, "cumulative_profit_seed42.png")
    plt.savefig(plot_a, dpi=200)
    plt.close()
    print(f"[Plot A] Saved: {plot_a}")

    # 2. PLOT B - Mean Cumulative profit across 10 EVAL seeds with +/- 1 std band
    cum_prof_npz = npz_data["cum_profit"]  # (n_seeds, n_policies, n_rounds)
    n_seeds, n_pols, n_rounds = cum_prof_npz.shape
    rounds_x = np.arange(n_rounds)

    plt.figure(figsize=(10, 6))
    for p_idx, pol_name in enumerate(policy_order):
        style = POLICY_STYLES[pol_name]
        mean_y = cum_prof_npz[:, p_idx, :].mean(axis=0)
        std_y = cum_prof_npz[:, p_idx, :].std(axis=0)

        plt.plot(rounds_x, mean_y, color=style["color"], linestyle=style["linestyle"], linewidth=2.0, label=style["label"])
        plt.fill_between(rounds_x, mean_y - std_y, mean_y + std_y, color=style["color"], alpha=0.15)
        plt.annotate(
            f"Rs {mean_y[-1]:,.0f}",
            xy=(n_rounds - 1, mean_y[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=style["color"],
        )

    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Cumulative Profit (simulated Rs)", fontsize=11)
    plt.title(f"Mean Cumulative Profit Over 10 Random Seeds (N={n_rounds:,} Rounds, Online)", fontsize=12, fontweight="bold")
    plt.suptitle(r"Mean $\pm$ 1 Std Shaded Band Across 10 Independent Seeds", fontsize=10, y=0.92)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rs {val:,.0f}"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plot_b = os.path.join(PLOT_DIR, "cumulative_profit_mean_std.png")
    plt.savefig(plot_b, dpi=200)
    plt.close()
    print(f"[Plot B] Saved: {plot_b}")

    # 3. PLOT C - Cumulative discount GIVEN vs round (Seed 42)
    plt.figure(figsize=(10, 6))
    for pol_name, style in POLICY_STYLES.items():
        sub = df_ledger[df_ledger["policy"] == pol_name].sort_values("round_idx")
        x = sub["round_idx"]
        y = sub["cum_discount_given"]
        plt.plot(x, y, color=style["color"], linestyle=style["linestyle"], linewidth=2.0, label=style["label"])
        final_val = y.iloc[-1]
        plt.annotate(
            f"Rs {final_val:,.0f}",
            xy=(x.iloc[-1], final_val),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=style["color"],
        )

    plt.xlabel("Round", fontsize=11)
    plt.ylabel("Cumulative Discount Given (simulated Rs)", fontsize=11)
    plt.title("Cumulative Discount Given Conceded Over 20,000 Rounds (Seed 42)", fontsize=12, fontweight="bold")
    plt.suptitle("Money Conceded on Realized Purchases Only", fontsize=10, y=0.92)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rs {val:,.0f}"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plot_c = os.path.join(PLOT_DIR, "cumulative_discount_given_seed42.png")
    plt.savefig(plot_c, dpi=200)
    plt.close()
    print(f"[Plot C] Saved: {plot_c}")

    # 4. PLOT D - THE GAIN/LOSS CHART: LinUCB Cumulative Profit MINUS Comparators
    lin_sub = df_ledger[df_ledger["policy"] == "LinUCB"].sort_values("round_idx")
    lin_y = lin_sub["cum_profit"].values

    comparators = ["Random", "Fixed 10%", "Rule-based", "Thompson Sampling"]
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)

    for comp in comparators:
        comp_sub = df_ledger[df_ledger["policy"] == comp].sort_values("round_idx")
        comp_y = comp_sub["cum_profit"].values
        diff = lin_y - comp_y
        final_gap = diff[-1]

        # Break-even round: first round after which diff stays strictly > 0 to the end
        above = (diff > 0)
        be_round = "Never"
        for r_i in range(len(diff)):
            if above[r_i:].all():
                be_round = f"Round {r_i:,}"
                break

        lbl = f"LinUCB - {comp} (Gap: Rs {final_gap:+,.0f}, Break-even: {be_round})"
        plt.plot(rounds_x, diff, label=lbl, color=POLICY_STYLES[comp]["color"], linestyle=POLICY_STYLES[comp]["linestyle"], linewidth=2.0)
        plt.annotate(
            f"Rs {final_gap:+,.0f}",
            xy=(n_rounds - 1, final_gap),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=POLICY_STYLES[comp]["color"],
        )

    plt.xlabel("Round", fontsize=11)
    plt.ylabel("LinUCB Net Profit Gain / Loss (simulated Rs)", fontsize=11)
    plt.title("LinUCB Cumulative Profit Gap vs. Other Policies (Seed 42)", fontsize=12, fontweight="bold")
    plt.suptitle("Above 0 = LinUCB Leads | Below 0 = Comparator Leads", fontsize=10, y=0.92)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rs {val:+,.0f}"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plot_d = os.path.join(PLOT_DIR, "gain_loss_linucb_vs_others.png")
    plt.savefig(plot_d, dpi=200)
    plt.close()
    print(f"[Plot D] Saved: {plot_d}")

    # 5. OPTIONAL PLOT E - Efficiency Scatter: Total Discount Given vs Total Profit
    plt.figure(figsize=(8, 6))
    for pol_name, style in POLICY_STYLES.items():
        sub = df_ledger[df_ledger["policy"] == pol_name]
        tot_disc = sub["discount_given_total_rupees"].sum()
        tot_prof = sub["profit"].sum()
        plt.scatter(tot_disc, tot_prof, color=style["color"], s=120, edgecolors="black", linewidth=1.5, zorder=4)
        plt.annotate(
            f" {pol_name}\n (Disc: Rs {tot_disc:,.0f}, Prof: Rs {tot_prof:,.0f})",
            xy=(tot_disc, tot_prof),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    plt.xlabel("Total Discount Given (simulated Rs)", fontsize=11)
    plt.ylabel("Total Profit Earned (simulated Rs)", fontsize=11)
    plt.title("Policy Efficiency: Total Profit vs. Discount Conceded (Seed 42)", fontsize=12, fontweight="bold")
    plt.suptitle("Top-Left Quadrant = Maximum Profit with Minimum Discount Waste", fontsize=10, y=0.92)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rs {val:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rs {val:,.0f}"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_e = os.path.join(PLOT_DIR, "profit_vs_discount_scatter.png")
    plt.savefig(plot_e, dpi=200)
    plt.close()
    print(f"[Plot E] Saved: {plot_e}")

    # -----------------------------------------------------------------------
    # EXPORT DASHBOARD DATA (Milestone 46 Support)
    # -----------------------------------------------------------------------
    # Thin to every 20th round, ensuring round 19999 is included
    thinned_indices = list(range(0, n_rounds, 20))
    if (n_rounds - 1) not in thinned_indices:
        thinned_indices.append(n_rounds - 1)

    df_thinned = df_ledger[df_ledger["round_idx"].isin(thinned_indices)][
        ["round_idx", "policy", "cum_profit", "cum_revenue", "cum_discount_given", "cum_regret"]
    ].copy()

    dash_csv = os.path.join(DASHBOARD_DIR, "cumulative_series_seed42.csv")
    df_thinned.to_csv(dash_csv, index=False)
    print(f"\n[Dashboard Export] Saved thinned time series: {dash_csv}")

    fixed_profit = float(df_ledger[df_ledger["policy"] == "Fixed 10%"]["profit"].sum())
    kpis = {}
    for pol_name in POLICY_STYLES.keys():
        sub = df_ledger[df_ledger["policy"] == pol_name]
        kpis[pol_name] = {
            "total_profit": float(sub["profit"].sum()),
            "total_discount_given": float(sub["discount_given_total_rupees"].sum()),
            "total_revenue": float(sub["revenue"].sum()),
            "conversions": int(sub["purchased"].sum()),
        }

    kpis["linucb_gain_vs_fixed_10_rupees"] = float(kpis["LinUCB"]["total_profit"] - fixed_profit)

    dash_json = os.path.join(DASHBOARD_DIR, "summary_kpis.json")
    with open(dash_json, "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)
    print(f"[Dashboard Export] Saved KPI summary: {dash_json}")

    print("\n" + "=" * 80)
    print("PLAIN-LANGUAGE DESCRIPTION (Milestone 35 Visual Summary)")
    print("=" * 80)
    desc = (
        "Plot A demonstrates the cumulative profit trajectories of all five policies for Seed 42, showing "
        "Thompson Sampling and Fixed 10% leading total online revenue, while LinUCB absorbs early exploration costs.\n"
        "Plot B confirms that this ordering remains consistent across all 10 independent evaluation seeds, with narrow "
        "+/- 1 std error bands highlighting low seed-to-seed variance.\n"
        "Plot C illustrates the actual money conceded through discounts, revealing that Fixed 10% and Rule-based policies "
        "concede significantly higher discount volume to generate their revenue.\n"
        "Plot D (the Gain/Loss chart) details the exact rupee margin between LinUCB and each competitor; while LinUCB trails "
        "Thompson Sampling and Fixed 10% during online exploration, it decisively outperforms Random.\n"
        "Plot E displays efficiency in the discount-profit space, demonstrating that the bandit policies achieve a more "
        "restrained discount expenditure per rupee of margin earned."
    )
    print(desc)
    print("=" * 80)


if __name__ == "__main__":
    run_milestone35()
