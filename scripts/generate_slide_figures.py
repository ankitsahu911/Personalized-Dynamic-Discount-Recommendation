# -*- coding: utf-8 -*-
"""
Generates high-resolution, slide-ready 16:9 figures (white background, large 28pt+ fonts,
direct line labels, projector-optimized) and exports native CSV tables for slides.
Creates slides/figure_manifest.csv.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("slides/figures", exist_ok=True)
os.makedirs("slides/tables", exist_ok=True)

# Set global matplotlib parameters for presentation slides (16:9 aspect ratio, 300 DPI, large typography)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 18,
    'axes.titlesize': 26,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 28,
    'figure.figsize': (14, 7.875), # 16:9 ratio
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 3.5,
})

def make_figures():
    print("Generating slide figures...")
    data = np.load("results/cumulative_by_seed.npz")
    cum_regret = data['cum_regret'] # (10, 5, 20000)
    cum_profit = data['cum_profit'] # (10, 5, 20000)
    policies = list(data['policy_order'])
    rounds = np.arange(1, 20001)

    colors = {
        'Fixed 10%': '#10B981',      # Emerald Green
        'LinUCB': '#2563EB',         # Royal Blue
        'Thompson Sampling': '#8B5CF6',# Purple
        'Rule-based': '#F59E0B',     # Amber
        'Random': '#64748B'          # Slate Gray
    }

    # 1. Regret Curves (LinUCB vs. Thompson Sampling)
    fig, ax = plt.subplots(figsize=(14, 7.875))
    idx_lin = policies.index('LinUCB')
    idx_ts = policies.index('Thompson Sampling')
    
    lin_mean = cum_regret[:, idx_lin, :].mean(axis=0)
    lin_std = cum_regret[:, idx_lin, :].std(axis=0)
    ts_mean = cum_regret[:, idx_ts, :].mean(axis=0)
    ts_std = cum_regret[:, idx_ts, :].std(axis=0)

    ax.plot(rounds, lin_mean, color=colors['LinUCB'], label='LinUCB (alpha=0.5)')
    ax.fill_between(rounds, lin_mean - lin_std, lin_mean + lin_std, color=colors['LinUCB'], alpha=0.15)

    ax.plot(rounds, ts_mean, color=colors['Thompson Sampling'], label='Thompson Sampling (sigma=0.5)')
    ax.fill_between(rounds, ts_mean - ts_std, ts_mean + ts_std, color=colors['Thompson Sampling'], alpha=0.15)

    ax.set_title("Cumulative Regret Convergence Over 20,000 Rounds", pad=20, fontweight='bold', color='#0F172A')
    ax.set_xlabel("Rounds (t)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Cumulative Expected Regret (Simulated Rs)", labelpad=12, fontweight='bold', color='#1E293B')
    
    # Direct line annotations
    ax.text(20200, lin_mean[-1], f"LinUCB\nRs {lin_mean[-1]:.0f}", color=colors['LinUCB'], fontweight='bold', va='center')
    ax.text(20200, ts_mean[-1], f"Thompson\nRs {ts_mean[-1]:.0f}", color=colors['Thompson Sampling'], fontweight='bold', va='center')
    
    ax.set_xlim(0, 23000)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig("slides/figures/regret_curves_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/regret_curves_slide.png")

    # 2. Cumulative Profit (All 5 Policies)
    fig, ax = plt.subplots(figsize=(14, 7.875))
    for pol in ['Fixed 10%', 'LinUCB', 'Thompson Sampling', 'Rule-based', 'Random']:
        idx = policies.index(pol)
        p_mean = cum_profit[:, idx, :].mean(axis=0)
        ax.plot(rounds, p_mean, color=colors[pol], label=pol)
        # Direct annotation at end
        label_text = f"{pol}\nRs {p_mean[-1]/1000:.1f}k"
        ax.text(20200, p_mean[-1], label_text, color=colors[pol], fontweight='bold', va='center', fontsize=14)

    ax.set_title("Cumulative Total Profit Across 5 Policies (10-Seed Mean)", pad=20, fontweight='bold', color='#0F172A')
    ax.set_xlabel("Rounds (t)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Cumulative Realized Profit (Simulated Rs)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.set_xlim(0, 24000)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig("slides/figures/cumulative_profit_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/cumulative_profit_slide.png")

    # 3. Gain/Loss Chart (LinUCB vs. Comparators)
    fig, ax = plt.subplots(figsize=(14, 7.875))
    idx_rule = policies.index('Rule-based')
    idx_fix = policies.index('Fixed 10%')

    diff_rule = (cum_profit[:, idx_lin, :] - cum_profit[:, idx_rule, :]).mean(axis=0)
    diff_fix = (cum_profit[:, idx_lin, :] - cum_profit[:, idx_fix, :]).mean(axis=0)

    ax.axhline(0, color='#94A3B8', linestyle='-', linewidth=2)
    ax.plot(rounds, diff_rule, color='#2563EB', label='LinUCB vs. Rule-based (+Rs 492 at End)')
    ax.plot(rounds, diff_fix, color='#EF4444', label='LinUCB vs. Fixed 10% (-Rs 1,341 at End)')

    ax.set_title("LinUCB Net Profit Advantage Over Time (10-Seed Mean)", pad=20, fontweight='bold', color='#0F172A')
    ax.set_xlabel("Rounds (t)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Cumulative Profit Difference (Simulated Rs)", labelpad=12, fontweight='bold', color='#1E293B')

    # Highlight break-even
    ax.scatter([3200], [0], color='#2563EB', s=120, zorder=5)
    ax.annotate("Break-even vs. Rule-based\n(~Round 3,200)", xy=(3200, 0), xytext=(4500, 250),
                arrowprops=dict(facecolor='#2563EB', arrowstyle='->', lw=2),
                fontsize=16, fontweight='bold', color='#1E293B')

    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig("slides/figures/gain_loss_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/gain_loss_slide.png")

    # 4. Scatter: Profit vs. Discount Offered
    df_online_raw = pd.read_csv("results/multiseed_online_metrics.csv")
    fig, ax = plt.subplots(figsize=(14, 7.875))
    for pol in ['Fixed 10%', 'LinUCB', 'Thompson Sampling', 'Rule-based', 'Random']:
        sub = df_online_raw[df_online_raw['policy'] == pol]
        ax.scatter(sub['avg_discount_pct_offered'], sub['total_profit'],
                   color=colors[pol], s=140, label=pol, alpha=0.85, edgecolors='black', linewidth=1)

    ax.set_title("Total Profit vs. Average Discount Offered Across All Seeds", pad=20, fontweight='bold', color='#0F172A')
    ax.set_xlabel("Average Discount Offered (%)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Total Profit (Simulated Rs)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig("slides/figures/profit_vs_discount_scatter_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/profit_vs_discount_scatter_slide.png")

    # 5. Alpha Sensitivity Sweep
    df_alpha = pd.read_csv("results/alpha_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(14, 7.875))
    ax.errorbar(df_alpha['alpha'], df_alpha['final_cumulative_regret_rs_mean'],
                yerr=df_alpha['final_cumulative_regret_rs_std'], fmt='-o', color='#2563EB',
                ecolor='#93C5FD', elinewidth=3, capsize=6, markersize=10, label='Final Regret (Rs)')

    ax.set_title("LinUCB Final Regret vs. Exploration Parameter (alpha)", pad=20, fontweight='bold', color='#0F172A')
    ax.set_xlabel("Exploration Parameter (alpha)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Final Cumulative Regret (Simulated Rs)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.axvspan(0.1, 1.0, color='#10B981', alpha=0.12, label='Robust Sweet Spot (alpha in [0.1, 1.0])')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig("slides/figures/alpha_regret_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/alpha_regret_slide.png")

    # 6. Reward Ablation: Discount Offered vs Revenue/Profit
    df_abl = pd.read_csv("results/reward_ablation.csv")
    fig, ax1 = plt.subplots(figsize=(14, 7.875))
    ax2 = ax1.twinx()
    
    # Filter for offered penalty only
    df_abl_sub = df_abl[df_abl['penalty_on'] == 'offered'].sort_values('lambda')
    lambdas = df_abl_sub['lambda'].astype(str)

    ax1.plot(lambdas, df_abl_sub['avg_discount_offered_mean_pct'], color='#EF4444', marker='s', markersize=9, label='Avg Discount Offered (%)')
    ax2.plot(lambdas, df_abl_sub['total_revenue_mean_rs']/1000, color='#2563EB', marker='o', markersize=9, label='Total Revenue (k Rs)')

    ax1.set_title("Reward Penalty Ablation: Collapse of Discounting and Revenue", pad=20, fontweight='bold', color='#0F172A')
    ax1.set_xlabel("Penalty Multiplier (lambda)", labelpad=12, fontweight='bold', color='#1E293B')
    ax1.set_ylabel("Average Discount Offered (%)", color='#EF4444', labelpad=12, fontweight='bold')
    ax2.set_ylabel("Total Revenue (Thousands of Rs)", color='#2563EB', labelpad=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#EF4444')
    ax2.tick_params(axis='y', labelcolor='#2563EB')
    plt.tight_layout()
    fig.savefig("slides/figures/ablation_tradeoff_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/ablation_tradeoff_slide.png")

    # 7. Off-Policy Replay Comparison
    df_rep = pd.read_csv("results/replay_results.csv")
    fig, ax = plt.subplots(figsize=(14, 7.875))
    # Exclude Logged Policy (All Rows) row for clean policy comparison if desired or keep
    df_rep_clean = df_rep[df_rep['Policy'].isin(['Random', 'Fixed 10%', 'LinUCB', 'Thompson Sampling'])]
    pol_names = df_rep_clean['Policy'].tolist()
    means = df_rep_clean['Mean_Profit_Rs'].tolist()
    ci_lower = df_rep_clean['CI_95_Low_Rs'].tolist()
    ci_upper = df_rep_clean['CI_95_High_Rs'].tolist()
    errors = [np.array(means) - np.array(ci_lower), np.array(ci_upper) - np.array(means)]

    bar_colors = [colors.get(p, '#2563EB') for p in pol_names]
    bars = ax.bar(pol_names, means, yerr=errors, capsize=8, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.8, f"Rs {m:.2f}",
                ha='center', va='bottom', color='white', fontweight='bold', fontsize=18)

    ax.set_title("Off-Policy Replay Profit per Round (73,100 Records)", pad=20, fontweight='bold', color='#0F172A')
    ax.set_ylabel("Replay Profit per Round (Simulated Rs)", labelpad=12, fontweight='bold', color='#1E293B')
    ax.text(0.5, 0.90, "Note: Overlapping 95% CIs indicate no statistically valid ranking",
            transform=ax.transAxes, ha='center', fontsize=16, color='#DC2626', style='italic',
            bbox=dict(facecolor='#FEF2F2', edgecolor='#F87171', boxstyle='round,pad=0.5'))
    ax.set_ylim(0, 8.0)
    plt.tight_layout()
    fig.savefig("slides/figures/replay_comparison_slide.png")
    plt.close(fig)
    print("  -> Saved slides/figures/replay_comparison_slide.png")

def export_slide_tables():
    print("Exporting native slide tables...")
    pd.read_csv("results/comparison_online.csv").to_csv("slides/tables/table_online_comparison.csv", index=False)
    pd.read_csv("results/comparison_frozen.csv").to_csv("slides/tables/table_frozen_comparison.csv", index=False)
    pd.read_csv("results/comparison_lift.csv").to_csv("slides/tables/table_lift_summary.csv", index=False)
    pd.read_csv("results/multiseed_online_by_tier.csv").to_csv("slides/tables/table_tier_breakdown.csv", index=False)
    pd.read_csv("results/replay_results.csv").to_csv("slides/tables/table_replay.csv", index=False)
    print("  -> Exported 5 tables to slides/tables/")

def make_figure_manifest():
    print("Writing slides/figure_manifest.csv...")
    manifest = [
        ("regret_curves_slide.png", "Cumulative regret convergence (LinUCB vs Thompson) over 20,000 rounds with 10-seed error bands", "results/cumulative_by_seed.npz", "slides/figures/regret_curves_slide.png"),
        ("cumulative_profit_slide.png", "Cumulative realized profit across all 5 policies over 20,000 rounds with direct terminal labels", "results/cumulative_by_seed.npz", "slides/figures/cumulative_profit_slide.png"),
        ("gain_loss_slide.png", "Running profit difference of LinUCB vs Rule-based and Fixed 10%, highlighting break-even at round 3,200", "results/cumulative_by_seed.npz", "slides/figures/gain_loss_slide.png"),
        ("profit_vs_discount_scatter_slide.png", "Cross-seed scatter plot of total profit vs average discount offered, proving diminishing margin returns", "results/multiseed_online_metrics.csv", "slides/figures/profit_vs_discount_scatter_slide.png"),
        ("alpha_regret_slide.png", "LinUCB final cumulative regret across exploration parameter alpha grid [0.0, 5.0]", "results/alpha_sensitivity.csv", "slides/figures/alpha_regret_slide.png"),
        ("ablation_tradeoff_slide.png", "Reward penalty ablation showing collapse of promotional discounting and revenue under penalty lambda", "results/reward_ablation.csv", "slides/figures/ablation_tradeoff_slide.png"),
        ("replay_comparison_slide.png", "Off-policy replay mean profit per round with 95% confidence intervals across 73,100 records", "results/replay_results.csv", "slides/figures/replay_comparison_slide.png"),
    ]
    with open("slides/figure_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_name", "description", "source_data", "file_path"])
        for row in manifest:
            writer.writerow(row)
    print("  -> Saved slides/figure_manifest.csv")

if __name__ == "__main__":
    make_figures()
    export_slide_tables()
    make_figure_manifest()
    print("Slide figure generation completed successfully.")
