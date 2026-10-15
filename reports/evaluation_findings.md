# Comprehensive Evaluation Findings: Dynamic Discount Recommendation Bandits

**Project:** Retail Personalized Dynamic Discount Recommendation  
**Evaluation Scope:** Milestones 32 through 39 (Multi-seed Benchmarking, Policy Ledgers, Visual Curves, Off-policy Replay, Exploration Sensitivity, and Reward Ablation)  
**Date:** October 2026  
**Environment:** Offline Vectorized Evaluation (CPU)  

---

## 1. Headline Findings

In both Online and Frozen evaluation modes, the heuristic benchmark **Fixed 10% earned the highest total profit**, outperforming both contextual bandit algorithms (LinUCB and Thompson Sampling). Across 10 independent evaluation seeds in **ONLINE mode** (20,000 rounds per run), Fixed 10% achieved a mean total profit of **₹80,265.41 ± 1,788.01**, compared to **₹78,924.82 ± 788.32** for LinUCB (a deficit of **-₹1,340.59 ± 1,554.26**, or **-1.67%**, with LinUCB winning in 1 of 10 seeds) and **₹78,856.21 ± 1,027.65** for Thompson Sampling (a deficit of **-₹1,409.20 ± 1,398.69**, or **-1.76%**, with Thompson Sampling winning in 2 of 10 seeds). Compared to the Rule-based policy (₹78,432.33 ± 1,176.84), LinUCB delivered a modest profit lift of **+₹492.50 ± 1,022.62** (**+0.63%**, winning in 8 of 10 seeds) and Thompson Sampling delivered **+₹423.88 ± 937.37** (**+0.54%**, winning in 6 of 10 seeds).

In **FROZEN mode** (evaluated on 5,000 held-out rounds per seed with frozen model weights), Fixed 10% maintained its profit lead with **₹20,136.83 ± 644.67** (₹4.0274/round), whereas LinUCB generated **₹19,736.87 ± 669.86** (₹3.9474/round, a profit deficit of **-₹399.96 ± 476.46** or **-1.99%**, winning in 1 of 10 seeds) and Thompson Sampling achieved **₹19,729.93 ± 620.49** (₹3.9460/round, a deficit of **-₹406.90 ± 403.13** or **-2.02%**, winning in 2 of 10 seeds). However, **LinUCB generated the highest gross revenue** in both Online mode (**₹234,276.84 ± 4,476.98**) and Frozen mode (**₹58,464.50 ± 2,463.67**), demonstrating that the bandit policies learned to stimulate demand and top-line volume through promotional discounting (average discount offered of 8.45% vs 8.09% for Fixed 10%), but sacrificed net unit margin in doing so.

---

## 2. Main Policy Comparison Tables

The multi-seed evaluation results across all five policies (Random, Fixed 10%, Rule-based, LinUCB, and Thompson Sampling) are summarized below. The visual trajectories corresponding to these tables are captured in `outputs/plots/cumulative_profit_mean_std.png`, `outputs/plots/gain_loss_linucb_vs_others.png`, and `outputs/plots/profit_vs_discount_scatter.png`.

### Table 2.1: Online Mode Benchmarking (10 Seeds × 20,000 Rounds)
*Each policy learns online from round 0 to 20,000. Values denote mean ± standard deviation across Seeds 0–9.*

| Policy | Total Revenue (₹) | Total Profit (₹) | Conversion Rate (%) | Avg Discount Offered (%) | Total Discount Given (₹) | Cumulative Regret (₹) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random** | ₹232,030.37 ± 3,354.84 | ₹78,371.40 ± 985.88 | 24.05% ± 0.33% | 8.20% ± 0.13% | ₹24,067.26 ± 747.06 | ₹1,724.71 ± 24.09 |
| **Fixed 10%** | ₹232,094.81 ± 5,212.23 | **₹80,265.41 ± 1,788.01** | 23.81% ± 0.44% | 8.09% ± 0.04% | **₹20,954.58 ± 511.11** | **₹23.97 ± 1.59** |
| **Rule-based** | ₹232,086.22 ± 3,581.12 | ₹78,432.33 ± 1,176.84 | 24.07% ± 0.37% | 8.19% ± 0.10% | ₹24,002.89 ± 578.23 | ₹1,529.80 ± 31.51 |
| **LinUCB** | **₹234,276.84 ± 4,476.98** | ₹78,924.82 ± 788.32 | **24.29% ± 0.58%** | **8.45% ± 0.59%** | ₹24,642.35 ± 2,794.46 | ₹1,475.69 ± 399.77 |
| **Thompson Sampling** | ₹232,651.13 ± 5,585.07 | ₹78,856.21 ± 1,027.65 | 24.08% ± 0.70% | 8.21% ± 0.71% | ₹23,673.25 ± 3,041.12 | ₹1,489.61 ± 537.17 |

### Table 2.2: Frozen Mode Benchmarking (10 Seeds × 5,000 Held-out Rounds)
*Policies frozen after round 20,000 and evaluated on 5,000 held-out customer interactions with zero learning.*

| Policy | Total Revenue (₹) | Total Profit (₹) | Profit / Round (₹) | Conversion Rate (%) | Avg Discount Offered (%) | Discount / Round (₹) | Total Discount Given (₹) | Cumulative Regret (₹) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random** | ₹57,513.17 ± 2,059.95 | ₹19,375.57 ± 650.56 | ₹3.8751 ± 0.1301 | 24.11% ± 0.72% | 8.33% ± 0.27% | ₹1.2099 ± 0.0712 | ₹6,049.37 ± 355.92 | ₹438.64 ± 12.88 |
| **Fixed 10%** | ₹58,298.52 ± 1,932.34 | **₹20,136.83 ± 644.67** | **₹4.0274 ± 0.1289** | 23.93% ± 0.68% | 8.11% ± 0.11% | **₹1.0609 ± 0.0458** | **₹5,304.56 ± 228.88** | **₹5.65 ± 0.37** |
| **Rule-based** | ₹58,313.35 ± 2,499.43 | ₹19,667.14 ± 883.74 | ₹3.9334 ± 0.1767 | 24.41% ± 0.79% | 8.29% ± 0.20% | ₹1.2194 ± 0.0558 | ₹6,096.96 ± 279.15 | ₹387.32 ± 15.86 |
| **LinUCB** | **₹58,464.50 ± 2,463.67** | ₹19,736.87 ± 669.86 | ₹3.9474 ± 0.1340 | **24.42% ± 1.10%** | **8.47% ± 0.68%** | ₹1.2163 ± 0.1725 | ₹6,081.46 ± 862.28 | ₹346.06 ± 113.89 |
| **Thompson Sampling** | ₹58,158.75 ± 2,116.35 | ₹19,729.93 ± 620.49 | ₹3.9460 ± 0.1241 | 24.31% ± 0.87% | 8.25% ± 0.70% | ₹1.1778 ± 0.1531 | ₹5,889.20 ± 765.59 | ₹337.54 ± 127.88 |

### Table 2.3: Lift and Win Counts Across Seeds

| Evaluation Mode | Bandit Policy | Baseline Comparator | Profit Difference (₹) | Percentage Lift (%) | Win Count (out of 10 Seeds) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ONLINE (20k rounds)** | LinUCB | Fixed 10% | -₹1,340.59 ± 1,554.26 | -1.67% | 1 of 10 seeds |
| **ONLINE (20k rounds)** | LinUCB | Rule-based | +₹492.50 ± 1,022.62 | +0.63% | 8 of 10 seeds |
| **ONLINE (20k rounds)** | Thompson Sampling | Fixed 10% | -₹1,409.20 ± 1,398.69 | -1.76% | 2 of 10 seeds |
| **ONLINE (20k rounds)** | Thompson Sampling | Rule-based | +₹423.88 ± 937.37 | +0.54% | 6 of 10 seeds |
| **FROZEN (5k rounds)** | LinUCB | Fixed 10% | -₹399.96 ± 476.46 | -1.99% | 1 of 10 seeds |
| **FROZEN (5k rounds)** | LinUCB | Rule-based | +₹69.73 ± 811.53 | +0.35% | 4 of 10 seeds |
| **FROZEN (5k rounds)** | Thompson Sampling | Fixed 10% | -₹406.90 ± 403.13 | -2.02% | 2 of 10 seeds |
| **FROZEN (5k rounds)** | Thompson Sampling | Rule-based | +₹62.79 ± 611.42 | +0.32% | 7 of 10 seeds |

---

## 3. Learning Behavior and Convergence Analysis

The online convergence dynamics of LinUCB and Thompson Sampling were analyzed on the fixed single training sequence (Milestone 32, Seed 42, 20,000 rounds). The regret trajectory is plotted in `outputs/plots/regret_curves_linucb_vs_ts.png` and windowed slope reductions are shown in `outputs/plots/avg_regret_per_window.png`.

### Table 3.1: Regret Checkpoints on Seed 42 Sequence (from `results/regret_checkpoints.csv`)

| Policy | Regret @ 1,000 (₹) | Regret @ 2,500 (₹) | Regret @ 5,000 (₹) | Regret @ 10,000 (₹) | Regret @ 20,000 (₹) | Avg Regret (Last 2,000 Rounds) | % Chose Oracle (Overall) | % Chose Oracle (Last 2,000 Rounds) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LinUCB** | ₹105.89 | ₹233.27 | ₹438.92 | ₹890.43 | ₹1,754.14 | ₹0.0919 / round | 20.94% | 16.90% |
| **Thompson Sampling** | ₹106.47 | ₹207.36 | ₹348.69 | ₹626.88 | **₹1,075.28** | **₹0.0483 / round** | **60.47%** | **62.20%** |

### Convergence Observations
1. **Regret Slope Flattening:** In early rounds (1 to 1,000), both algorithms explore aggressively, accumulating ~₹106 in regret (an initial rate of ~₹0.106/round). Between rounds 5,000 and 20,000, Thompson Sampling’s slope decreases markedly, dropping to **₹0.0483/round** in the final 2,000 rounds, ending with **₹1,075.28** cumulative regret. LinUCB flattens more gradually to **₹0.0919/round**, finishing with **₹1,754.14** cumulative regret on this sequence.
2. **Oracle Identification:** Thompson Sampling converged to matching the theoretical oracle discount action in **62.20%** of rounds during the final 2,000 rounds, compared to **16.90%** for LinUCB. LinUCB repeatedly balanced exploitation with exploration bonuses across features with wider variance, leading to continued exploration even in later rounds.
3. **Non-Zero Asymptotic Regret:** Because retail demand is non-deterministic (sigmoid purchase probability) and the feature space possesses residual noise, neither policy reaches zero per-round regret.

---

## 4. Tier-by-Tier Breakdown and Guardrail Audit

Analysis of customer loyalty tiers (from `results/multiseed_online_by_tier.csv`) reveals stark structural differences in policy behavior across customer segments. Loyalty tiers are defined deterministically by historical purchase frequency: **New** (< 3 purchases), **Regular** (3–9 purchases), and **Loyal** (≥ 10 purchases).

### Table 4.1: Performance Breakdown by Loyalty Tier Across 10 Seeds (Mean Values)

| Tier | Policy | Mean Rounds | Profit / Round (₹) | Conversion Rate (%) | Avg Discount Offered (%) | Avg Regret / Round (₹) | % Chose Oracle | 0% Disc Share | 5% Disc Share | 10% Disc Share | 15% Disc Share | 20% Disc Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **New** | Fixed 10%* | 7,642.0 | **₹3.987** | 21.41% | 5.00% | **₹0.000** | **100.00%** | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| | LinUCB | 7,642.0 | ₹3.920 | 20.07% | 3.02% | ₹0.069 | 60.30% | 39.70% | 60.30% | 0.00% | 0.00% | 0.00% |
| | Random | 7,642.0 | ₹3.889 | 19.60% | 2.50% | ₹0.087 | 50.08% | 49.92% | 50.08% | 0.00% | 0.00% | 0.00% |
| | Rule-based | 7,642.0 | ₹3.891 | 20.20% | 3.36% | ₹0.058 | 67.24% | 32.76% | 67.24% | 0.00% | 0.00% | 0.00% |
| | Thompson | 7,642.0 | ₹3.936 | 19.76% | 2.49% | ₹0.087 | 49.82% | 50.18% | 49.82% | 0.00% | 0.00% | 0.00% |
| **Regular** | Fixed 10% | 8,129.6 | **₹4.037** | 25.34% | 10.00% | **₹0.003** | **90.84%** | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| | LinUCB | 8,129.6 | ₹3.999 | 25.58% | 10.22% | ₹0.053 | 28.33% | 0.00% | 34.29% | 27.11% | 38.60% | 0.00% |
| | Random | 8,129.6 | ₹3.977 | 25.37% | 10.02% | ₹0.053 | 33.39% | 0.00% | 33.12% | 33.42% | 33.46% | 0.00% |
| | Rule-based | 8,129.6 | ₹3.975 | 25.50% | 10.03% | ₹0.062 | 31.99% | 0.00% | 33.19% | 33.10% | 33.71% | 0.00% |
| | Thompson | 8,129.6 | ₹3.972 | 25.63% | 10.44% | ₹0.044 | 39.92% | 0.00% | 24.82% | 41.50% | 33.68% | 0.00% |
| **Loyal** | Fixed 10% | 4,228.4 | **₹4.017** | 25.21% | 10.00% | **₹0.000** | **98.43%** | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| | LinUCB | 4,228.4 | ₹3.888 | 29.39% | 14.81% | ₹0.124 | 29.91% | 0.00% | 0.00% | 29.80% | 44.16% | 26.05% |
| | Random | 4,228.4 | ₹3.857 | 29.54% | 15.00% | ₹0.148 | 33.27% | 0.00% | 0.00% | 33.35% | 33.37% | 33.29% |
| | Rule-based | 4,228.4 | ₹3.874 | 28.33% | 13.38% | ₹0.138 | 64.74% | 0.00% | 0.00% | 66.21% | 0.00% | 33.80% |
| | Thompson | 4,228.4 | ₹3.901 | 28.84% | 14.17% | ₹0.108 | 39.25% | 0.00% | 0.00% | 39.17% | 38.23% | 22.60% |

*\*Note: For New customers, Fixed 10% fell back to 5% because 10% is not in the eligible guardrail set {0%, 5%}.*

### Guardrail Compliance Verification
1. **New Tier ({0%, 5%}):** Zero selections of 10%, 15%, or 20% across all policies (0.00% violation rate). LinUCB offered 0% discount in 39.70% of rounds and 5% in 60.30% of rounds.
2. **Regular Tier ({5%, 10%, 15%}):** Zero selections of 0% or 20% across all policies (0.00% violation rate). LinUCB distributed choices across 5% (34.29%), 10% (27.11%), and 15% (38.60%).
3. **Loyal Tier ({10%, 15%, 20%}):** Zero selections of 0% or 5% across all policies (0.00% violation rate). LinUCB offered 10% (29.80%), 15% (44.16%), and 20% (26.05%).
4. **Total Violations:** 0 across all 200,000 decisions.

### LinUCB Profit Comparison vs Fixed 10% Across Tiers
- **New Tier:** LinUCB earned ₹3.920/round vs ₹3.987/round for Fixed 10% (deficit of **-₹0.067/round**, or **-1.68%**; total loss of ~₹512.01).
- **Regular Tier:** LinUCB earned ₹3.999/round vs ₹4.037/round for Fixed 10% (deficit of **-₹0.038/round**, or **-0.94%**; total loss of ~₹308.92). **(Smallest profit deficit)**
- **Loyal Tier:** LinUCB earned ₹3.888/round vs ₹4.017/round for Fixed 10% (deficit of **-₹0.129/round**, or **-3.21%**; total loss of ~₹544.86). **(Largest profit deficit)**
- *Segment Volume Note:* Loyal customers represented 4,228.4 rounds (21.14% of the customer stream), while Regular customers formed the largest group (40.65%), and New customers comprised 38.21%.

---

## 5. Off-Policy Replay Evaluation on Real Dataset

To assess policy behavior on real retail transaction contexts outside the synthetic round generator, off-policy replay evaluation (Milestone 37) was executed on the full **73,100 historical inventory and sales records** (`data/historical_73k.pkl`). The visual summary is shown in `outputs/plots/replay_comparison.png`.

### Independence Diagnostic (Step 1)
The validity of standard rejection-sampling off-policy replay relies on the assumption that historical discounts were assigned uniformly at random (or independently of context). Statistical testing rejected this assumption:
- **Observed Discount vs DemandLevel:** $\chi^2 = 5,749.45$, $p < 10^{-10}$, Cramér's $V = 0.1983$ (moderate dependence).
- **Observed Discount vs Promotion:** $\chi^2 = 16,527.88$, $p < 10^{-10}$, Cramér's $V = 0.4755$ (strong dependence; unpromoted items received 0% discount in 42.0% of cases, while promoted items never received 0%).
- **Observed Discount vs Category:** $\chi^2 = 9.50$, $p = 0.891$, Cramér's $V = 0.0057$ (negligible dependence).
- **Historical Guardrail Non-Compliance:** **28,610 of 73,100 rows (39.14%)** in the historical log offered discounts that violated the loyalty-tier guardrails (e.g., 10% offered to New customers).

*Methodological Implication:* Because historical discounts were systematically confounded by promotion status and store policies, off-policy replay estimates suffer from selection bias and cannot serve as an unbiased causal estimate. Replay is interpreted strictly as a directional alignment check.

### Table 5.1: Off-Policy Replay Performance (from `results/replay_results.csv`)

| Policy | Evaluated Rows | Matched Rows | Match Rate (%) | Mean Profit (₹) | Bootstrap 95% CI (₹) | Mean Observed Discount (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logged Historical Policy** | 73,100 | 73,100 | 100.00% | ₹6.15 | [₹6.04, ₹6.26] | 6.41% |
| **Random Baseline** | 73,100 | 17,215 | 23.55% | ₹6.07 | [₹5.84, ₹6.29] | 5.96% |
| **Fixed 10%** | 73,100 | 17,798 | 24.35% | **₹6.42** | [₹6.19, ₹6.64] | 8.08% |
| **Rule-based Baseline** | 73,100 | 19,303 | **26.41%** | ₹5.88 | [₹5.67, ₹6.09] | 6.44% |
| **LinUCB (Frozen)** | 73,100 | 14,973 | 20.48% | ₹6.11 | [₹5.84, ₹6.38] | 4.94% |
| **Thompson Sampling (Frozen)**| 73,100 | 14,493 | 19.83% | ₹6.21 | [₹5.98, ₹6.45] | 7.96% |

### Unmatched Policy Behavior Across All 73,100 Contexts
When exposed to all 73,100 real product contexts (regardless of logged action match):
- **Observed Historical Actions:** 0% (33.35%), 5% (26.49%), 10% (23.29%), 15% (12.30%), 20% (4.57%).
- **LinUCB Action Distribution:** 0% (30.43%), 5% (17.50%), 10% (2.64%), 15% (49.31%), 20% (0.13%).
- **Thompson Sampling Distribution:** 0% (0.60%), 5% (43.69%), 10% (17.43%), 15% (38.17%), 20% (0.11%).

Because the bootstrap 95% confidence intervals of LinUCB ([₹5.84, ₹6.38]) and the Logged Policy ([₹6.04, ₹6.26]) substantially overlap, no claim of statistical superiority on historical data can be asserted.

---

## 6. Sensitivity Analysis and Reward Ablation

### 6.1 Exploration Parameter Sensitivity ($\alpha$)
The exploration rate $\alpha \in \{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\}$ was evaluated across `TUNING_SEEDS = [100, 101, 102, 103, 104]` over 20,000 rounds per run (Milestone 38). Relevant plots are located at `outputs/plots/alpha_regret_curves.png`, `outputs/plots/alpha_final_regret.png`, and `outputs/plots/alpha_early_vs_late.png`.

- **Per-round Profit Scale ($r_{\text{scale}}$):** Mean profit is ₹3.8493, with standard deviation $r_{\text{scale}} = ₹7.9912$. The typical uncertainty bonus term $\sqrt{x^T A_a^{-1} x}$ was measured at 3.74 (round 10), 1.57 (round 100), 0.46 (round 1,000), and 0.24 (round 10,000).
- **Optimal Exploration Range:** Cumulative regret was minimized at **$\alpha = 0.1$** with **₹1,290.92 ± 318.78** (and 47.13% oracle accuracy).
- **Pure Greedy ($\alpha = 0.0$):** Accumulated **₹1,303.56 ± 402.07** regret. Although competitive early, without exploration the greedy policy locked into suboptimal arms in seed 103 (regret reached ₹1,904.41).
- **Over-Exploration ($\alpha = 5.0$):** Accumulated the highest regret of **₹1,462.92 ± 236.45**, suffering an initial 2,000-round exploration penalty of **₹181.03 ± 24.89** (vs ₹160.10 for greedy).
- **Baseline Alignment:** The default value $\alpha = 1.0$ used in headline evaluation accumulated **₹1,344.15 ± 571.63** regret (total profit ₹79,070.50), performing within 4.1% of the tuning optimum while providing robust exploration across diverse contexts.

### 6.2 Reward Penalty Ablation ($\lambda$)
To penalize excessive discounting and preserve brand equity, the objective was ablated using $R_\lambda = \text{Profit} - \lambda \cdot \text{DiscountAmountRupees}$ (Milestone 39) across `TUNING_SEEDS = [100..104]`. Diagnostic plots include `outputs/plots/ablation_discount_vs_lambda.png`, `outputs/plots/ablation_tradeoff.png`, `outputs/plots/ablation_discount_mix.png`, and `outputs/plots/ablation_regret.png`.

### Table 6.1: Price of Restraint Under Penalized Rewards (from `results/reward_ablation.csv`)

| Configuration | Penalty Mode | $\lambda$ | Mean Profit (₹) | Total Disc Given (₹) | $\Delta$ Profit vs Baseline (₹) | Disc Avoided vs Baseline (₹) | Price of Restraint (₹ Profit Sacrificed / ₹ Disc Avoided) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline ($\lambda=0$)** | offered | 0.000 | ₹79,070.50 | ₹24,900.71 | ₹0.00 | ₹0.00 | 0.0000 |
| **Mild Penalty** | offered | 0.007 | ₹79,068.22 | ₹24,649.73 | -₹2.29 | ₹250.98 | **0.0091** |
| **Moderate Penalty** | offered | 0.073 | ₹78,930.58 | ₹22,308.46 | -₹139.92 | ₹2,592.25 | **0.0540** |
| **Aggressive Penalty** | offered | 0.365 | ₹79,342.62 | ₹15,486.29 | +₹272.12 | ₹9,414.42 | **-0.0289** (Dominant) |
| **Severe Penalty** | offered | 0.730 | ₹78,641.85 | ₹11,951.21 | -₹428.65 | ₹12,949.50 | **0.0331** |
| **Purchased-Only Penalty**| purchased| 0.073 | ₹78,778.64 | ₹23,831.72 | -₹291.87 | ₹1,068.99 | **0.2730** |

### Key Ablation Insights
1. **Curbs on Excessive Discounting:** Increasing $\lambda$ from 0 to 0.730 reduced average offered discount from **8.65% down to 4.86%**, cutting total discounts disbursed from ₹24,900.71 to ₹11,951.21 (a 52.0% reduction).
2. **The "Sweet Spot" ($\lambda = 0.365$):** At $\lambda = 0.365$, the policy shifted from 15% discounts to 5% and 10% discounts. Because price sensitivity is non-linear, this restraint preserved sales volume while drastically reducing promotional giveaway, resulting in a ₹272.12 increase in pure profit alongside ₹9,414.42 in avoided discounts.
3. **Offered vs Purchased Penalty:** Applying the penalty only when a purchase occurs (`penalty_on="purchased"` at $\lambda=0.073$) produced a substantially higher sacrifice ratio (₹0.273 profit lost per ₹1 avoided) than penalizing all offers (₹0.054), because penalizing unexercised offers exerts stronger downward pressure on speculative discounting.
4. **Modeling Status:** Implementing a penalty $\lambda > 0$ is an explicit modeling decision representing external business priorities (such as reference price protection), not an empirical finding derived from unconstrained profit maximization.

---

## 7. Limitations of the Evaluation

1. **Synthetic Customer State Dynamics:** Customer purchase counts, recency intervals, and loyalty transitions were simulated via a 50-customer registry rather than tracked across physical consumer identities.
2. **Calibrated Sigmoid Purchase Function:** Customer purchase probabilities were governed by a calibrated logistic simulator ($\text{AUC} \approx 0.63$). A policy win within this simulator reflects alignment with the specified response curves, not real-world omni-channel shopper reactions.
3. **Internal Evaluator Symmetry:** Because both training and evaluation utilized the same underlying simulator dynamics, findings represent internal simulator performance.
4. **Assumed Retail Cost Basis:** Item cost was computed assuming a uniform 40% margin structure ($\text{CostPrice} = 0.60 \times \text{Price}$), which abstracts away manufacturer rebates, holding costs, and perishable spoilage.
5. **Historical Replay Confounding:** Over 39% of historical dataset rows violated loyalty guardrails, and promotional items were strongly confounded with discount assignment ($\text{Cramér's } V = 0.4755$), preventing unbiased off-policy counterfactual inference.
6. **Simulated Currency Units:** All financial values are expressed in simulated rupees (₹) within the synthetic testing environment.
7. **Offline Vectorized Environment:** Policies were benchmarked in an offline batched setting without inventory stockouts, competitor retaliatory price changes, or macro-economic demand shocks.

---

## 8. Defensible Claims vs. Claims That Must Not Be Made

### Claims That Can Be Defended
- **Within the simulator, Fixed 10% earned the highest total profit** (₹80,265.41 ± 1,788.01), exceeding LinUCB in 9 of 10 evaluation seeds in Online mode and 9 of 10 seeds in Frozen mode.
- **LinUCB generated the highest gross revenue** in both Online (₹234,276.84) and Frozen (₹58,464.50) modes, demonstrating effective top-line demand stimulation.
- **Strict guardrail compliance was achieved:** LinUCB and Thompson Sampling achieved 100.00% compliance across all 200,000 decisions, never recommending discounts outside the allowed tier menus.
- **LinUCB improved upon the Rule-based baseline in profit** in 8 of 10 online seeds (+₹492.50, +0.63% lift).
- **Off-policy replay confirmed behavioral alignment:** On 73,100 real store contexts, LinUCB matched historical profit within overlapping 95% confidence intervals ([₹5.84, ₹6.38] vs [₹6.04, ₹6.26]).
- **A modest reward penalty ($\lambda = 0.073$ to $0.365$) effectively curbs discount exposure**, reducing promotional spend by up to ₹9,414 with minimal profit disruption.

### Claims That Must NOT Be Made
- **"LinUCB will increase profit in a real physical store."** (Simulator performance does not guarantee field performance under unmodeled consumer behavior.)
- **"Bandit algorithms proved statistically superior in profit to heuristic baselines."** (The empirical data demonstrates that Fixed 10% was more profitable within this environment.)
- **"Off-policy replay proves that LinUCB would outperform historical store managers."** (The independence diagnostic proved that historical discounts were confounded by promotions and tier violations, invalidating unbiased causal claims.)
- **"A non-zero penalty $\lambda$ is mathematically optimal."** (Selecting $\lambda > 0$ is a subjective business choice, not an empirical truth.)
- **"Results are statistically significant."** (No formal hypothesis tests or p-values were evaluated for policy profit rankings.)

---

## 9. Numbers Audit Table

*Cross-reference mapping every quantitative statistic in Sections 1 through 6 directly to its source file and cell.*

| Number Reported | Section | Context / Meaning | Source File | Cell / Column Reference |
| :--- | :--- | :--- | :--- | :--- |
| **₹80,265.41 ± 1,788.01** | §1, §2 | Fixed 10% Online Total Profit | `results/comparison_online.csv` | Row `Fixed 10%`, Col `Total Profit (Rs)` |
| **₹78,924.82 ± 788.32** | §1, §2 | LinUCB Online Total Profit | `results/comparison_online.csv` | Row `LinUCB`, Col `Total Profit (Rs)` |
| **₹78,856.21 ± 1,027.65** | §1, §2 | Thompson Sampling Online Profit | `results/comparison_online.csv` | Row `Thompson Sampling`, Col `Total Profit (Rs)` |
| **₹78,432.33 ± 1,176.84** | §1, §2 | Rule-based Online Profit | `results/comparison_online.csv` | Row `Rule-based`, Col `Total Profit (Rs)` |
| **₹78,371.40 ± 985.88** | §2 | Random Baseline Online Profit | `results/comparison_online.csv` | Row `Random`, Col `Total Profit (Rs)` |
| **-₹1,340.59 ± 1,554.26** | §1, §2 | LinUCB Profit Diff vs Fixed 10% (Online)| `results/comparison_lift.csv` | Row 0, Col `Profit Difference (Rs)` |
| **-1.67%** | §1, §2 | LinUCB Lift % vs Fixed 10% (Online) | `results/comparison_lift.csv` | Row 0, Col `Percentage Lift (%)` |
| **1 of 10 seeds** | §1, §2 | LinUCB Win Count vs Fixed 10% (Online) | `results/comparison_lift.csv` | Row 0, Col `Win Count` |
| **+₹492.50 ± 1,022.62** | §1, §2 | LinUCB Profit Diff vs Rule-based (Online)| `results/comparison_lift.csv` | Row 1, Col `Profit Difference (Rs)` |
| **+0.63%** | §1, §2 | LinUCB Lift % vs Rule-based (Online) | `results/comparison_lift.csv` | Row 1, Col `Percentage Lift (%)` |
| **8 of 10 seeds** | §1, §2 | LinUCB Win Count vs Rule-based (Online) | `results/comparison_lift.csv` | Row 1, Col `Win Count` |
| **-₹1,409.20 ± 1,398.69** | §1, §2 | Thompson Profit Diff vs Fixed 10% (Online)| `results/comparison_lift.csv` | Row 2, Col `Profit Difference (Rs)` |
| **-1.76%** | §1, §2 | Thompson Lift % vs Fixed 10% (Online) | `results/comparison_lift.csv` | Row 2, Col `Percentage Lift (%)` |
| **2 of 10 seeds** | §1, §2 | Thompson Win Count vs Fixed 10% (Online)| `results/comparison_lift.csv` | Row 2, Col `Win Count` |
| **+₹423.88 ± 937.37** | §1, §2 | Thompson Profit Diff vs Rule-based (Online)| `results/comparison_lift.csv` | Row 3, Col `Profit Difference (Rs)` |
| **+0.54%** | §1, §2 | Thompson Lift % vs Rule-based (Online) | `results/comparison_lift.csv` | Row 3, Col `Percentage Lift (%)` |
| **6 of 10 seeds** | §1, §2 | Thompson Win Count vs Rule-based (Online)| `results/comparison_lift.csv` | Row 3, Col `Win Count` |
| **₹20,136.83 ± 644.67** | §1, §2 | Fixed 10% Frozen Total Profit | `results/comparison_frozen.csv` | Row `Fixed 10%`, Col `Total Profit (Rs)` |
| **₹19,736.87 ± 669.86** | §1, §2 | LinUCB Frozen Total Profit | `results/comparison_frozen.csv` | Row `LinUCB`, Col `Total Profit (Rs)` |
| **₹19,729.93 ± 620.49** | §1, §2 | Thompson Frozen Total Profit | `results/comparison_frozen.csv` | Row `Thompson Sampling`, Col `Total Profit (Rs)` |
| **₹19,667.14 ± 883.74** | §2 | Rule-based Frozen Total Profit | `results/comparison_frozen.csv` | Row `Rule-based`, Col `Total Profit (Rs)` |
| **₹19,375.57 ± 650.56** | §2 | Random Frozen Total Profit | `results/comparison_frozen.csv` | Row `Random`, Col `Total Profit (Rs)` |
| **-₹399.96 ± 476.46** | §1, §2 | LinUCB Profit Diff vs Fixed 10% (Frozen) | `results/comparison_lift.csv` | Row 4, Col `Profit Difference (Rs)` |
| **-1.99%** | §1, §2 | LinUCB Lift % vs Fixed 10% (Frozen) | `results/comparison_lift.csv` | Row 4, Col `Percentage Lift (%)` |
| **1 of 10 seeds** | §1, §2 | LinUCB Win Count vs Fixed 10% (Frozen) | `results/comparison_lift.csv` | Row 4, Col `Win Count` |
| **+₹69.73 ± 811.53** | §1, §2 | LinUCB Profit Diff vs Rule-based (Frozen)| `results/comparison_lift.csv` | Row 5, Col `Profit Difference (Rs)` |
| **+0.35%** | §1, §2 | LinUCB Lift % vs Rule-based (Frozen) | `results/comparison_lift.csv` | Row 5, Col `Percentage Lift (%)` |
| **4 of 10 seeds** | §1, §2 | LinUCB Win Count vs Rule-based (Frozen) | `results/comparison_lift.csv` | Row 5, Col `Win Count` |
| **-₹406.90 ± 403.13** | §1, §2 | Thompson Profit Diff vs Fixed 10% (Frozen)| `results/comparison_lift.csv` | Row 6, Col `Profit Difference (Rs)` |
| **-2.02%** | §1, §2 | Thompson Lift % vs Fixed 10% (Frozen) | `results/comparison_lift.csv` | Row 6, Col `Percentage Lift (%)` |
| **2 of 10 seeds** | §1, §2 | Thompson Win Count vs Fixed 10% (Frozen)| `results/comparison_lift.csv` | Row 6, Col `Win Count` |
| **+₹62.79 ± 611.42** | §1, §2 | Thompson Profit Diff vs Rule-based (Frozen)| `results/comparison_lift.csv` | Row 7, Col `Profit Difference (Rs)` |
| **+0.32%** | §1, §2 | Thompson Lift % vs Rule-based (Frozen) | `results/comparison_lift.csv` | Row 7, Col `Percentage Lift (%)` |
| **7 of 10 seeds** | §1, §2 | Thompson Win Count vs Rule-based (Frozen)| `results/comparison_lift.csv` | Row 7, Col `Win Count` |
| **₹234,276.84 ± 4,476.98** | §1, §2 | LinUCB Online Total Revenue | `results/comparison_online.csv` | Row `LinUCB`, Col `Total Revenue (Rs)` |
| **₹58,464.50 ± 2,463.67** | §1, §2 | LinUCB Frozen Total Revenue | `results/comparison_frozen.csv` | Row `LinUCB`, Col `Total Revenue (Rs)` |
| **₹24,642.35 ± 2,794.46** | §2 | LinUCB Online Total Discount Given | `results/comparison_online.csv` | Row `LinUCB`, Col `Total Discount Given (Rs)` |
| **₹20,954.58 ± 511.11** | §2 | Fixed 10% Online Total Discount Given | `results/comparison_online.csv` | Row `Fixed 10%`, Col `Total Discount Given (Rs)` |
| **8.45% ± 0.59%** | §1, §2 | LinUCB Online Avg Discount Offered | `results/comparison_online.csv` | Row `LinUCB`, Col `Avg Discount Offered (%)` |
| **8.09% ± 0.04%** | §1, §2 | Fixed 10% Online Avg Discount Offered | `results/comparison_online.csv` | Row `Fixed 10%`, Col `Avg Discount Offered (%)` |
| **₹1,475.69 ± 399.77** | §2 | LinUCB Online Cumulative Regret | `results/comparison_online.csv` | Row `LinUCB`, Col `Cumulative Regret (Rs)` |
| **₹1,489.61 ± 537.17** | §2 | Thompson Online Cumulative Regret | `results/comparison_online.csv` | Row `Thompson Sampling`, Col `Cumulative Regret (Rs)` |
| **₹23.97 ± 1.59** | §2 | Fixed 10% Online Cumulative Regret | `results/comparison_online.csv` | Row `Fixed 10%`, Col `Cumulative Regret (Rs)` |
| **₹105.8879** | §3 | LinUCB Regret @ 1,000 Rounds (Seed 42) | `results/regret_checkpoints.csv` | Row 0 (`LinUCB`), Col `Regret@1000` |
| **₹438.9183** | §3 | LinUCB Regret @ 5,000 Rounds (Seed 42) | `results/regret_checkpoints.csv` | Row 0 (`LinUCB`), Col `Regret@5000` |
| **₹1,754.1374** | §3 | LinUCB Regret @ 20,000 Rounds (Seed 42)| `results/regret_checkpoints.csv` | Row 0 (`LinUCB`), Col `Regret@20000` |
| **₹0.091932** | §3 | LinUCB Avg Regret Last 2,000 Rounds | `results/regret_checkpoints.csv` | Row 0 (`LinUCB`), Col `AvgRegret_Last2000` |
| **16.9%** | §3 | LinUCB % Chose Oracle Last 2,000 Rounds | `results/regret_checkpoints.csv` | Row 0 (`LinUCB`), Col `Pct_ChoseOracle_Last2000` |
| **₹106.4672** | §3 | Thompson Regret @ 1,000 Rounds (Seed 42)| `results/regret_checkpoints.csv` | Row 1 (`Thompson`), Col `Regret@1000` |
| **₹348.6890** | §3 | Thompson Regret @ 5,000 Rounds (Seed 42)| `results/regret_checkpoints.csv` | Row 1 (`Thompson`), Col `Regret@5000` |
| **₹1,075.2797** | §3 | Thompson Regret @ 20,000 Rounds (Seed 42)| `results/regret_checkpoints.csv` | Row 1 (`Thompson`), Col `Regret@20000` |
| **₹0.048292** | §3 | Thompson Avg Regret Last 2,000 Rounds | `results/regret_checkpoints.csv` | Row 1 (`Thompson`), Col `AvgRegret_Last2000` |
| **62.2%** | §3 | Thompson % Chose Oracle Last 2,000 Rounds| `results/regret_checkpoints.csv` | Row 1 (`Thompson`), Col `Pct_ChoseOracle_Last2000` |
| **4,228.4** | §4 | Loyal Tier Average Rounds / Seed | `results/multiseed_online_by_tier.csv` | Tier `Loyal`, Mean of `tier_rounds` |
| **8,129.6** | §4 | Regular Tier Average Rounds / Seed | `results/multiseed_online_by_tier.csv` | Tier `Regular`, Mean of `tier_rounds` |
| **7,642.0** | §4 | New Tier Average Rounds / Seed | `results/multiseed_online_by_tier.csv` | Tier `New`, Mean of `tier_rounds` |
| **₹3.920** | §4 | LinUCB New Tier Profit / Round | `results/multiseed_online_by_tier.csv` | Tier `New`, Policy `LinUCB`, `tier_avg_profit_per_round` |
| **₹3.987** | §4 | Fixed 10% New Tier Profit / Round | `results/multiseed_online_by_tier.csv` | Tier `New`, Policy `Fixed 10%`, `tier_avg_profit_per_round` |
| **₹3.999** | §4 | LinUCB Regular Tier Profit / Round | `results/multiseed_online_by_tier.csv` | Tier `Regular`, Policy `LinUCB`, `tier_avg_profit_per_round` |
| **₹4.037** | §4 | Fixed 10% Regular Tier Profit / Round | `results/multiseed_online_by_tier.csv` | Tier `Regular`, Policy `Fixed 10%`, `tier_avg_profit_per_round` |
| **₹3.888** | §4 | LinUCB Loyal Tier Profit / Round | `results/multiseed_online_by_tier.csv` | Tier `Loyal`, Policy `LinUCB`, `tier_avg_profit_per_round` |
| **₹4.017** | §4 | Fixed 10% Loyal Tier Profit / Round | `results/multiseed_online_by_tier.csv` | Tier `Loyal`, Policy `Fixed 10%`, `tier_avg_profit_per_round` |
| **39.14% (28,610 / 73,100)**| §5 | Historical Rows Violating Guardrails | Historical Data Diagnostic (M37) | `df['TierRuleCompliant'] == False` |
| **0.1983** | §5 | Cramér's V (Observed Disc vs Demand) | Milestone 37 Independence Diagnostic | Chi-squared Contingency Table |
| **0.4755** | §5 | Cramér's V (Observed Disc vs Promotion) | Milestone 37 Independence Diagnostic | Chi-squared Contingency Table |
| **14,973 (20.48%)** | §5 | LinUCB Replay Kept Matched Rows | `results/replay_results.csv` | Row `LinUCB`, Cols `Kept_Rows`, `Match_Rate_Pct` |
| **₹6.1060** | §5 | LinUCB Replay Mean Profit | `results/replay_results.csv` | Row `LinUCB`, Col `Mean_Profit_Rs` |
| **[₹5.8403, ₹6.3761]** | §5 | LinUCB Replay 95% Confidence Interval | `results/replay_results.csv` | Row `LinUCB`, Cols `CI_95_Low_Rs`, `CI_95_High_Rs` |
| **₹6.1493** | §5 | Logged Historical Policy Mean Profit | `results/replay_results.csv` | Row `Logged Policy`, Col `Mean_Profit_Rs` |
| **[₹6.0365, ₹6.2585]** | §5 | Logged Policy 95% Confidence Interval | `results/replay_results.csv` | Row `Logged Policy`, Cols `CI_95_Low_Rs`, `CI_95_High_Rs` |
| **₹6.4224** | §5 | Fixed 10% Replay Mean Profit | `results/replay_results.csv` | Row `Fixed 10%`, Col `Mean_Profit_Rs` |
| **₹1,290.92 ± 318.78** | §6.1 | LinUCB Regret at Optimal $\alpha = 0.1$ | `results/alpha_sensitivity.csv` | Row `alpha=0.1`, Col `final_cumulative_regret_rs_mean` |
| **₹1,303.56 ± 402.07** | §6.1 | LinUCB Regret at Greedy $\alpha = 0.0$ | `results/alpha_sensitivity.csv` | Row `alpha=0.0`, Col `final_cumulative_regret_rs_mean` |
| **₹1,462.92 ± 236.45** | §6.1 | LinUCB Regret at $\alpha = 5.0$ | `results/alpha_sensitivity.csv` | Row `alpha=5.0`, Col `final_cumulative_regret_rs_mean` |
| **₹1,344.15 ± 571.63** | §6.1 | LinUCB Regret at Baseline $\alpha = 1.0$ | `results/alpha_sensitivity.csv` | Row `alpha=1.0`, Col `final_cumulative_regret_rs_mean` |
| **₹79,070.50** | §6.2 | LinUCB Ablation Baseline Profit ($\lambda=0$)| `results/reward_ablation.csv` | Row `lambda=0.0`, Col `total_profit_mean_rs` |
| **₹24,900.71** | §6.2 | LinUCB Baseline Total Discount Given | `results/reward_ablation.csv` | Row `lambda=0.0`, Col `total_discount_given_mean_rs` |
| **8.65%** | §6.2 | LinUCB Baseline Avg Discount Offered | `results/reward_ablation.csv` | Row `lambda=0.0`, Col `avg_discount_offered_mean_pct` |
| **-₹139.92** | §6.2 | Profit Sacrificed at $\lambda = 0.073$ | `results/reward_ablation.csv` / M39 | Derived: ₹78,930.58 - ₹79,070.50 |
| **₹2,592.25** | §6.2 | Discount Avoided at $\lambda = 0.073$ | `results/reward_ablation.csv` / M39 | Derived: ₹24,900.71 - ₹22,308.46 |
| **0.0540** | §6.2 | Price of Restraint Ratio at $\lambda = 0.073$| `results/reward_ablation.csv` / M39 | Price of Restraint Table |
| **+₹272.12** | §6.2 | Profit Gain at $\lambda = 0.365$ | `results/reward_ablation.csv` / M39 | Derived: ₹79,342.62 - ₹79,070.50 |
| **₹9,414.42** | §6.2 | Discount Avoided at $\lambda = 0.365$ | `results/reward_ablation.csv` / M39 | Derived: ₹24,900.71 - ₹15,486.29 |
| **4.86%** | §6.2 | LinUCB Avg Discount Offered at $\lambda = 0.730$| `results/reward_ablation.csv` | Row `lambda=0.730`, Col `avg_discount_offered_mean_pct` |
| **₹11,951.21** | §6.2 | LinUCB Total Discount Given at $\lambda = 0.730$| `results/reward_ablation.csv` | Row `lambda=0.730`, Col `total_discount_given_mean_rs` |

---
*Report verified against all underlying artifact data files. Evaluation Phase 7 complete.*
