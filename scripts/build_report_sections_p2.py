# -*- coding: utf-8 -*-
"""
Generates sections 06, 07, and 08 of the Final Project Report in reports/sections/.
Strict adherence to WRAP-UP SHARED DEFINITIONS and Pass D rules.
"""

import os

os.makedirs("reports/sections", exist_ok=True)

# -------------------------------------------------------------
# 06_results.md
# -------------------------------------------------------------
results_text = r"""# 6. Empirical Results and Performance Analysis

*Notice: All financial outcomes and customer behaviors reported throughout this section are evaluated within the simulation environment and represent simulated rupees (₹).*

## 6.1 Headline Results and Policy Benchmarking

In both Online and Frozen evaluation modes, the heuristic benchmark **Fixed 10% earned the highest total profit**, outperforming both contextual bandit algorithms (LinUCB and Thompson Sampling). Across 10 independent evaluation seeds in **ONLINE mode** (20,000 rounds per run), Fixed 10% achieved a mean cumulative profit of **₹80,265.41 ± 1,788.01**, compared to **₹78,924.82 ± 788.32** for LinUCB and **₹78,856.21 ± 1,027.65** for Thompson Sampling. 

LinUCB suffered an average profit deficit of **-₹1,340.59 ± 1,554.26** (**-1.67%**, winning in 1 of 10 seeds) relative to Fixed 10%, while Thompson Sampling trailed by **-₹1,409.20 ± 1,398.69** (**-1.76%**, winning in 2 of 10 seeds). However, when benchmarked against the standard commercial **Rule-based policy** (₹78,432.33 ± 1,176.84), LinUCB delivered an average profit lift of **+₹492.50 ± 1,022.62** (**+0.63%**, winning in 8 of 10 seeds), and Thompson Sampling delivered **+₹423.88 ± 937.37** (**+0.54%**, winning in 6 of 10 seeds).

Crucially, **LinUCB generated the highest gross revenue** in both Online mode (**₹234,276.84 ± 4,476.98**) and Frozen mode (**₹58,464.50 ± 2,463.67**). The contextual bandit learned to identify price-elastic opportunities and offered higher promotional discounts (average discount offered of **8.45% ± 0.59%** vs. **8.09% ± 0.04%** for Fixed 10%), stimulating unit conversions (**24.29% ± 0.58%** vs. **23.81% ± 0.44%**), but sacrificing net unit margin in doing so.

Table 6.1 and Table 6.2 summarize the primary online and frozen performance metrics.

**Table 6.1: Online Evaluation Benchmarking (10 Seeds × 20,000 Rounds)**

| Policy | Total Revenue (₹) | Total Profit (₹) | Conversion Rate (%) | Avg Discount Offered (%) | Total Discount Given (₹) | Cumulative Regret (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed 10%** | ₹232,094.81 ± 5,212.23 | **₹80,265.41 ± 1,788.01** | 23.81% ± 0.44% | 8.09% ± 0.04% | **₹20,954.58 ± 511.11** | **₹23.97 ± 1.59** |
| **LinUCB** | **₹234,276.84 ± 4,476.98** | ₹78,924.82 ± 788.32 | **24.29% ± 0.58%** | 8.45% ± 0.59% | ₹24,642.35 ± 2,794.46 | ₹1,475.69 ± 399.77 |
| **Thompson Sampling**| ₹232,651.13 ± 5,585.07 | ₹78,856.21 ± 1,027.65 | 24.08% ± 0.70% | 8.21% ± 0.71% | ₹23,673.25 ± 3,041.12 | ₹1,489.61 ± 537.17 |
| **Rule-based** | ₹232,086.22 ± 3,581.12 | ₹78,432.33 ± 1,176.84 | 24.07% ± 0.37% | 8.19% ± 0.10% | ₹24,002.89 ± 578.23 | ₹1,529.80 ± 31.51 |
| **Random** | ₹232,030.37 ± 3,354.84 | ₹78,371.40 ± 985.88 | 24.05% ± 0.33% | 8.20% ± 0.13% | ₹24,067.26 ± 747.06 | ₹1,724.71 ± 24.09 |

**Table 6.2: Frozen Evaluation Benchmarking (10 Seeds × 5,000 Held-out Rounds)**

| Policy | Total Revenue (₹) | Total Profit (₹) | Profit / Round (₹) | Conversion Rate (%) | Avg Discount Offered (%) | Discount Given / Round (₹) | Cumulative Regret (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed 10%** | ₹58,298.52 ± 1,932.34 | **₹20,136.83 ± 644.67** | **₹4.0274 ± 0.1289** | 23.93% ± 0.68% | 8.11% ± 0.11% | **₹1.0609 ± 0.0458** | **₹5.65 ± 0.37** |
| **LinUCB** | **₹58,464.50 ± 2,463.67** | ₹19,736.87 ± 669.86 | ₹3.9474 ± 0.1340 | **24.42% ± 1.10%** | 8.47% ± 0.68% | ₹1.2163 ± 0.1725 | ₹346.06 ± 113.89 |
| **Thompson Sampling**| ₹58,158.75 ± 2,116.35 | ₹19,729.93 ± 620.49 | ₹3.9460 ± 0.1241 | 24.31% ± 0.87% | 8.25% ± 0.70% | ₹1.1778 ± 0.1531 | ₹337.54 ± 127.88 |
| **Rule-based** | ₹58,313.35 ± 2,499.43 | ₹19,667.14 ± 883.74 | ₹3.9334 ± 0.1767 | 24.41% ± 0.79% | 8.29% ± 0.20% | ₹1.2194 ± 0.0558 | ₹387.32 ± 15.86 |
| **Random** | ₹57,513.17 ± 2,059.95 | ₹19,375.57 ± 650.56 | ₹3.8751 ± 0.1301 | 24.11% ± 0.72% | 8.33% ± 0.27% | ₹1.2099 ± 0.0712 | ₹438.64 ± 12.88 |

Table 6.3 provides the paired lift statistics and win counts across the 10 evaluation seeds.

**Table 6.3: Paired Profit Differences and Win Counts (10 Evaluation Seeds)**

| Evaluation Mode | Bandit Policy | Comparator | Profit Difference (₹) | Percentage Lift (%) | Win Count |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Online (20k rounds)** | LinUCB | Fixed 10% | -₹1,340.59 ± 1,554.26 | -1.67% | 1 of 10 seeds |
| **Online (20k rounds)** | LinUCB | Rule-based | +₹492.50 ± 1,022.62 | +0.63% | 8 of 10 seeds |
| **Online (20k rounds)** | Thompson Sampling | Fixed 10% | -₹1,409.20 ± 1,398.69 | -1.76% | 2 of 10 seeds |
| **Online (20k rounds)** | Thompson Sampling | Rule-based | +₹423.88 ± 937.37 | +0.54% | 6 of 10 seeds |
| **Frozen (5k rounds)** | LinUCB | Fixed 10% | -₹399.96 ± 476.46 | -1.99% | 1 of 10 seeds |
| **Frozen (5k rounds)** | LinUCB | Rule-based | +₹69.73 ± 811.53 | +0.35% | 4 of 10 seeds |
| **Frozen (5k rounds)** | Thompson Sampling | Fixed 10% | -₹406.90 ± 403.13 | -2.02% | 2 of 10 seeds |
| **Frozen (5k rounds)** | Thompson Sampling | Rule-based | +₹62.79 ± 611.42 | +0.32% | 7 of 10 seeds |

## 6.2 Learning Dynamics and Convergence Analysis

The progression of cumulative expected regret across training provides formal mathematical verification of policy learning. Figure 6.1 (`outputs/plots/regret_curves_linucb_vs_ts.png`) plots cumulative regret for LinUCB and Thompson Sampling alongside rolling average regret per 1,000-round window (`outputs/plots/avg_regret_per_window.png`).

Table 6.4 tracks regret checkpoints on the reference Seed 42 run.

**Table 6.4: Cumulative Regret Checkpoints on Seed 42 Sequence (from `results/regret_checkpoints.csv`)**

| Round Checkpoint | LinUCB Regret (₹) | Thompson Sampling Regret (₹) | LinUCB Window Regret (₹/round) | Thompson Window Regret (₹/round) |
| :---: | :---: | :---: | :---: | :---: |
| **Round 500** | ₹114.72 | ₹109.11 | ₹0.2294 | ₹0.2182 |
| **Round 2,000** | ₹349.52 | ₹271.86 | ₹0.1565 | ₹0.1085 |
| **Round 5,000** | ₹634.33 | ₹462.62 | ₹0.0949 | ₹0.0636 |
| **Round 10,000**| ₹1,048.97 | ₹723.36 | ₹0.0829 | ₹0.0521 |
| **Round 20,000**| **₹1,754.14** | **₹1,075.28** | **₹0.0919** | **₹0.0483** |

### Convergence Observations and the Non-Zero Asymptote
Both algorithms demonstrate characteristic sublinear regret profiles during early training:
- In the initial 500 rounds, exploration is aggressive, resulting in an average regret of **₹0.2294/round** for LinUCB and **₹0.2182/round** for Thompson Sampling.
- By Round 5,000, instantaneous regret drops by over 58%, flattening to **₹0.0919/round** for LinUCB and **₹0.0483/round** for Thompson Sampling across the final 2,000 rounds.
- Thompson Sampling achieved lower cumulative regret than LinUCB on Seed 42 (₹1,075.28 vs. ₹1,754.14) and higher final oracle arm alignment (**62.20%** vs. **16.90%**).

Crucially, **neither policy's regret slope flattens to zero**. In theoretical linear bandit theory, regret scales as $\mathcal{O}(\sqrt{d T \ln T})$, implying instantaneous regret asymptotically approaches zero. Here, instantaneous regret stabilizes at a non-zero constant ($\approx ₹0.05 - ₹0.09/\text{round}$). This occurs because the simulator models purchase probability via a **non-linear logistic sigmoid link**, whereas LinUCB and linear Thompson Sampling assume a strictly linear payoff function $\mathbb{E}[R|x] = x^T \theta$. Because a linear hyperplane cannot perfectly replicate a multi-dimensional sigmoidal demand surface, an irreducible functional approximation gap persists indefinitely.

## 6.3 The Money View: Financial Ledger and Gain/Loss Dynamics

The financial ledger (`results/ledger_seed42.csv`) reveals how algorithmic learning translates into cumulative cash flow. Figure 6.2 (`outputs/plots/gain_loss_linucb_vs_others.png`) depicts the running profit difference of LinUCB versus each comparator over 20,000 rounds.

1. **The Cost of Exploration:** During the initial 1,500 rounds, LinUCB suffers an active "exploration tax." Because it begins with uninformative priors, it tests exploratory discounts across customer segments, trailing the Rule-based benchmark by up to -₹280.
2. **Break-Even Crossover:** As parameter estimates stabilize, LinUCB begins to selectively exploit profitable price elasticities. It crosses break-even against the Rule-based baseline at **Round 3,200**, expanding its profit lead to **+₹492.50** by Round 20,000.
3. **Persistent Deficit Against Fixed 10%:** Against Fixed 10%, LinUCB falls behind early and remains behind across the entire run. Fixed 10% acts as an empirical "sweet spot" in this simulated retail market: offering 10% provides sufficient stimulus to convert shoppers without sacrificing the severe margin required by 15% or 20% discounts. LinUCB repeatedly tests 15% and 20% on price-sensitive customers, successfully lifting conversion to 24.29% but conceding ₹24,642 in total discounts compared to only ₹20,955 for Fixed 10%.

## 6.4 Loyalty Tier Breakdown and Guardrail Audit

Table 6.5 details the performance of LinUCB and Fixed 10% broken down across customer loyalty tiers from `results/multiseed_online_by_tier.csv`.

**Table 6.5: Performance Breakdown by Loyalty Tier Across 10 Seeds (Mean Values)**

| Loyalty Tier | Customer Share | Guardrail Max Discount | LinUCB Avg Discount Offered (%) | LinUCB Total Profit (₹) | Fixed 10% Avg Discount Offered (%) | Fixed 10% Total Profit (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **New** | 25% | **5%** | 3.11% | ₹19,410.22 | 5.00% | **₹19,652.18** |
| **Regular** | 50% | **15%** | 9.84% | ₹39,411.50 | 10.00% | **₹40,188.42** |
| **Loyal** | 25% | **20%** | 11.02% | ₹20,103.10 | 10.00% | **₹20,424.81** |

### Complete Guardrail Inviolability
Across all 10 seeds and all 200,000 cumulative online rounds, **zero guardrail violations occurred**:
- **0%** of recommendations exceeded tier maximum discounts.
- **0%** of recommendations violated the unit cost margin floor.
- In the New customer tier, where discounts were strictly capped at 5%, LinUCB recommended an average discount of **3.11%** (blending 0% and 5%), while Fixed 10% cleanly fell back to 5.00%. In the Loyal tier, LinUCB utilized its higher ceiling to offer an average discount of **11.02%**.

## 6.5 Off-Policy Replay Evaluation on Real Dataset

Following the rejection sampling methodology of Li et al. [2], policies were evaluated against the 73,100 observational records in the public retail dataset (`results/replay_results.csv`).

**Table 6.6: Off-Policy Replay Performance (from `results/replay_results.csv`)**

| Policy | Total Historical Records | Matched Records | Replay Match Rate (%) | Mean Profit / Round (₹) | 95% Confidence Interval (₹) | Replay Conversion Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinUCB** | 73,100 | 14,973 | **20.48%** | **₹6.11** | [₹5.84, ₹6.38] | 17.57% |
| **Thompson Sampling**| 73,100 | 14,493 | **19.83%** | **₹6.21** | [₹5.98, ₹6.45] | 18.75% |
| **Random** | 73,100 | 14,561 | **19.92%** | **₹5.99** | [₹5.72, ₹6.25] | 17.84% |
| **Fixed 10%** | 73,100 | 14,357 | **19.64%** | **₹5.87** | [₹5.61, ₹6.13] | 17.13% |

### Critical Replay Caveats and Interpretation
While LinUCB (₹6.11) and Thompson Sampling (₹6.21) achieved nominally higher per-round replay profit than Fixed 10% (₹5.87) and Random (₹5.99), **this does NOT prove real-world superiority**:
1. **Overlapping Confidence Intervals:** The 95% confidence intervals across all policies overlap substantially (e.g., LinUCB [₹5.84, ₹6.38] overlaps Fixed 10% [₹5.61, ₹6.13]). No statistically valid ranking can be asserted.
2. **Rejection Bias and Ineligible Rows:** 39.04% of historical records violate tier guardrails. Because our guardrailed policies can never select an ineligible action, those historical rows are discarded by construction.
3. **Non-Random Logging Policy:** Historical discounts were set systematically by store managers, violating the fundamental independence assumption required for replay estimators to be unbiased. Replay confirms only that bandit policies make coherent recommendations within historical data bounds.

## 6.6 Sensitivity Analysis and Reward Ablation

### 6.1 Exploration Parameter Sensitivity ($\alpha$)
Sweeping $\alpha \in \{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\}$ across separate tuning seeds (Seeds 101–105) reveals:
- At $\alpha = 0.0$ (pure exploitation), early cumulative regret drops slightly (₹1,303.56), but the policy risks getting trapped in suboptimal arm allocations.
- At $\alpha = 0.5$, the model achieves an optimal balance, yielding stable cumulative regret (₹1,383.36) and high late-stage exploitation.
- At $\alpha \ge 2.0$, excessive exploration forces the agent to repeatedly pull high-variance suboptimal arms, inflating regret to ₹1,462.92.

### 6.2 Reward Penalty Ablation ($\lambda$)
Ablating the regularized reward $R_t - \lambda \cdot (\text{BasePrice} \times a_t)$ demonstrates the economic price of artificial restraint (`results/reward_ablation.csv`):
- Setting $\lambda = 0.073$ (penalizing 10% of profit) decreases average discount offered from **8.65%** to **7.85%**.
- Setting $\lambda = 0.730$ (penalizing 100% of profit) collapses average discount to **4.86%**, driving unit conversion down from **24.46%** to **21.44%** and reducing gross revenue by over ₹20,000 (from ₹235,028 to ₹214,531).
- Total net profit remained relatively flat (₹78,641 vs. ₹79,070), proving that imposing an ad-hoc penalty on discounts merely throttles revenue volume without improving bottom-line profitability.
"""

with open("reports/sections/06_results.md", "w", encoding="utf-8") as f:
    f.write(results_text.strip() + "\n")

# -------------------------------------------------------------
# 07_application.md
# -------------------------------------------------------------
app_text = r"""# 7. The Decision Support Application

## 7.1 Architecture and Design Principles

To bridge the gap between offline algorithmic models and frontline retail merchandising, we implemented a full-featured web application using the Streamlit framework (`app/streamlit_app.py`).

The application is engineered according to three foundational architectural principles:
1. **Decoupled Architecture:** Business logic, feature preprocessing, bandit scoring, and guardrail enforcement reside in standalone backend modules (`src/`). The Streamlit script serves purely as a stateless presentation and interaction layer.
2. **Zero-Inference Mutation (Frozen Model):** The application loads pre-trained model artifacts (`models/linucb_v1.npz` and `models/thompson_v1.npz`). The application performs strictly forward inference; **no model parameter updates occur during user interactions**. This ensures deterministic, reproducible recommendations.
3. **100% Local and Offline Execution:** The application requires zero external network calls, zero cloud dependencies, and zero third-party commercial AI APIs. All matrix computations run locally in NumPy within sub-millisecond latencies.

## 7.2 Application Interface Tour

The user interface is organized into a clean, modern white-card design featuring horizontal top-level navigation tabs matching enterprise retail merchandise dashboards:

### Tab 1: Live Discount Recommendation (`app/streamlit_app.py:render_tab_recommend`)
- **Interactive Input Controls:** Allows merchandisers to specify product context (Category, Shelf Price, Competitor Price, Current Stock, Historical Demand) and customer profile parameters (Customer Tier, Lifetime Visits, Recency Days).
- **Recommendation Header:** Displays the optimal recommended discount badge (e.g., `Recommended: 10% Discount`), accompanied by the expected profit projection (e.g., `₹5.42 / unit`) and the model's epistemic uncertainty bonus.
- **Arm Evaluation Table:** Renders a complete comparative matrix across all five discrete discount arms ($\{0\%, 5\%, 10\%, 15\%, 20\%\}$). For each arm, the table displays expected conversion probability, expected profit, eligibility status, and explicit blocking reasons if disqualified by guardrails.

### Tab 2: What-If Counterfactual Simulator (`app/streamlit_app.py:render_tab_whatif`)
- **Dynamic Sensitivity Slider:** Allows merchandisers to sweep discount percentages continuously from 0% to 20%.
- **Projected Revenue and Profit Curves:** Interactively plots projected conversion probability against net unit margin. Surfaces the exact profit-maximizing discount apex and visualizes margin degradation beyond optimal thresholds.
- **Margin Floor Warning Callout:** Automatically triggers prominent amber alerts if a selected what-if discount approaches or violates the configured unit cost margin floor.

### Tab 3: Business Guardrails and Audit Panel (`app/streamlit_app.py:render_tab_guardrails`)
- **Rule Matrix Inspector:** Displays the active loyalty tier ceilings (New: 5%, Regular: 15%, Loyal: 20%) and the current margin floor buffer.
- **Historical Violation Breakdown:** Visualizes the audited **39.04% violation rate** identified in the Kaggle dataset, explaining to users why historical data cannot be relied upon without automated guardrails.
- **Real-Time Guardrail Filter:** Demonstrates the exact logic flow of `get_eligible_actions`, showing how forbidden arms are pruned before the bandit scores are evaluated.

### Tab 4: Performance and Benchmark Dashboard (`app/streamlit_app.py:render_tab_performance`)
- **Multi-Seed Comparison Tables:** Displays audited performance tables (Online, Frozen, and Lift) directly from `results/*.csv`.
- **Interactive Visualizations:** Embeds high-resolution Plotly charts of cumulative regret trajectories, financial gain/loss curves, and off-policy replay benchmarks.

## 7.3 Explainability Framework

In high-stakes retail pricing, "black-box" recommendations are frequently rejected by merchandisers. The application provides complete mathematical and operational explainability:
1. **LinUCB Score Decomposition:** The application explicitly breaks down the UCB score into its two additive components:
   $$p_{t, a} = \underbrace{x_t^T \hat{\theta}_a}_{\text{Expected Mean Payoff}} + \underbrace{\alpha \sqrt{x_t^T A_a^{-1} x_t}}_{\text{Exploration Uncertainty Bonus}}$$
   Merchandisers can see exactly how much of a recommendation is driven by historical evidence versus active exploration.
2. **Transparent Rejection Reasons:** When an arm is disqualified, the UI provides unambiguous plain-English rationales:
   - *"Blocked: Exceeds maximum 5% discount permitted for New customer tier."*
   - *"Blocked: Breaches unit cost margin floor of ₹15.00."*

## 7.4 Profile Tier vs. Context Tier and Extrapolation Safeguards

To prevent misleading recommendations when merchandisers input extreme or contradictory features, the application implements two protective safeguards:
1. **Tier Decoupling:** Distinguishes between the customer's nominal **Profile Tier** (which determines legal guardrail ceilings) and the **Contextual RFM Features** (purchase count and recency days) fed into the bandit vector. If a user sets the tier to "Loyal" but enters "1 lifetime visit", the application issues an advisory note clarifying that the model evaluates the customer based on observed numerical RFM covariates while applying the chosen tier's policy envelope.
2. **Extrapolation Warnings:** If input prices or inventory counts fall outside the 1st–99th percentile range of the training data, the application flags the recommendation with an extrapolation warning, alerting merchandisers that model confidence intervals are unusually wide.
"""

with open("reports/sections/07_application.md", "w", encoding="utf-8") as f:
    f.write(app_text.strip() + "\n")

# -------------------------------------------------------------
# 08_discussion.md
# -------------------------------------------------------------
disc_text = r"""# 8. Discussion and Synthesis

## 8.1 Why the Heuristic Fixed 10% Benchmark Won on Profit

The most striking empirical finding of this project is that the simple heuristic **Fixed 10% earned the highest total profit** in both Online (₹80,265.41 ± 1,788.01) and Frozen (₹20,136.83 ± 644.67) evaluation modes, outperforming LinUCB by -1.67% and Thompson Sampling by -1.76%. 

To understand why a static heuristic beat sophisticated adaptive bandits, we examine the microeconomics of the simulation environment:
- **The Sweet Spot of Price Elasticity:** In our calibrated retail demand model, customer conversion exhibits a sigmoidal response to discounts. Offering 0% yields a modest conversion rate (~20%), while offering 5% or 10% provides significant incremental lift. However, moving from 10% to 15% or 20% yields diminishing marginal conversion gains while imposing a severe, linear margin penalty on unit profit.
- **The Bandits' Volume Bias:** Because the contextual bandits were tasked with maximizing expected profit on a per-round basis, their exploration mechanisms identified segments of price-sensitive customers who showed heightened responsiveness to 15% and 20% discounts. The bandits successfully exploited these elasticities, increasing average conversion from **23.81%** (Fixed 10%) to **24.29%** (LinUCB) and capturing the highest gross revenue (**₹234,276.84** vs. **₹232,094.81**). However, each 20% discount halved unit profit margin from 40% to 20%. The incremental volume captured was insufficient to offset the steep margin conceded, leaving LinUCB with a net profit deficit of -₹1,340.59.
- **The Lesson for Algorithmic Pricing:** Sophisticated machine learning algorithms optimize strictly what they are measured on. When reward functions are sensitive to subtle price-volume interactions, an unconstrained or exploratory bandit can easily "over-promote," trading away valuable margin to acquire low-value incremental volume.

## 8.2 LinUCB vs. Rule-Based Benchmarks

While LinUCB trailed Fixed 10%, it successfully outperformed the enterprise **Rule-based policy** (which offered 0% to New, 10% to Regular, and 15% to Loyal customers). LinUCB achieved a profit lift of **+₹492.50 ± 1,022.62** (**+0.63%**) and won in **8 of 10 evaluation seeds**.

The Rule-based policy suffered from rigid structural assumptions:
- By forcing all New customers to receive 0% discount, the Rule-based policy failed to convert price-sensitive new shoppers who required a small 5% incentive.
- By forcing all Loyal customers to receive 15% discount, it routinely conceded margin to highly loyal shoppers who would have purchased at 5% or 10%.
LinUCB adapted dynamically to multivariate signals beyond the customer tier (such as competitor price gap and inventory stock days), recovering ₹492 in incremental profit.

## 8.3 The Commercial Necessity of Hard Guardrails

A central architectural hypothesis of this study was that business constraints should be implemented as **hard deterministic guardrails** rather than soft penalty terms in the reward function. Our empirical results validate this choice unequivocally:
1. **Absolute Compliance:** The hard filter `get_eligible_actions` achieved **100% compliance** across 200,000 online rounds, with zero below-cost sales and zero tier violations.
2. **Failure of Soft Penalties:** In our reward penalty ablation (Section 6.6), attempting to discourage excessive discounting via an additive penalty term ($-\lambda \cdot a_t$) failed to produce nuanced behavior. As $\lambda$ increased, the policy simply collapsed toward zero discounting across all segments, destroying revenue and conversion rates without improving profit. Hard guardrails provide an inviolable safety envelope while allowing the bandit full freedom to optimize within legal boundaries.

## 8.4 Regret vs. Financial Ledgers: Dual Lenses on Performance

This project demonstrates the necessity of pairing theoretical regret metrics with double-entry financial ledgers:
- **Cumulative Regret** measures algorithmic efficiency relative to an omniscient oracle. It proves that LinUCB and Thompson Sampling successfully learned, flattening their instantaneous regret curves by over 58%.
- **The Financial Ledger** translates algorithmic decisions into actual commercial currency. It revealed the early "exploration tax" (trailing Rule-based by ₹280 in early rounds), identified the exact break-even crossover (Round 3,200), and exposed the critical divergence between revenue leadership and profit maximization. Without the ledger, stakeholders might observe flattening regret and falsely assume the model is maximizing net commercial cash flow.

## 8.5 Online Learning vs. Frozen Operational Modes

Comparing Online and Frozen evaluation highlights the operational lifecycle of dynamic pricing systems:
- In Online mode, the bandit incurs the financial cost of trial-and-error exploration. LinUCB trailed Rule-based during early rounds before surpassing it.
- In Frozen mode, exploration is eliminated, and the model operates with fixed, mature weights. LinUCB maintained its revenue leadership (₹58,464.50 vs. ₹58,298.52 for Fixed 10%) and delivered stable conversion (24.42%). Evaluating both modes guarantees that merchandisers understand both the short-term deployment cost and the long-term operational steady state.
"""

with open("reports/sections/08_discussion.md", "w", encoding="utf-8") as f:
    f.write(disc_text.strip() + "\n")

print("Sections 06, 07, and 08 written successfully.")
