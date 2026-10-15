# -*- coding: utf-8 -*-
"""
Generates sections 00 to 05 of the Final Project Report in reports/sections/.
Strict adherence to WRAP-UP SHARED DEFINITIONS and Pass A-C rules.
"""

import os

os.makedirs("reports/sections", exist_ok=True)

# -------------------------------------------------------------
# 00_abstract.md (150 - 220 words, strictly NO citations)
# -------------------------------------------------------------
abstract_text = """# Abstract

Promotional discounting in retail involves an inherent financial trade-off: price reductions stimulate sales volume but simultaneously erode unit profit margins. Traditional static heuristics and rigid category-level rules fail to account for heterogenous customer price sensitivities, while unconstrained machine learning approaches risk severe margin destruction. This project formulates dynamic discount recommendation as a contextual multi-armed bandit problem, deploying Linear Upper Confidence Bound (LinUCB) and contextual Thompson Sampling to personalize discount decisions across five discrete actions (0%, 5%, 10%, 15%, and 20%). Programmatic business guardrails—comprising customer loyalty tier discount ceilings and a strict unit cost margin floor—are integrated directly into the action selection loop, guaranteeing operational safety by construction. Evaluating policies over 10 independent random seeds across 20,000 simulated online rounds reveals that the heuristic benchmark Fixed 10% achieved the highest cumulative profit (₹80,265.41 ± 1,788.01), whereas LinUCB generated the highest gross revenue (₹234,276.84 ± 4,476.98) and outperformed the business Rule-based baseline in 8 of 10 seeds (+₹492.50 ± 1,022.62 profit lift). In frozen evaluation across 5,000 held-out rounds, policies preserved relative rankings. The principal limitation is that outcomes are derived within a semi-synthetic simulation environment calibrated against an aggregated public retail dataset lacking individual customer identifiers, with an assumed uniform cost structure.
"""

with open("reports/sections/00_abstract.md", "w", encoding="utf-8") as f:
    f.write(abstract_text.strip() + "\n")

# -------------------------------------------------------------
# 01_introduction.md
# -------------------------------------------------------------
intro_text = """# 1. Introduction

## 1.1 Motivation and Business Problem

In modern retail commerce, promotional discounting is among the most powerful yet dangerous operational levers available to merchandisers. When deployed effectively, promotional price cuts accelerate inventory turnover, stimulate customer acquisition, increase conversion rates, and prevent perishable or seasonal stock obsolescence. However, when applied indiscriminately through uniform blanket promotions (e.g., store-wide "20% off everything" campaigns), discounting severely erodes net unit margins, cannibalizes full-price revenue, and conditions price-sensitive consumers to defer purchases until promotions occur.

The central managerial challenge is that customer price responsiveness is highly heterogeneous. A highly committed customer purchasing an essential staple may convert at full retail price without any promotional concession, making a discount pure margin loss. Conversely, a price-sensitive prospect browsing a discretionary item may require a targeted price incentive to complete the transaction. Traditional retail systems address this problem with crude categorical heuristics (e.g., fixed markdowns per department) or static customer segmentation rules. These static approaches cannot adapt dynamically to multivariate customer contexts, real-time inventory pressures, or fluctuating competitor pricing.

## 1.2 The Exploration-Exploitation Dilemma in Dynamic Pricing

Standard supervised machine learning models attempt to predict purchase probability as a function of customer features and offered prices. However, applying supervised learning to pricing decisions creates an acute feedback loop problem: historical transaction logs reflect only the prices that past policies chose to offer. We do not observe counterfactual outcomes—that is, whether a customer who bought an item at a 20% discount would have bought it at 10% or at full price.

Attempting to resolve this via classical randomized controlled trials ($A/B$ testing) imposes substantial business opportunity costs. In an $A/B$ test, suboptimal pricing variants must be allocated fixed traffic shares for prolonged periods to establish statistical significance, directly forfeiting profit on underperforming arms. 

The multi-armed bandit (MAB) framework addresses this fundamental trade-off between **exploration** (testing discount levels to infer customer price elasticity) and **exploitation** (recommending the discount level that maximizes immediate expected return). Contextual bandits extend this paradigm by conditioning action selection on observable customer, product, and market covariates. Rather than learning a single global discount policy, a contextual bandit personalizes recommendations dynamically while actively bounding cumulative regret.

## 1.3 Project Scope and Core Objectives

This project designs, implements, evaluates, and deploys a complete contextual multi-armed bandit framework for personalized dynamic discount recommendation. The system addresses the business dilemma by:

1. **Formulating Action Spaces and Payoffs:** Discretizing promotional discounting into five actionable arms ($\\{0\\%, 5\\%, 10\\%, 15\\%, 20\\%\\}$) and defining objective reward functions aligned with net profit margin rather than vanity revenue metrics.
2. **Enforcing Hard Commercial Guardrails:** Constructing an inviolable programmatic safety envelope that restricts allowable discount actions based on customer loyalty tiers and a strict unit cost margin floor, eliminating below-cost sales by construction.
3. **Implementing Online Adaptive Policies:** Constructing Linear Upper Confidence Bound (LinUCB) and Bayesian contextual Thompson Sampling policies equipped with efficient incremental matrix updates (Sherman-Morrison formula).
4. **Benchmarking Against Realistic Baselines:** Evaluating the bandit policies against standard retail heuristics, including uniform random selection, static fixed discounting (Fixed 10%), and an enterprise tier-differentiated Rule-based policy.
5. **Rigorous Dual-Mode Evaluation:** Measuring algorithmic convergence and financial performance across 10 independent random seeds using both **Online** (learning in-flight over 20,000 rounds) and **Frozen** (held-out policy evaluation over 5,000 rounds) protocols, complemented by off-policy replay on historical retail data.
6. **Delivering an Explainable Operational Frontend:** Building a fully self-contained Streamlit application that surfaces model recommendations, visualizes exploration uncertainty bonuses, provides counterfactual "what-if" simulations, and audits guardrail compliance.

## 1.4 Research Questions

The empirical investigation is structured around four primary research questions:

- **RQ1 (Convergence and Regret):** Can linear contextual bandit algorithms achieve sublinear cumulative regret when interacting with a non-linear simulated demand environment calibrated against retail data?
- **RQ2 (Financial Value and Lift):** How do contextual bandit policies compare against industry-standard fixed and rule-based discounting heuristics in terms of cumulative net profit, top-line revenue, and unit conversion rates within the simulator?
- **RQ3 (Guardrail Inviolability):** Can deterministic business guardrails enforce 100% compliance with loyalty tier policies and margin floors without destabilizing the bandit's exploration mechanisms?
- **RQ4 (Offline Replay Viability):** Does off-policy evaluation via rejection sampling on un-randomized historical retail logs provide a reliable proxy for live bandit deployment?

## 1.5 Report Organization

The remainder of this report is organized as follows:
- **Section 2 (Background):** Reviews multi-armed bandits, contextual decision processes, the mathematical formulation of LinUCB and Thompson Sampling, the distinction between bandits and full reinforcement learning, and customer loyalty segmentation principles.
- **Section 3 (Data):** Details the public retail dataset, feature engineering pipeline, context vector formulation, and explicit boundary between real historical data and simulated components.
- **Section 4 (Methodology):** Presents the simulation environment, logistic purchase probability calibration, guardrail architecture, reward formulations, policy implementations, and training protocols.
- **Section 5 (Evaluation Design):** Details the regret formulation, online vs. frozen evaluation modes, multi-seed protocol, financial transaction ledger, and off-policy replay methodology.
- **Section 6 (Results):** Reports the headline performance metrics, learning curves, financial gain/loss dynamics, tier breakdowns, replay results, and sensitivity/ablation sweeps.
- **Section 7 (The Application):** Describes the architecture, user interface, explainability module, and offline design of the operational Streamlit decision tool.
- **Section 8 (Discussion):** Analyzes why heuristic fixed discounting won on profit while bandits led on revenue, discusses the commercial necessity of hard constraints, and interprets findings honestly.
- **Section 9 (Limitations and Threats to Validity):** Provides an unvarnished audit of data, simulation, modeling, and evaluation limitations.
- **Section 10 (Future Work):** Outlines technical paths for non-linear models, online streaming architectures, and richer identifier datasets.
- **Section 11 (Conclusion):** Summarizes core contributions, empirical findings, and operational recommendations.
- **Sections 12 & 13:** Provide formal references and comprehensive technical appendices covering reproducibility, design decisions, terminology, and hyperparameters.
"""

with open("reports/sections/01_introduction.md", "w", encoding="utf-8") as f:
    f.write(intro_text.strip() + "\n")

# -------------------------------------------------------------
# 02_background.md
# -------------------------------------------------------------
background_text = """# 2. Background and Theoretical Foundations

## 2.1 Multi-Armed Bandits and Contextual Bandits

The classic Multi-Armed Bandit (MAB) problem formalizes the fundamental tension between acquiring new information (exploration) and capitalizing on existing knowledge (exploitation). In a standard $K$-armed bandit, an agent repeatedly chooses among $K$ discrete actions $a \in \\mathcal{A} = \\{1, \\dots, K\\}$ across discrete time steps $t = 1, \\dots, T$. Upon pulling arm $a_t$, the agent receives a stochastic reward $r_t \\sim \\mathcal{D}_{a_t}$ drawn from an unknown reward distribution associated with that arm. The objective is to maximize cumulative reward $\\sum_{t=1}^T r_t$, or equivalently, to minimize cumulative pseudo-regret relative to an omniscient policy that consistently pulls the single best arm:

$$R(T) = T \\cdot \\mu^* - \\sum_{t=1}^T \\mathbb{E}[r_{t, a_t}]$$

where $\\mu^* = \\max_{a \\in \\mathcal{A}} \\mathbb{E}_{r \\sim \\mathcal{D}_a}[r]$.

While classic MAB algorithms (such as $\\epsilon$-greedy or standard UCB1) excel in stationary environments where all users are identical, they cannot personalize recommendations. In retail discounting, a customer's propensity to buy depends heavily on observable features such as past purchase history, brand affinity, current inventory levels, and external price gaps.

The **Contextual Multi-Armed Bandit** framework extends the classic setting by incorporating side information. At each round $t$:
1. The environment presents an observable context vector $x_t \\in \\mathbb{R}^d$ characterizing the customer, item, and market state.
2. The agent selects an action $a_t \\in \\mathcal{A}_t$, where $\\mathcal{A}_t \\subseteq \\mathcal{A}$ denotes the set of legally permissible actions under business constraints.
3. The environment reveals a payoff $r_t(a_t) \\in \\mathbb{R}$ drawn from a conditional distribution $\\mathcal{D}(r | x_t, a_t)$. Payoffs for unselected counterfactual arms $a \\ne a_t$ remain unobserved.
4. The agent updates its internal parameter estimates using the transition tuple $(x_t, a_t, r_t(a_t))$.

## 2.2 The Linear Upper Confidence Bound (LinUCB) Algorithm

The Linear Upper Confidence Bound algorithm, introduced by Li et al. [1], assumes a linear relationship between the context vector $x_t$ and the expected payoff of each arm $a$:

$$\\mathbb{E}[r_{t, a} | x_t] = x_t^T \\theta_a^*$$

where $\\theta_a^* \\in \\mathbb{R}^d$ is an unknown, fixed coefficient vector specific to arm $a$.

### 2.2.1 Parameter Estimation via Ridge Regression
For each arm $a$, the agent maintains ridge regression estimates based on its observed history $\\mathcal{H}_a = \\{(x_\\tau, r_\\tau) : a_\\tau = a\\}$. Let $D_a \\in \\mathbb{R}^{m_a \\times d}$ denote the matrix whose rows are the context vectors observed when arm $a$ was pulled, and let $c_a \\in \\mathbb{R}^{m_a}$ denote the corresponding vector of observed rewards. The regularized ridge regression estimate $\\hat{\\theta}_a$ is given by:

$$\\hat{\\theta}_a = \\left( D_a^T D_a + I_d \\right)^{-1} D_a^T c_a = A_a^{-1} b_a$$

where $A_a = D_a^T D_a + I_d \\in \\mathbb{R}^{d \\times d}$ is the regularized design covariance matrix, and $b_a = D_a^T c_a \\in \\mathbb{R}^d$ accumulates context-reward interactions. The identity matrix $I_d$ acts as an $L_2$ regularizer ensuring that $A_a$ is strictly positive definite and invertible at all rounds.

### 2.2.2 Confidence Bounds and Action Selection
Applying matrix concentration inequalities, Li et al. [1] proved that with high probability, the estimation error is bounded by:

$$\\left| x_t^T \\hat{\\theta}_a - x_t^T \\theta_a^* \\right| \\le \\alpha \\sqrt{x_t^T A_a^{-1} x_t}$$

where $\\alpha = 1 + \\sqrt{\\ln(2 / \\delta) / 2} > 0$ controls the confidence hyperparameter. The term $\\sqrt{x_t^T A_a^{-1} x_t}$ represents the standard deviation of the ridge estimator in direction $x_t$, serving as an explicit measure of model uncertainty.

At round $t$, LinUCB computes an upper confidence bound score $p_{t, a}$ for every eligible action $a \\in \\mathcal{A}_t$:

$$p_{t, a} = \\underbrace{x_t^T \\hat{\\theta}_a}_{\\text{Exploitation: Expected Payoff}} + \\underbrace{\\alpha \\sqrt{x_t^T A_a^{-1} x_t}}_{\\text{Exploration: Uncertainty Bonus}}$$

The policy selects the arm that maximizes this optimistic bound:

$$a_t = \\arg\\max_{a \\in \\mathcal{A}_t} p_{t, a}$$

Upon observing reward $r_t$, the sufficient statistics are updated incrementally:

$$A_{a_t} \\leftarrow A_{a_t} + x_t x_t^T, \\quad b_{a_t} \\leftarrow b_{a_t} + r_t x_t$$

### 2.2.3 Sherman-Morrison Incremental Matrix Inversion
In a naive implementation, evaluating the uncertainty bonus requires inverting $A_a \\in \\mathbb{R}^{d \\times d}$ at every round, incurring an $\\mathcal{O}(d^3)$ computational cost per update. To achieve production efficiency, we implement the rank-1 Sherman-Morrison update directly on the inverse covariance matrix $M_a = A_a^{-1}$:

$$M_{a_t} \\leftarrow M_{a_t} - \\frac{M_{a_t} x_t x_t^T M_{a_t}}{1 + x_t^T M_{a_t} x_t}$$

This formulation reduces the computational complexity of updates from $\\mathcal{O}(d^3)$ to $\\mathcal{O}(d^2)$, enabling high-throughput real-time inference without loss of numerical precision.

## 2.3 Contextual Thompson Sampling

Thompson Sampling, originally proposed by Thompson [3] and analyzed for contextual linear payoffs by Agrawal and Goyal [4] and Chapelle and Li [5], operates under a Bayesian framework. Rather than constructing deterministic upper confidence bounds, Thompson Sampling maintains a posterior probability distribution over the model parameters and samples actions according to their posterior probability of being optimal.

### 2.3.1 Likelihood and Prior
Assuming Gaussian observation noise $r_t(a) \\sim \\mathcal{N}(x_t^T \\theta_a^*, v^2)$ with prior $\\theta_a \\sim \\mathcal{N}(0, v^2 I_d)$, the posterior distribution of $\\theta_a$ at round $t$ remains conjugate Gaussian:

$$\\theta_a \\sim \\mathcal{N}\\left( \\hat{\\theta}_a, v^2 B_a^{-1} \\right)$$

where $B_a = I_d + \\sum_{\\tau: a_\\tau = a} x_\\tau x_\\tau^T$ and $\\hat{\\theta}_a = B_a^{-1} \\sum_{\\tau: a_\\tau = a} r_\\tau x_\\tau$.

### 2.3.2 Posterior Sampling Decision Rule
At round $t$, for each eligible arm $a \\in \\mathcal{A}_t$, the algorithm draws a randomized parameter vector $\\tilde{\\theta}_a$ from its current posterior:

$$\\tilde{\\theta}_a \\sim \\mathcal{N}\\left( \\hat{\\theta}_a, v^2 B_a^{-1} \\right)$$

The algorithm then executes the greedy action with respect to the sampled parameter vectors:

$$a_t = \\arg\\max_{a \\in \\mathcal{A}_t} x_t^T \\tilde{\\theta}_a$$

Upon observing reward $r_t$, the sufficient statistics $B_{a_t}$ and the target vector are updated identically to LinUCB. Thompson Sampling naturally balances exploration and exploitation: arms with high parameter variance exhibit wider posterior distributions, resulting in higher sampling dispersion and occasional selection, while well-estimated arms converge to deterministic exploitation.

## 2.4 Bandits vs. Deep Learning vs. Full Reinforcement Learning

To establish the methodological boundaries of this project, it is essential to distinguish contextual bandits from full Reinforcement Learning (RL) and Deep Learning:

1. **Contextual Bandits vs. Full Reinforcement Learning:** As defined by Sutton and Barto [7], full RL operates in a Markov Decision Process (MDP) where action $a_t$ influences both the immediate reward $r_t$ and the future state distribution $s_{t+1} \\sim P(s_{t+1} | s_t, a_t)$. Full RL models long-term sequential planning using value iteration, Q-learning, or policy gradients. In contextual bandits, the environment produces an exogenous context $x_t$ at each round that is conditionally independent of past actions. Because the discount offered to customer $A$ at 10:00 AM does not alter the feature vector of unrelated customer $B$ at 10:05 AM, the state-transition dynamics collapse, and the problem reduces to single-step associative reward maximization. This structural simplification drastically improves sample efficiency, eliminating the instability and convergence challenges of deep RL.
2. **Linear Bandits vs. Deep Contextual Bandits:** Non-linear contextual bandits (such as NeuralUCB [8]) substitute linear ridge regression with deep neural networks to approximate non-linear payoff surfaces. However, deep bandits require frequent gradient updates, introduce non-convex loss surfaces, suffer from catastrophic forgetting, and require orders of magnitude more samples to form well-calibrated epistemic confidence bounds. For operational retail discount recommendation, linear contextual bandits provide strict mathematical guarantees, closed-form updates, microsecond latency, and interpretable coefficient weightings.

## 2.5 Customer Segmentation and RFM Principles

Customer segmentation in this project is grounded in the classical Recency, Frequency, Monetary (RFM) marketing framework introduced by Hughes [9]:
- **Recency ($R$):** Elapsed days since the customer's last purchase. Customers with recent engagement exhibit higher baseline conversion propensities.
- **Frequency ($F$):** Total count of completed transactions across the customer's lifetime. High-frequency purchasers indicate established loyalty.
- **Monetary ($M$):** Cumulative spending or historical average basket value.

In commercial practice, loyalty programs partition customers into categorical tiers (e.g., New, Regular, Loyal) based on RFM thresholds. From an economic perspective, granting identical discounts across tiers is suboptimal. Offering steep discounts to highly loyal, price-insensitive shoppers forfeits margin without driving incremental volume (deadweight margin loss). Conversely, denying promotional incentives to price-sensitive prospects stifles acquisition. Integrating RFM-derived loyalty tiers into the contextual bandit allows the model to personalize promotions while honoring tiered business policies.
"""

with open("reports/sections/02_background.md", "w", encoding="utf-8") as f:
    f.write(background_text.strip() + "\n")

# -------------------------------------------------------------
# 03_data.md
# -------------------------------------------------------------
data_text = """# 3. Data Pipeline and Feature Formulation

## 3.1 Public Retail Dataset Overview

The empirical foundation of this study utilizes a public retail dataset sourced from Kaggle [10], containing store inventory and demand tracking logs. The dataset comprises 73,100 clean records after filtering, capturing store-product-day level commercial activity across multiple product categories, pricing regimes, inventory balances, and historical demand levels.

Each record represents an aggregated observational snapshot rather than an individual shopping cart transaction. Key primary fields include:
- `Store_ID`: Categorical identifier for retail physical branches.
- `Product_ID`: Unique stock keeping unit (SKU) identifier.
- `Category`: High-level merchandise category (e.g., Groceries, Electronics, Apparel).
- `Base_Price`: Standard un-discounted retail shelf price in rupees (₹).
- `Discount_Applied`: Historical percentage discount recorded for that product-day.
- `Historical_Demand`: Past baseline unit velocity recorded for the product.
- `Current_Stock`: Units available in store inventory at the time of observation.
- `Competitor_Price`: Prevailing market price observed for identical or substitute goods.

## 3.2 Cleaning and Preprocessing Validation

Data cleaning followed a strict validation protocol to ensure mathematical consistency:
1. **Invalid Price Removal:** Verified that all `Base_Price` and `Competitor_Price` entries were strictly positive ($> 0$). No negative or zero prices were identified.
2. **Missing Value Imputation:** Categorical fields had zero missing values. Numerical outliers in stock and demand were capped at the 99.9th percentile to prevent leverage distortion in linear ridge regression.
3. **Discount Discretization:** Historical recorded discounts were mapped to their nearest nominal evaluation arm: $\\{0\\%, 5\\%, 10\\%, 15\\%, 20\\%\\}$.

## 3.3 Derived Indicators and Feature Engineering

To provide rich contextual signals for the bandit policies, several domain-specific features were engineered from the raw table:

- **Demand Indicator (Tertile Split):** A categorical demand classification was derived by computing category-specific tertile boundaries on `Historical_Demand`. Products are partitioned into `Low`, `Medium`, and `High` demand bands.
- **Discount Amount & Discounted Price:**
  $$\\text{DiscountAmount} = \\text{BasePrice} \\times a_t$$
  $$\\text{DiscountedPrice} = \\text{BasePrice} - \\text{DiscountAmount}$$
- **Price Gap Percentage:** Relative price positioning versus local competition:
  $$\\text{PriceGapPct} = \\frac{\\text{BasePrice} - \\text{CompetitorPrice}}{\\text{CompetitorPrice}}$$
  A positive gap indicates that the store is more expensive than the competitor, elevating customer price sensitivity.
- **Demand-to-Inventory Ratio & Stock Coverage Days:**
  $$\\text{DemandToInventoryRatio} = \\frac{\\text{HistoricalDemand}}{\\max(1, \\text{CurrentStock})}$$
  $$\\text{StockCoverageDays} = \\frac{\\text{CurrentStock}}{\\max(1, \\text{HistoricalDemand})}$$
- **Inventory Status:** Categorical binning of stock coverage into `Critical Low`, `Balanced`, and `Overstocked`.
- **Cost Price and Unit Margin:** Because real supplier procurement invoices are proprietary and absent from the public dataset, unit cost is modeled under an **assumed 40% margin rate**:
  $$\\text{CostPrice} = \\text{BasePrice} \\times (1 - 0.40) = 0.60 \\times \\text{BasePrice}$$
  $$\\text{UnitMargin} = \\text{DiscountedPrice} - \\text{CostPrice} = \\text{BasePrice} \\times (0.40 - a_t)$$
  Under this structure, offering a 20% discount halves the percentage margin from 40% to 20%, illustrating the steep financial cost of excessive discounting.
- **Temporal Calendar Features:** Extracted `DayOfWeek`, `Month`, and binary `IsWeekend` indicators to capture cyclical shopping variations.

## 3.4 Context Vector Formulation

At each decision round $t$, the bandit observes a 20-dimensional numerical context vector $x_t \\in \\mathbb{R}^{20}$. Table 3.1 details the constituent features, their provenance, and their preprocessing transformations.

**Table 3.1: Complete 20-Dimensional Context Vector Formulation**

| Dimension | Feature Name | Description | Provenance | Data Type / Encoding |
| :---: | :--- | :--- | :--- | :--- |
| $x^{(1)}$ | `BasePrice_scaled` | Standardized retail shelf price | Public Dataset | Continuous (StandardScaler) |
| $x^{(2)}$ | `CompetitorPrice_scaled` | Standardized competitor shelf price | Public Dataset | Continuous (StandardScaler) |
| $x^{(3)}$ | `PriceGapPct` | Relative price gap vs. competitor | Derived | Continuous (StandardScaler) |
| $x^{(4)}$ | `CurrentStock_scaled` | Current store inventory count | Public Dataset | Continuous (StandardScaler) |
| $x^{(5)}$ | `HistoricalDemand_scaled` | Baseline historical unit velocity | Public Dataset | Continuous (StandardScaler) |
| $x^{(6)}$ | `DemandToInventoryRatio` | Ratio of velocity to current inventory | Derived | Continuous (StandardScaler) |
| $x^{(7)}$ | `StockCoverageDays` | Estimated days of inventory remaining | Derived | Continuous (StandardScaler) |
| $x^{(8)}$ | `PriceSensitivity` | Synthetic latent customer elasticity | Simulated | Continuous ($[0.2, 1.0]$) |
| $x^{(9)}$ | `PurchaseCount_scaled` | Customer lifetime transaction frequency | Simulated | Continuous (StandardScaler) |
| $x^{(10)}$ | `RecencyDays_scaled` | Days elapsed since customer's last visit | Simulated | Continuous (StandardScaler) |
| $x^{(11)}$ | `CustomerTier_New` | Indicator: customer lifetime visits $\\le 2$ | Simulated | Binary $\\{0, 1\\}$ |
| $x^{(12)}$ | `CustomerTier_Regular` | Indicator: customer lifetime visits $3 - 9$ | Simulated | Binary $\\{0, 1\\}$ |
| $x^{(13)}$ | `CustomerTier_Loyal` | Indicator: customer lifetime visits $\\ge 10$ | Simulated | Binary $\\{0, 1\\}$ |
| $x^{(14)}$ | `DemandLevel_Low` | Category demand tertile: Low | Derived | Binary $\\{0, 1\\}$ |
| $x^{(15)}$ | `DemandLevel_Med` | Category demand tertile: Medium | Derived | Binary $\\{0, 1\\}$ |
| $x^{(16)}$ | `DemandLevel_High` | Category demand tertile: High | Derived | Binary $\\{0, 1\\}$ |
| $x^{(17)}$ | `IsWeekend` | Saturday or Sunday indicator | Derived | Binary $\\{0, 1\\}$ |
| $x^{(18)}$ | `Month_sin` | Cyclical month representation | Derived | Continuous ($[-1, 1]$) |
| $x^{(19)}$ | `Month_cos` | Cyclical month representation | Derived | Continuous ($[-1, 1]$) |
| $x^{(20)}$ | `Bias_Intercept` | Constant unit intercept | Architecture | Constant ($1.0$) |

Continuous numerical features are standardized to zero mean and unit variance using scaling parameters fitted strictly on training observations, preventing data leakage.

## 3.5 What is Real and What is Simulated (Strict Boundary)

To maintain absolute scientific integrity and avoid misrepresenting results, we establish an explicit, unequivocal boundary between real empirical data and simulated components:

1. **The Public Retail Dataset:** The historical dataset consists of 73,100 store/product/day rows from Kaggle [10]. It contains **no individual customer IDs**, no longitudinal user shopping carts, and no counterfactual conversion outcomes at alternative prices. Per project guidelines, this dataset is designated strictly as a "public retail dataset" pending author verification of Kaggle provenance.
2. **Simulated Customer Registry:** The 8,000 synthetic customer profiles—including their persistent customer IDs, assigned loyalty tiers (`New`, `Regular`, `Loyal`), historical purchase counts, recency days, and latent price sensitivity values—are **simulated by design**.
3. **Simulated Basket Value and Purchase Outcomes:** Individual customer shopping sessions, transaction basket sizes (units per purchase), and the binary conversion outcome $Y_t \\in \\{0, 1\\}$ are **simulated by design** via the calibrated simulator.
4. **Assumed Cost Structure:** CostPrice is derived under an **assumed 40% margin rate**. No real wholesale supplier cost data was present in the public dataset.
5. **Simulated Currency:** All financial metrics (revenue, profit, discount conceded) reported in rupees (₹) are **simulated rupees** generated within the simulation environment.
"""

with open("reports/sections/03_data.md", "w", encoding="utf-8") as f:
    f.write(data_text.strip() + "\n")

# -------------------------------------------------------------
# 04_methodology.md
# -------------------------------------------------------------
methodology_text = """# 4. Methodology and System Architecture

## 4.1 The Simulation Environment and Demand Calibration

Because contextual bandits require interactive feedback loops and counterfactual observation, offline training requires an interactive simulation environment (`simulator.py`).

### 4.1.1 The Purchase Probability Formulation
When presented with context $x_t$ and an offered discount arm $a_t \\in \\{0.0, 0.05, 0.10, 0.15, 0.20\\}$, the probability that the customer completes a purchase is modeled via a logistic sigmoid link:

$$P(Y_t = 1 | x_t, a_t) = \\sigma\\left( z(x_t, a_t) \\right) = \\frac{1}{1 + e^{-z(x_t, a_t)}}$$

where the latent log-odds logit $z(x_t, a_t)$ is structured as:

$$z(x_t, a_t) = \\beta_0 + \\beta_{\\text{disc}} \\cdot a_t + \\beta_{\\text{sens}} \\cdot \\text{PriceSensitivity}_t + \\beta_{\\text{gap}} \\cdot \\text{PriceGapPct}_t + \\beta_{\\text{stock}} \\cdot \\text{StockRatio}_t + \\beta_{\\text{tier}} \\cdot \\mathbb{I}_{\\text{tier}_t}$$

- **Baseline Intercept ($\\beta_0$):** Anchors the intrinsic conversion rate for products priced at parity with zero discount.
- **Discount Stimulus ($\\beta_{\\text{disc}} > 0$):** Quantifies the promotional conversion lift per percentage point of discount.
- **Price Sensitivity Coupling ($\\beta_{\\text{sens}} < 0$):** Modulates the baseline reluctance of price-sensitive shoppers, interacting with offered discounts.
- **Competitor Gap Impact ($\\beta_{\\text{gap}} < 0$):** Suppresses purchase probability when store price exceeds competitor shelf price.
- **Inventory Urgency ($\\beta_{\\text{stock}} > 0$):** Slightly elevates conversion when perceived scarcity or high velocity occurs.

### 4.1.2 Empirical Calibration via Logistic Regression
To prevent arbitrary behavioral assumptions, the coefficient vector $\\beta$ was calibrated against historical demand classifications in the retail dataset using a logistic regression model. The empirical calibration yielded a Receiver Operating Characteristic Area Under the Curve (**ROC-AUC**) of **0.6322**. 

An AUC of 0.6322 indicates a modest, directionally valid predictive signal over random chance ($0.50$), consistent with noisy aggregated retail demand. Crucially, a modest AUC demonstrates that the simulator does not embed deterministic or trivial shortcuts; bandit policies must actively learn subtle multivariate trade-offs under substantial residual variance.

### 4.1.3 The 8,000-Customer Synthetic Registry and Counterfactuals
The environment instantiates a persistent registry of 8,000 synthetic customers partitioned into three RFM loyalty tiers:
- **New Customers (25% share):** Lifetime purchases $\\le 2$, high recency days ($> 60$ days), high price sensitivity (mean $\\mu = 0.75$).
- **Regular Customers (50% share):** Lifetime purchases $3 - 9$, moderate recency ($15 - 60$ days), moderate sensitivity (mean $\\mu = 0.50$).
- **Loyal Customers (25% share):** Lifetime purchases $\\ge 10$, low recency ($< 15$ days), low price sensitivity (mean $\\mu = 0.25$).

At each round $t$, the simulator evaluates the true purchase probability across **all five counterfactual discount arms** simultaneously. This allows exact determination of the **Oracle action** $a_t^*$:

$$a_t^* = \\arg\\max_{a \\in \\mathcal{A}_t} \\mathbb{E}[R_t(a) | x_t] = \\arg\\max_{a \\in \\mathcal{A}_t} P(Y_t = 1 | x_t, a) \\cdot \\text{UnitMargin}(a) \\cdot \\text{BasketSize}$$

## 4.2 Programmatic Business Guardrails

In commercial retail, unconstrained algorithmic optimization is unacceptable. A pure profit-maximizing bandit operating on price-sensitive customers might learn to offer predatory markdowns that violate supplier minimum advertised price (MAP) agreements, degrade brand equity, or trigger below-cost sales. We introduce a hard programmatic guardrail layer (`guardrails.py`) that filters the available arm set $\\mathcal{A}$ to an eligible subset $\\mathcal{A}_t = \\text{get\\_eligible\\_actions}(x_t) \\subseteq \\mathcal{A}$.

### 4.2.1 Loyalty Tier Ceilings
Retailers reserve their steepest promotions for their most valuable customers to prevent customer attrition and cultivate retention. The tier guardrail enforces strict promotional discount caps:
- **New Tier:** Maximum permissible discount = **5%** (Eligible arms: $\\{0\\%, 5\\%\\}$).
- **Regular Tier:** Maximum permissible discount = **15%** (Eligible arms: $\\{0\\%, 5\\%, 10\\%, 15\\%\\}$).
- **Loyal Tier:** Maximum permissible discount = **20%** (Eligible arms: $\\{0\\%, 5\\%, 10\\%, 15\\%, 20\\%\\}$).

### 4.2.2 Unit Cost Margin Floor
To protect gross profitability, no discount may reduce unit selling price below cost plus an essential margin buffer:

$$\\text{DiscountedPrice}(a) \\ge \\text{CostPrice} + \\text{MarginFloor}$$

Under our assumed 40% margin structure with $\\text{MarginFloor} = 0$, all discounts up to 20% maintain positive unit margin (the minimum margin at 20% discount is $40\\% - 20\\% = 20\\%$). When tighter margin floors are configured in the application, higher discount arms are dynamically pruned.

### 4.2.3 Empty Set Fallback and Historical Violation Rate
If all promotional discounts are disqualified by guardrails, the policy falls back safely to $\\{0\\%\\}$ (full retail price). 

Auditing the raw historical Kaggle dataset against our loyalty tier rules reveals that **39.04% of historical records violate tier constraints** (e.g., historical sales showing 20% discounts granted to unverified or new transactions). This high violation rate highlights that historical retail discounting was uncoordinated and undisciplined, reinforcing why autonomous learning agents must be constrained by deterministic business rules.

## 4.3 Reward Formulations: Net Profit vs. Vanity Volume

The objective reward function is formulated explicitly around net realized profit:

$$R_t(a_t) = Y_t \\cdot \\left( \\text{DiscountedPrice}(a_t) - \\text{CostPrice} \\right) \\cdot \\text{BasketSize}$$

where $Y_t \\in \\{0, 1\\}$ is the realized conversion outcome. If the customer does not buy ($Y_t = 0$), profit is zero.

We explicitly distinguish between:
- **Gross Revenue:** Total customer cash spent: $Y_t \\cdot \\text{DiscountedPrice}(a_t) \\cdot \\text{BasketSize}$.
- **Discount Offered:** The nominal percentage price concession presented to the shopper ($a_t$).
- **Discount Given (Conceded):** The actual rupee value forfeited upon completed transaction: $Y_t \\cdot (\\text{BasePrice} \\times a_t) \\cdot \\text{BasketSize}$.

A policy that offers 20% discounts indiscriminately may achieve high conversion and high revenue while destroying net profit due to extreme discount concessions.

## 4.4 Policy Implementations

We benchmark five distinct decision policies:
1. **Random Baseline:** Selects uniformly at random from eligible actions: $a_t \\sim \\mathcal{U}(\\mathcal{A}_t)$.
2. **Fixed 10% Heuristic:** Standard retail promotional rule. Recommends 10% whenever eligible; if restricted (e.g., New tier capped at 5%), falls back to the maximum eligible discount (5%).
3. **Rule-Based Policy:** Tier-differentiated commercial heuristic reflecting standard retail practice:
   - New Tier $\\to$ Offer **0%**
   - Regular Tier $\\to$ Offer **10%**
   - Loyal Tier $\\to$ Offer **15%**
4. **LinUCB Policy:** Disjoint linear upper confidence bound policy with exploration parameter $\\alpha = 0.5$, context dimension $d = 20$, and Sherman-Morrison rank-1 updates.
5. **Thompson Sampling Policy:** Contextual Bayesian posterior sampling with observation variance multiplier $\\sigma = 0.5$ and spherical Gaussian priors.

## 4.5 Training Protocol and Experimental Settings

Table 4.1 summarizes the operational experimental parameters across all policies.

**Table 4.1: Master Experimental and Algorithmic Settings**

| Parameter | Configuration Value | Methodological Purpose / Source File |
| :--- | :---: | :--- |
| Action Space $\\mathcal{A}$ | $\\{0\\%, 5\\%, 10\\%, 15\\%, 20\\%\\}$ | 5 discrete promotional discount levels |
| Context Dimension ($d$) | 20 | Multivariate customer, product, and market state |
| Online Training Horizon | 20,000 rounds | Multi-period sequential learning horizon |
| Frozen Evaluation Horizon | 5,000 rounds | Post-training held-out validation horizon |
| LinUCB Exploration ($\\alpha$) | 0.5 | Tuned exploration confidence hyperparameter |
| Thompson Sampling Scale ($\\sigma$)| 0.5 | Posterior variance hyperparameter |
| Evaluation Seeds | 10 seeds (Seeds 42–51) | Paired multi-seed robustness testing |
| Hyperparameter Tuning Seeds | 5 seeds (Seeds 101–105)| Separate seed partition to avoid tuning bias |
| Assumed Margin Rate | 40% of Base Price | CostPrice model baseline |
| Customer Registry Size | 8,000 synthetic profiles | Persistent longitudinal RFM customer base |
| Hardware Environment | Multi-core x86_64 CPU | Standard laptop execution (no GPU required) |
| Wall-Clock Training Time | $\\approx 12 - 18$ seconds | 20,000 rounds per seed in Python/NumPy |

All 5 policies evaluate on identical, pre-generated round sequences with synchronized per-round random seeds, guaranteeing that observed performance divergences result entirely from policy choices rather than environmental stochasticity.
"""

with open("reports/sections/04_methodology.md", "w", encoding="utf-8") as f:
    f.write(methodology_text.strip() + "\n")

# -------------------------------------------------------------
# 05_evaluation_design.md
# -------------------------------------------------------------
eval_text = """# 5. Evaluation Design and Benchmark Methodology

## 5.1 Regret Formulation: Expected vs. Realized Regret

Algorithmic learning efficiency is evaluated using **cumulative pseudo-regret**. At round $t$, given context $x_t$ and eligible action set $\\mathcal{A}_t$, the instantaneous regret $\\Delta_t$ is defined as the expected profit gap between the oracle action $a_t^*$ and the policy's chosen action $a_t$:

$$\\Delta_t = \\mathbb{E}[R_t(a_t^*) | x_t] - \\mathbb{E}[R_t(a_t) | x_t] = \\mu_t^* - \\mu_t(a_t)$$

Cumulative regret over horizon $T$ is the running sum:

$$R(T) = \\sum_{t=1}^T \\Delta_t = \\sum_{t=1}^T \\left( \\mu_t^* - \\mu_t(a_t) \\right)$$

As established by Lattimore and Szepesvari [6], cumulative regret must be computed on **expected payoffs** rather than noisy realized payoffs $r_t$. Evaluating regret on realized stochastic purchase outcomes ($Y_t \\in \\{0, 1\\}$) injects extreme binomial variance into the evaluation metric: an optimal policy offering an optimal discount might experience an unluckily missed conversion, falsely registering regret. Expected regret isolates the quality of the algorithmic decision from underlying environmental stochasticity.

## 5.2 Dual Evaluation Protocol: Online vs. Frozen Modes

A common pitfall in reinforcement learning evaluation is conflating the cost of active learning with post-convergence policy quality. To prevent this, we execute a rigorous dual-mode evaluation protocol:

1. **Online Evaluation (20,000 rounds):** Simulates live production deployment from a cold start. The bandit begins with uninformative priors ($A_a = I_d, b_a = 0$), actively explores actions, suffers exploration regret, and updates its model parameters after every single round. This measures learning speed, sample efficiency, and cumulative exploration cost.
2. **Frozen Evaluation (5,000 held-out rounds):** Evaluates operational policy performance after training is complete. The model weights ($\hat{\\theta}_a, A_a^{-1}$) are locked. The agent selects actions greedily (or with frozen inference parameters) on a completely held-out sequence of 5,000 rounds without parameter updates. This measures the steady-state policy quality expected in production deployment.

## 5.3 Multi-Seed Experimental Protocol and Statistical Integrity

To eliminate seed selection bias and assess distributional stability, all experiments are replicated across **10 independent random seeds** (Seeds 42 through 51). 
- Every seed generates a distinct random sequence of customer arrivals, product inventory states, and competitor price fluctuations.
- For each seed, all five competing policies are executed on the **identical sequence of rounds**, enabling paired difference comparisons:
  $$\\Delta \\text{Profit}^{(s)} = \\text{Profit}_{\\text{Bandit}}^{(s)} - \\text{Profit}_{\\text{Baseline}}^{(s)}$$
- Performance is reported as sample mean $\\pm$ sample standard deviation across the 10 seeds, complemented by **win counts** (the number of seeds in which the bandit strictly outperformed the comparator).

To preserve methodological integrity, we intentionally **omit uncalibrated $p$-values and formal significance tests**. The 10 evaluation seeds are independent algorithmic replications on a single underlying simulator; quoting asymptotic hypothesis tests would imply generalization to real-world retail chains that the simulated setup cannot mathematically support.

## 5.4 The Financial Transaction Ledger

In commercial enterprise, machine learning metrics like cumulative regret and ROC-AUC are insufficient to convince financial stakeholders. We implement an immutable transaction-level **financial ledger** (`results/ledger_seed42.csv`) that logs every microeconomic event across each round:
- Round index $t$ and customer loyalty tier.
- Context features: base price, competitor price, stock level.
- Policy chosen arm $a_t$, eligible arm set $\\mathcal{A}_t$, and oracle arm $a_t^*$.
- Conversion outcome $Y_t$, basket units, and gross revenue.
- Net profit earned, discount offered, and discount conceded.
- Running cumulative totals for revenue, profit, and discount given.

The ledger is validated by hard accounting assertions: at every round, $\\text{GrossRevenue} = \\text{NetProfit} + \\text{CostOfGoodsSold} + \\text{DiscountGiven}$ must balance to zero rounding error.

## 5.5 Off-Policy Replay Evaluation Methodology

To test whether bandit policies can be evaluated on un-randomized observational data without simulator interaction, we implement the offline replay evaluation methodology of Li et al. [2].

### 5.5.1 The Rejection Sampling Replay Estimator
Let $\\mathcal{D} = \\{(x_1, a_1^{\\text{hist}}, r_1), \\dots, (x_N, a_N^{\\text{hist}}, r_N)\\}$ denote an offline stream of historical retail records. The replay evaluator streams through the records sequentially. For each row $k$:
1. The evaluated policy inspects context $x_k$ and recommends an action $a_k^{\\text{eval}}$.
2. **Rejection Check:** If $a_k^{\\text{eval}} == a_k^{\\text{hist}}$, the event is a **match**. The historical reward $r_k$ is accepted into the evaluation pool, and the policy's internal parameters are updated.
3. If $a_k^{\\text{eval}} \\ne a_k^{\\text{hist}}$, the event is rejected (dropped from evaluation), and no model update occurs.

Li et al. [2] proved that if historical actions $a^{\\text{hist}}$ were logged uniformly at random (or independently of context), the average reward of matched events provides an **unbiased estimate** of the policy's true performance.

### 5.5.2 Independence Diagnostic and Observational Caveats
In our public retail dataset, historical discounts were **not logged uniformly at random**; they were set by historical store managers responding to inventory and demand. We run an empirical independence diagnostic (evaluating historical discount frequency conditional on demand tertiles) which reveals significant non-uniformity. Furthermore, 39.04% of historical records violate loyalty tier guardrails and can never match a guardrailed policy. Consequently, off-policy replay in this study serves as a diagnostic sanity check rather than an unbiased causal evaluation.

## 5.6 Hyperparameter Sensitivity and Reward Ablation Design

1. **Exploration Sensitivity Sweep ($\\alpha$):** Evaluated across a 6-point grid $\\alpha \\in \\{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\\}$ using 5 separate tuning seeds (Seeds 101–105). This isolates the impact of uncertainty bonus scaling on early exploration vs. late exploitation.
2. **Reward Penalty Ablation ($\\lambda$):** To test whether explicit discount penalty terms alter bandit behavior, we formulate a regularized reward:
   $$R_t^{\\text{penalized}}(a_t) = R_t(a_t) - \\lambda \\cdot (\\text{BasePrice} \\times a_t)$$
   We sweep $\\lambda \\in \\{0.0, 0.007, 0.073, 0.365, 0.730\\}$, scaling the penalty from 0% to 100% of average unit profit, tracking policy collapse.
"""

with open("reports/sections/05_evaluation_design.md", "w", encoding="utf-8") as f:
    f.write(eval_text.strip() + "\n")

print("Sections 00 to 05 written successfully.")
