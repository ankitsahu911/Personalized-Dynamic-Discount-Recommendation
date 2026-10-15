# Personalized Dynamic Discount Recommendation Using Contextual Multi-Armed Bandits
**Final Project Technical Report**  
*Academic Project in Machine Learning, Reinforcement Learning & Pricing Optimization*  
*Author: Student Project Submission*  
*Date: October 2026*  

---


# Abstract

Promotional discounting in retail involves an inherent financial trade-off: price reductions stimulate sales volume but simultaneously erode unit profit margins. Traditional static heuristics and rigid category-level rules fail to account for heterogenous customer price sensitivities, while unconstrained machine learning approaches risk severe margin destruction. This project formulates dynamic discount recommendation as a contextual multi-armed bandit problem, deploying Linear Upper Confidence Bound (LinUCB) and contextual Thompson Sampling to personalize discount decisions across five discrete actions (0%, 5%, 10%, 15%, and 20%). Programmatic business guardrails—comprising customer loyalty tier discount ceilings and a strict unit cost margin floor—are integrated directly into the action selection loop, guaranteeing operational safety by construction. Evaluating policies over 10 independent random seeds across 20,000 simulated online rounds reveals that the heuristic benchmark Fixed 10% achieved the highest cumulative profit (₹80,265.41 ± 1,788.01), whereas LinUCB generated the highest gross revenue (₹234,276.84 ± 4,476.98) and outperformed the business Rule-based baseline in 8 of 10 seeds (+₹492.50 ± 1,022.62 profit lift). In frozen evaluation across 5,000 held-out rounds, policies preserved relative rankings. The principal limitation is that outcomes are derived within a semi-synthetic simulation environment calibrated against an aggregated public retail dataset lacking individual customer identifiers, with an assumed uniform cost structure.

---

# Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Motivation and Business Problem](#11-motivation-and-business-problem)
  - [1.2 The Exploration-Exploitation Dilemma in Dynamic Pricing](#12-the-exploration-exploitation-dilemma-in-dynamic-pricing)
  - [1.3 Project Scope and Core Objectives](#13-project-scope-and-core-objectives)
  - [1.4 Research Questions](#14-research-questions)
  - [1.5 Report Organization](#15-report-organization)
- [2. Background and Theoretical Foundations](#2-background-and-theoretical-foundations)
  - [2.1 Multi-Armed Bandits and Contextual Bandits](#21-multi-armed-bandits-and-contextual-bandits)
  - [2.2 The Linear Upper Confidence Bound (LinUCB) Algorithm](#22-the-linear-upper-confidence-bound-linucb-algorithm)
  - [2.3 Contextual Thompson Sampling](#23-contextual-thompson-sampling)
  - [2.4 Bandits vs. Deep Learning vs. Full Reinforcement Learning](#24-bandits-vs-deep-learning-vs-full-reinforcement-learning)
  - [2.5 Customer Segmentation and RFM Principles](#25-customer-segmentation-and-rfm-principles)
- [3. Data Pipeline and Feature Formulation](#3-data-pipeline-and-feature-formulation)
  - [3.1 Public Retail Dataset Overview](#31-public-retail-dataset-overview)
  - [3.2 Cleaning and Preprocessing Validation](#32-cleaning-and-preprocessing-validation)
  - [3.3 Derived Indicators and Feature Engineering](#33-derived-indicators-and-feature-engineering)
  - [3.4 Context Vector Formulation](#34-context-vector-formulation)
  - [3.5 What is Real and What is Simulated (Strict Boundary)](#35-what-is-real-and-what-is-simulated-strict-boundary)
- [4. Methodology and System Architecture](#4-methodology-and-system-architecture)
  - [4.1 The Simulation Environment and Demand Calibration](#41-the-simulation-environment-and-demand-calibration)
  - [4.2 Programmatic Business Guardrails](#42-programmatic-business-guardrails)
  - [4.3 Reward Formulations: Net Profit vs. Vanity Volume](#43-reward-formulations-net-profit-vs-vanity-volume)
  - [4.4 Policy Implementations](#44-policy-implementations)
  - [4.5 Training Protocol and Experimental Settings](#45-training-protocol-and-experimental-settings)
- [5. Evaluation Design and Benchmark Methodology](#5-evaluation-design-and-benchmark-methodology)
  - [5.1 Regret Formulation: Expected vs. Realized Regret](#51-regret-formulation-expected-vs-realized-regret)
  - [5.2 Dual Evaluation Protocol: Online vs. Frozen Modes](#52-dual-evaluation-protocol-online-vs-frozen-modes)
  - [5.3 Multi-Seed Experimental Protocol and Statistical Integrity](#53-multi-seed-experimental-protocol-and-statistical-integrity)
  - [5.4 The Financial Transaction Ledger](#54-the-financial-transaction-ledger)
  - [5.5 Off-Policy Replay Evaluation Methodology](#55-off-policy-replay-evaluation-methodology)
  - [5.6 Hyperparameter Sensitivity and Reward Ablation Design](#56-hyperparameter-sensitivity-and-reward-ablation-design)
- [6. Empirical Results and Performance Analysis](#6-empirical-results-and-performance-analysis)
  - [6.1 Headline Results and Policy Benchmarking](#61-headline-results-and-policy-benchmarking)
  - [6.2 Learning Dynamics and Convergence Analysis](#62-learning-dynamics-and-convergence-analysis)
  - [6.3 The Money View: Financial Ledger and Gain/Loss Dynamics](#63-the-money-view-financial-ledger-and-gain/loss-dynamics)
  - [6.4 Loyalty Tier Breakdown and Guardrail Audit](#64-loyalty-tier-breakdown-and-guardrail-audit)
  - [6.5 Off-Policy Replay Evaluation on Real Dataset](#65-off-policy-replay-evaluation-on-real-dataset)
  - [6.6 Sensitivity Analysis and Reward Ablation](#66-sensitivity-analysis-and-reward-ablation)
- [7. The Decision Support Application](#7-the-decision-support-application)
  - [7.1 Architecture and Design Principles](#71-architecture-and-design-principles)
  - [7.2 Application Interface Tour](#72-application-interface-tour)
  - [7.3 Explainability Framework](#73-explainability-framework)
  - [7.4 Profile Tier vs. Context Tier and Extrapolation Safeguards](#74-profile-tier-vs-context-tier-and-extrapolation-safeguards)
- [8. Discussion and Synthesis](#8-discussion-and-synthesis)
  - [8.1 Why the Heuristic Fixed 10% Benchmark Won on Profit](#81-why-the-heuristic-fixed-10%-benchmark-won-on-profit)
  - [8.2 LinUCB vs. Rule-Based Benchmarks](#82-linucb-vs-rule-based-benchmarks)
  - [8.3 The Commercial Necessity of Hard Guardrails](#83-the-commercial-necessity-of-hard-guardrails)
  - [8.4 Regret vs. Financial Ledgers: Dual Lenses on Performance](#84-regret-vs-financial-ledgers-dual-lenses-on-performance)
  - [8.5 Online Learning vs. Frozen Operational Modes](#85-online-learning-vs-frozen-operational-modes)
- [9. Limitations and Threats to Validity](#9-limitations-and-threats-to-validity)
  - [9.1 Data Limitations](#91-data-limitations)
  - [9.2 Simulation Limitations](#92-simulation-limitations)
  - [9.3 Evaluation Limitations](#93-evaluation-limitations)
  - [9.4 Modeling Assumptions](#94-modeling-assumptions)
  - [9.5 Practical Deployment Barriers](#95-practical-deployment-barriers)
- [10. Future Work](#10-future-work)
- [11. Conclusion](#11-conclusion)
- [12. References](#12-references)
- [13. Appendices](#13-appendices)
  - [Appendix A: Reproducibility Guide](#appendix-a-reproducibility-guide)
  - [Appendix B: Design-Decision Log](#appendix-b-design-decision-log)
  - [Appendix C: Glossary of Terminology and Notation](#appendix-c-glossary-of-terminology-and-notation)
  - [Appendix D: Complete Hyperparameters and Settings Table](#appendix-d-complete-hyperparameters-and-settings-table)
  - [Appendix E: 50-Milestone Completion Map](#appendix-e-50-milestone-completion-map)

---

# 1. Introduction

## 1.1 Motivation and Business Problem

In modern retail commerce, promotional discounting is among the most powerful yet dangerous operational levers available to merchandisers. When deployed effectively, promotional price cuts accelerate inventory turnover, stimulate customer acquisition, increase conversion rates, and prevent perishable or seasonal stock obsolescence. However, when applied indiscriminately through uniform blanket promotions (e.g., store-wide "20% off everything" campaigns), discounting severely erodes net unit margins, cannibalizes full-price revenue, and conditions price-sensitive consumers to defer purchases until promotions occur.

The central managerial challenge is that customer price responsiveness is highly heterogeneous. A highly committed customer purchasing an essential staple may convert at full retail price without any promotional concession, making a discount pure margin loss. Conversely, a price-sensitive prospect browsing a discretionary item may require a targeted price incentive to complete the transaction. Traditional retail systems address this problem with crude categorical heuristics (e.g., fixed markdowns per department) or static customer segmentation rules. These static approaches cannot adapt dynamically to multivariate customer contexts, real-time inventory pressures, or fluctuating competitor pricing.

## 1.2 The Exploration-Exploitation Dilemma in Dynamic Pricing

Standard supervised machine learning models attempt to predict purchase probability as a function of customer features and offered prices. However, applying supervised learning to pricing decisions creates an acute feedback loop problem: historical transaction logs reflect only the prices that past policies chose to offer. We do not observe counterfactual outcomes—that is, whether a customer who bought an item at a 20% discount would have bought it at 10% or at full price.

Attempting to resolve this via classical randomized controlled trials ($A/B$ testing) imposes substantial business opportunity costs. In an $A/B$ test, suboptimal pricing variants must be allocated fixed traffic shares for prolonged periods to establish statistical significance, directly forfeiting profit on underperforming arms. 

The multi-armed bandit (MAB) framework addresses this fundamental trade-off between **exploration** (testing discount levels to infer customer price elasticity) and **exploitation** (recommending the discount level that maximizes immediate expected return). Contextual bandits extend this paradigm by conditioning action selection on observable customer, product, and market covariates. Rather than learning a single global discount policy, a contextual bandit personalizes recommendations dynamically while actively bounding cumulative regret.

## 1.3 Project Scope and Core Objectives

This project designs, implements, evaluates, and deploys a complete contextual multi-armed bandit framework for personalized dynamic discount recommendation. The system addresses the business dilemma by:

1. **Formulating Action Spaces and Payoffs:** Discretizing promotional discounting into five actionable arms ($\{0\%, 5\%, 10\%, 15\%, 20\%\}$) and defining objective reward functions aligned with net profit margin rather than vanity revenue metrics.
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

---

# 2. Background and Theoretical Foundations

## 2.1 Multi-Armed Bandits and Contextual Bandits

The classic Multi-Armed Bandit (MAB) problem formalizes the fundamental tension between acquiring new information (exploration) and capitalizing on existing knowledge (exploitation). In a standard $K$-armed bandit, an agent repeatedly chooses among $K$ discrete actions $a \in \mathcal{A} = \{1, \dots, K\}$ across discrete time steps $t = 1, \dots, T$. Upon pulling arm $a_t$, the agent receives a stochastic reward $r_t \sim \mathcal{D}_{a_t}$ drawn from an unknown reward distribution associated with that arm. The objective is to maximize cumulative reward $\sum_{t=1}^T r_t$, or equivalently, to minimize cumulative pseudo-regret relative to an omniscient policy that consistently pulls the single best arm:

$$R(T) = T \cdot \mu^* - \sum_{t=1}^T \mathbb{E}[r_{t, a_t}]$$

where $\mu^* = \max_{a \in \mathcal{A}} \mathbb{E}_{r \sim \mathcal{D}_a}[r]$.

While classic MAB algorithms (such as $\epsilon$-greedy or standard UCB1) excel in stationary environments where all users are identical, they cannot personalize recommendations. In retail discounting, a customer's propensity to buy depends heavily on observable features such as past purchase history, brand affinity, current inventory levels, and external price gaps.

The **Contextual Multi-Armed Bandit** framework extends the classic setting by incorporating side information. At each round $t$:
1. The environment presents an observable context vector $x_t \in \mathbb{R}^d$ characterizing the customer, item, and market state.
2. The agent selects an action $a_t \in \mathcal{A}_t$, where $\mathcal{A}_t \subseteq \mathcal{A}$ denotes the set of legally permissible actions under business constraints.
3. The environment reveals a payoff $r_t(a_t) \in \mathbb{R}$ drawn from a conditional distribution $\mathcal{D}(r | x_t, a_t)$. Payoffs for unselected counterfactual arms $a \ne a_t$ remain unobserved.
4. The agent updates its internal parameter estimates using the transition tuple $(x_t, a_t, r_t(a_t))$.

## 2.2 The Linear Upper Confidence Bound (LinUCB) Algorithm

The Linear Upper Confidence Bound algorithm, introduced by Li et al. [1], assumes a linear relationship between the context vector $x_t$ and the expected payoff of each arm $a$:

$$\mathbb{E}[r_{t, a} | x_t] = x_t^T \theta_a^*$$

where $\theta_a^* \in \mathbb{R}^d$ is an unknown, fixed coefficient vector specific to arm $a$.

### 2.2.1 Parameter Estimation via Ridge Regression
For each arm $a$, the agent maintains ridge regression estimates based on its observed history $\mathcal{H}_a = \{(x_\tau, r_\tau) : a_\tau = a\}$. Let $D_a \in \mathbb{R}^{m_a \times d}$ denote the matrix whose rows are the context vectors observed when arm $a$ was pulled, and let $c_a \in \mathbb{R}^{m_a}$ denote the corresponding vector of observed rewards. The regularized ridge regression estimate $\hat{\theta}_a$ is given by:

$$\hat{\theta}_a = \left( D_a^T D_a + I_d \right)^{-1} D_a^T c_a = A_a^{-1} b_a$$

where $A_a = D_a^T D_a + I_d \in \mathbb{R}^{d \times d}$ is the regularized design covariance matrix, and $b_a = D_a^T c_a \in \mathbb{R}^d$ accumulates context-reward interactions. The identity matrix $I_d$ acts as an $L_2$ regularizer ensuring that $A_a$ is strictly positive definite and invertible at all rounds.

### 2.2.2 Confidence Bounds and Action Selection
Applying matrix concentration inequalities, Li et al. [1] proved that with high probability, the estimation error is bounded by:

$$\left| x_t^T \hat{\theta}_a - x_t^T \theta_a^* \right| \le \alpha \sqrt{x_t^T A_a^{-1} x_t}$$

where $\alpha = 1 + \sqrt{\ln(2 / \delta) / 2} > 0$ controls the confidence hyperparameter. The term $\sqrt{x_t^T A_a^{-1} x_t}$ represents the standard deviation of the ridge estimator in direction $x_t$, serving as an explicit measure of model uncertainty.

At round $t$, LinUCB computes an upper confidence bound score $p_{t, a}$ for every eligible action $a \in \mathcal{A}_t$:

$$p_{t, a} = \underbrace{x_t^T \hat{\theta}_a}_{\text{Exploitation: Expected Payoff}} + \underbrace{\alpha \sqrt{x_t^T A_a^{-1} x_t}}_{\text{Exploration: Uncertainty Bonus}}$$

The policy selects the arm that maximizes this optimistic bound:

$$a_t = \arg\max_{a \in \mathcal{A}_t} p_{t, a}$$

Upon observing reward $r_t$, the sufficient statistics are updated incrementally:

$$A_{a_t} \leftarrow A_{a_t} + x_t x_t^T, \quad b_{a_t} \leftarrow b_{a_t} + r_t x_t$$

### 2.2.3 Sherman-Morrison Incremental Matrix Inversion
In a naive implementation, evaluating the uncertainty bonus requires inverting $A_a \in \mathbb{R}^{d \times d}$ at every round, incurring an $\mathcal{O}(d^3)$ computational cost per update. To achieve production efficiency, we implement the rank-1 Sherman-Morrison update directly on the inverse covariance matrix $M_a = A_a^{-1}$:

$$M_{a_t} \leftarrow M_{a_t} - \frac{M_{a_t} x_t x_t^T M_{a_t}}{1 + x_t^T M_{a_t} x_t}$$

This formulation reduces the computational complexity of updates from $\mathcal{O}(d^3)$ to $\mathcal{O}(d^2)$, enabling high-throughput real-time inference without loss of numerical precision.

## 2.3 Contextual Thompson Sampling

Thompson Sampling, originally proposed by Thompson [3] and analyzed for contextual linear payoffs by Agrawal and Goyal [4] and Chapelle and Li [5], operates under a Bayesian framework. Rather than constructing deterministic upper confidence bounds, Thompson Sampling maintains a posterior probability distribution over the model parameters and samples actions according to their posterior probability of being optimal.

### 2.3.1 Likelihood and Prior
Assuming Gaussian observation noise $r_t(a) \sim \mathcal{N}(x_t^T \theta_a^*, v^2)$ with prior $\theta_a \sim \mathcal{N}(0, v^2 I_d)$, the posterior distribution of $\theta_a$ at round $t$ remains conjugate Gaussian:

$$\theta_a \sim \mathcal{N}\left( \hat{\theta}_a, v^2 B_a^{-1} \right)$$

where $B_a = I_d + \sum_{\tau: a_\tau = a} x_\tau x_\tau^T$ and $\hat{\theta}_a = B_a^{-1} \sum_{\tau: a_\tau = a} r_\tau x_\tau$.

### 2.3.2 Posterior Sampling Decision Rule
At round $t$, for each eligible arm $a \in \mathcal{A}_t$, the algorithm draws a randomized parameter vector $\tilde{\theta}_a$ from its current posterior:

$$\tilde{\theta}_a \sim \mathcal{N}\left( \hat{\theta}_a, v^2 B_a^{-1} \right)$$

The algorithm then executes the greedy action with respect to the sampled parameter vectors:

$$a_t = \arg\max_{a \in \mathcal{A}_t} x_t^T \tilde{\theta}_a$$

Upon observing reward $r_t$, the sufficient statistics $B_{a_t}$ and the target vector are updated identically to LinUCB. Thompson Sampling naturally balances exploration and exploitation: arms with high parameter variance exhibit wider posterior distributions, resulting in higher sampling dispersion and occasional selection, while well-estimated arms converge to deterministic exploitation.

## 2.4 Bandits vs. Deep Learning vs. Full Reinforcement Learning

To establish the methodological boundaries of this project, it is essential to distinguish contextual bandits from full Reinforcement Learning (RL) and Deep Learning:

1. **Contextual Bandits vs. Full Reinforcement Learning:** As defined by Sutton and Barto [7], full RL operates in a Markov Decision Process (MDP) where action $a_t$ influences both the immediate reward $r_t$ and the future state distribution $s_{t+1} \sim P(s_{t+1} | s_t, a_t)$. Full RL models long-term sequential planning using value iteration, Q-learning, or policy gradients. In contextual bandits, the environment produces an exogenous context $x_t$ at each round that is conditionally independent of past actions. Because the discount offered to customer $A$ at 10:00 AM does not alter the feature vector of unrelated customer $B$ at 10:05 AM, the state-transition dynamics collapse, and the problem reduces to single-step associative reward maximization. This structural simplification drastically improves sample efficiency, eliminating the instability and convergence challenges of deep RL.
2. **Linear Bandits vs. Deep Contextual Bandits:** Non-linear contextual bandits (such as NeuralUCB [8]) substitute linear ridge regression with deep neural networks to approximate non-linear payoff surfaces. However, deep bandits require frequent gradient updates, introduce non-convex loss surfaces, suffer from catastrophic forgetting, and require orders of magnitude more samples to form well-calibrated epistemic confidence bounds. For operational retail discount recommendation, linear contextual bandits provide strict mathematical guarantees, closed-form updates, microsecond latency, and interpretable coefficient weightings.

## 2.5 Customer Segmentation and RFM Principles

Customer segmentation in this project is grounded in the classical Recency, Frequency, Monetary (RFM) marketing framework introduced by Hughes [9]:
- **Recency ($R$):** Elapsed days since the customer's last purchase. Customers with recent engagement exhibit higher baseline conversion propensities.
- **Frequency ($F$):** Total count of completed transactions across the customer's lifetime. High-frequency purchasers indicate established loyalty.
- **Monetary ($M$):** Cumulative spending or historical average basket value.

In commercial practice, loyalty programs partition customers into categorical tiers (e.g., New, Regular, Loyal) based on RFM thresholds. From an economic perspective, granting identical discounts across tiers is suboptimal. Offering steep discounts to highly loyal, price-insensitive shoppers forfeits margin without driving incremental volume (deadweight margin loss). Conversely, denying promotional incentives to price-sensitive prospects stifles acquisition. Integrating RFM-derived loyalty tiers into the contextual bandit allows the model to personalize promotions while honoring tiered business policies.

---

# 3. Data Pipeline and Feature Formulation

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
3. **Discount Discretization:** Historical recorded discounts were mapped to their nearest nominal evaluation arm: $\{0\%, 5\%, 10\%, 15\%, 20\%\}$.

## 3.3 Derived Indicators and Feature Engineering

To provide rich contextual signals for the bandit policies, several domain-specific features were engineered from the raw table:

- **Demand Indicator (Tertile Split):** A categorical demand classification was derived by computing category-specific tertile boundaries on `Historical_Demand`. Products are partitioned into `Low`, `Medium`, and `High` demand bands.
- **Discount Amount & Discounted Price:**
  $$\text{DiscountAmount} = \text{BasePrice} \times a_t$$
  $$\text{DiscountedPrice} = \text{BasePrice} - \text{DiscountAmount}$$
- **Price Gap Percentage:** Relative price positioning versus local competition:
  $$\text{PriceGapPct} = \frac{\text{BasePrice} - \text{CompetitorPrice}}{\text{CompetitorPrice}}$$
  A positive gap indicates that the store is more expensive than the competitor, elevating customer price sensitivity.
- **Demand-to-Inventory Ratio & Stock Coverage Days:**
  $$\text{DemandToInventoryRatio} = \frac{\text{HistoricalDemand}}{\max(1, \text{CurrentStock})}$$
  $$\text{StockCoverageDays} = \frac{\text{CurrentStock}}{\max(1, \text{HistoricalDemand})}$$
- **Inventory Status:** Categorical binning of stock coverage into `Critical Low`, `Balanced`, and `Overstocked`.
- **Cost Price and Unit Margin:** Because real supplier procurement invoices are proprietary and absent from the public dataset, unit cost is modeled under an **assumed 40% margin rate**:
  $$\text{CostPrice} = \text{BasePrice} \times (1 - 0.40) = 0.60 \times \text{BasePrice}$$
  $$\text{UnitMargin} = \text{DiscountedPrice} - \text{CostPrice} = \text{BasePrice} \times (0.40 - a_t)$$
  Under this structure, offering a 20% discount halves the percentage margin from 40% to 20%, illustrating the steep financial cost of excessive discounting.
- **Temporal Calendar Features:** Extracted `DayOfWeek`, `Month`, and binary `IsWeekend` indicators to capture cyclical shopping variations.

## 3.4 Context Vector Formulation

At each decision round $t$, the bandit observes a 20-dimensional numerical context vector $x_t \in \mathbb{R}^{20}$. Table 3.1 details the constituent features, their provenance, and their preprocessing transformations.

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
| $x^{(11)}$ | `CustomerTier_New` | Indicator: customer lifetime visits $\le 2$ | Simulated | Binary $\{0, 1\}$ |
| $x^{(12)}$ | `CustomerTier_Regular` | Indicator: customer lifetime visits $3 - 9$ | Simulated | Binary $\{0, 1\}$ |
| $x^{(13)}$ | `CustomerTier_Loyal` | Indicator: customer lifetime visits $\ge 10$ | Simulated | Binary $\{0, 1\}$ |
| $x^{(14)}$ | `DemandLevel_Low` | Category demand tertile: Low | Derived | Binary $\{0, 1\}$ |
| $x^{(15)}$ | `DemandLevel_Med` | Category demand tertile: Medium | Derived | Binary $\{0, 1\}$ |
| $x^{(16)}$ | `DemandLevel_High` | Category demand tertile: High | Derived | Binary $\{0, 1\}$ |
| $x^{(17)}$ | `IsWeekend` | Saturday or Sunday indicator | Derived | Binary $\{0, 1\}$ |
| $x^{(18)}$ | `Month_sin` | Cyclical month representation | Derived | Continuous ($[-1, 1]$) |
| $x^{(19)}$ | `Month_cos` | Cyclical month representation | Derived | Continuous ($[-1, 1]$) |
| $x^{(20)}$ | `Bias_Intercept` | Constant unit intercept | Architecture | Constant ($1.0$) |

Continuous numerical features are standardized to zero mean and unit variance using scaling parameters fitted strictly on training observations, preventing data leakage.

## 3.5 What is Real and What is Simulated (Strict Boundary)

To maintain absolute scientific integrity and avoid misrepresenting results, we establish an explicit, unequivocal boundary between real empirical data and simulated components:

1. **The Public Retail Dataset:** The historical dataset consists of 73,100 store/product/day rows from Kaggle [10]. It contains **no individual customer IDs**, no longitudinal user shopping carts, and no counterfactual conversion outcomes at alternative prices. Per project guidelines, this dataset is designated strictly as a "public retail dataset" pending author verification of Kaggle provenance.
2. **Simulated Customer Registry:** The 8,000 synthetic customer profiles—including their persistent customer IDs, assigned loyalty tiers (`New`, `Regular`, `Loyal`), historical purchase counts, recency days, and latent price sensitivity values—are **simulated by design**.
3. **Simulated Basket Value and Purchase Outcomes:** Individual customer shopping sessions, transaction basket sizes (units per purchase), and the binary conversion outcome $Y_t \in \{0, 1\}$ are **simulated by design** via the calibrated simulator.
4. **Assumed Cost Structure:** CostPrice is derived under an **assumed 40% margin rate**. No real wholesale supplier cost data was present in the public dataset.
5. **Simulated Currency:** All financial metrics (revenue, profit, discount conceded) reported in rupees (₹) are **simulated rupees** generated within the simulation environment.

---

# 4. Methodology and System Architecture

## 4.1 The Simulation Environment and Demand Calibration

Because contextual bandits require interactive feedback loops and counterfactual observation, offline training requires an interactive simulation environment (`simulator.py`).

### 4.1.1 The Purchase Probability Formulation
When presented with context $x_t$ and an offered discount arm $a_t \in \{0.0, 0.05, 0.10, 0.15, 0.20\}$, the probability that the customer completes a purchase is modeled via a logistic sigmoid link:

$$P(Y_t = 1 | x_t, a_t) = \sigma\left( z(x_t, a_t) \right) = \frac{1}{1 + e^{-z(x_t, a_t)}}$$

where the latent log-odds logit $z(x_t, a_t)$ is structured as:

$$z(x_t, a_t) = \beta_0 + \beta_{\text{disc}} \cdot a_t + \beta_{\text{sens}} \cdot \text{PriceSensitivity}_t + \beta_{\text{gap}} \cdot \text{PriceGapPct}_t + \beta_{\text{stock}} \cdot \text{StockRatio}_t + \beta_{\text{tier}} \cdot \mathbb{I}_{\text{tier}_t}$$

- **Baseline Intercept ($\beta_0$):** Anchors the intrinsic conversion rate for products priced at parity with zero discount.
- **Discount Stimulus ($\beta_{\text{disc}} > 0$):** Quantifies the promotional conversion lift per percentage point of discount.
- **Price Sensitivity Coupling ($\beta_{\text{sens}} < 0$):** Modulates the baseline reluctance of price-sensitive shoppers, interacting with offered discounts.
- **Competitor Gap Impact ($\beta_{\text{gap}} < 0$):** Suppresses purchase probability when store price exceeds competitor shelf price.
- **Inventory Urgency ($\beta_{\text{stock}} > 0$):** Slightly elevates conversion when perceived scarcity or high velocity occurs.

### 4.1.2 Empirical Calibration via Logistic Regression
To prevent arbitrary behavioral assumptions, the coefficient vector $\beta$ was calibrated against historical demand classifications in the retail dataset using a logistic regression model. The empirical calibration yielded a Receiver Operating Characteristic Area Under the Curve (**ROC-AUC**) of **0.6322**. 

An AUC of 0.6322 indicates a modest, directionally valid predictive signal over random chance ($0.50$), consistent with noisy aggregated retail demand. Crucially, a modest AUC demonstrates that the simulator does not embed deterministic or trivial shortcuts; bandit policies must actively learn subtle multivariate trade-offs under substantial residual variance.

### 4.1.3 The 8,000-Customer Synthetic Registry and Counterfactuals
The environment instantiates a persistent registry of 8,000 synthetic customers partitioned into three RFM loyalty tiers:
- **New Customers (25% share):** Lifetime purchases $\le 2$, high recency days ($> 60$ days), high price sensitivity (mean $\mu = 0.75$).
- **Regular Customers (50% share):** Lifetime purchases $3 - 9$, moderate recency ($15 - 60$ days), moderate sensitivity (mean $\mu = 0.50$).
- **Loyal Customers (25% share):** Lifetime purchases $\ge 10$, low recency ($< 15$ days), low price sensitivity (mean $\mu = 0.25$).

At each round $t$, the simulator evaluates the true purchase probability across **all five counterfactual discount arms** simultaneously. This allows exact determination of the **Oracle action** $a_t^*$:

$$a_t^* = \arg\max_{a \in \mathcal{A}_t} \mathbb{E}[R_t(a) | x_t] = \arg\max_{a \in \mathcal{A}_t} P(Y_t = 1 | x_t, a) \cdot \text{UnitMargin}(a) \cdot \text{BasketSize}$$

## 4.2 Programmatic Business Guardrails

In commercial retail, unconstrained algorithmic optimization is unacceptable. A pure profit-maximizing bandit operating on price-sensitive customers might learn to offer predatory markdowns that violate supplier minimum advertised price (MAP) agreements, degrade brand equity, or trigger below-cost sales. We introduce a hard programmatic guardrail layer (`guardrails.py`) that filters the available arm set $\mathcal{A}$ to an eligible subset $\mathcal{A}_t = \text{get\_eligible\_actions}(x_t) \subseteq \mathcal{A}$.

### 4.2.1 Loyalty Tier Ceilings
Retailers reserve their steepest promotions for their most valuable customers to prevent customer attrition and cultivate retention. The tier guardrail enforces strict promotional discount caps:
- **New Tier:** Maximum permissible discount = **5%** (Eligible arms: $\{0\%, 5\%\}$).
- **Regular Tier:** Maximum permissible discount = **15%** (Eligible arms: $\{0\%, 5\%, 10\%, 15\%\}$).
- **Loyal Tier:** Maximum permissible discount = **20%** (Eligible arms: $\{0\%, 5\%, 10\%, 15\%, 20\%\}$).

### 4.2.2 Unit Cost Margin Floor
To protect gross profitability, no discount may reduce unit selling price below cost plus an essential margin buffer:

$$\text{DiscountedPrice}(a) \ge \text{CostPrice} + \text{MarginFloor}$$

Under our assumed 40% margin structure with $\text{MarginFloor} = 0$, all discounts up to 20% maintain positive unit margin (the minimum margin at 20% discount is $40\% - 20\% = 20\%$). When tighter margin floors are configured in the application, higher discount arms are dynamically pruned.

### 4.2.3 Empty Set Fallback and Historical Violation Rate
If all promotional discounts are disqualified by guardrails, the policy falls back safely to $\{0\%\}$ (full retail price). 

Auditing the raw historical Kaggle dataset against our loyalty tier rules reveals that **39.04% of historical records violate tier constraints** (e.g., historical sales showing 20% discounts granted to unverified or new transactions). This high violation rate highlights that historical retail discounting was uncoordinated and undisciplined, reinforcing why autonomous learning agents must be constrained by deterministic business rules.

## 4.3 Reward Formulations: Net Profit vs. Vanity Volume

The objective reward function is formulated explicitly around net realized profit:

$$R_t(a_t) = Y_t \cdot \left( \text{DiscountedPrice}(a_t) - \text{CostPrice} \right) \cdot \text{BasketSize}$$

where $Y_t \in \{0, 1\}$ is the realized conversion outcome. If the customer does not buy ($Y_t = 0$), profit is zero.

We explicitly distinguish between:
- **Gross Revenue:** Total customer cash spent: $Y_t \cdot \text{DiscountedPrice}(a_t) \cdot \text{BasketSize}$.
- **Discount Offered:** The nominal percentage price concession presented to the shopper ($a_t$).
- **Discount Given (Conceded):** The actual rupee value forfeited upon completed transaction: $Y_t \cdot (\text{BasePrice} \times a_t) \cdot \text{BasketSize}$.

A policy that offers 20% discounts indiscriminately may achieve high conversion and high revenue while destroying net profit due to extreme discount concessions.

## 4.4 Policy Implementations

We benchmark five distinct decision policies:
1. **Random Baseline:** Selects uniformly at random from eligible actions: $a_t \sim \mathcal{U}(\mathcal{A}_t)$.
2. **Fixed 10% Heuristic:** Standard retail promotional rule. Recommends 10% whenever eligible; if restricted (e.g., New tier capped at 5%), falls back to the maximum eligible discount (5%).
3. **Rule-Based Policy:** Tier-differentiated commercial heuristic reflecting standard retail practice:
   - New Tier $\to$ Offer **0%**
   - Regular Tier $\to$ Offer **10%**
   - Loyal Tier $\to$ Offer **15%**
4. **LinUCB Policy:** Disjoint linear upper confidence bound policy with exploration parameter $\alpha = 0.5$, context dimension $d = 20$, and Sherman-Morrison rank-1 updates.
5. **Thompson Sampling Policy:** Contextual Bayesian posterior sampling with observation variance multiplier $\sigma = 0.5$ and spherical Gaussian priors.

## 4.5 Training Protocol and Experimental Settings

Table 4.1 summarizes the operational experimental parameters across all policies.

**Table 4.1: Master Experimental and Algorithmic Settings**

| Parameter | Configuration Value | Methodological Purpose / Source File |
| :--- | :---: | :--- |
| Action Space $\mathcal{A}$ | $\{0\%, 5\%, 10\%, 15\%, 20\%\}$ | 5 discrete promotional discount levels |
| Context Dimension ($d$) | 20 | Multivariate customer, product, and market state |
| Online Training Horizon | 20,000 rounds | Multi-period sequential learning horizon |
| Frozen Evaluation Horizon | 5,000 rounds | Post-training held-out validation horizon |
| LinUCB Exploration ($\alpha$) | 0.5 | Tuned exploration confidence hyperparameter |
| Thompson Sampling Scale ($\sigma$)| 0.5 | Posterior variance hyperparameter |
| Evaluation Seeds | 10 seeds (Seeds 42–51) | Paired multi-seed robustness testing |
| Hyperparameter Tuning Seeds | 5 seeds (Seeds 101–105)| Separate seed partition to avoid tuning bias |
| Assumed Margin Rate | 40% of Base Price | CostPrice model baseline |
| Customer Registry Size | 8,000 synthetic profiles | Persistent longitudinal RFM customer base |
| Hardware Environment | Multi-core x86_64 CPU | Standard laptop execution (no GPU required) |
| Wall-Clock Training Time | $\approx 12 - 18$ seconds | 20,000 rounds per seed in Python/NumPy |

All 5 policies evaluate on identical, pre-generated round sequences with synchronized per-round random seeds, guaranteeing that observed performance divergences result entirely from policy choices rather than environmental stochasticity.

---

# 5. Evaluation Design and Benchmark Methodology

## 5.1 Regret Formulation: Expected vs. Realized Regret

Algorithmic learning efficiency is evaluated using **cumulative pseudo-regret**. At round $t$, given context $x_t$ and eligible action set $\mathcal{A}_t$, the instantaneous regret $\Delta_t$ is defined as the expected profit gap between the oracle action $a_t^*$ and the policy's chosen action $a_t$:

$$\Delta_t = \mathbb{E}[R_t(a_t^*) | x_t] - \mathbb{E}[R_t(a_t) | x_t] = \mu_t^* - \mu_t(a_t)$$

Cumulative regret over horizon $T$ is the running sum:

$$R(T) = \sum_{t=1}^T \Delta_t = \sum_{t=1}^T \left( \mu_t^* - \mu_t(a_t) \right)$$

As established by Lattimore and Szepesvari [6], cumulative regret must be computed on **expected payoffs** rather than noisy realized payoffs $r_t$. Evaluating regret on realized stochastic purchase outcomes ($Y_t \in \{0, 1\}$) injects extreme binomial variance into the evaluation metric: an optimal policy offering an optimal discount might experience an unluckily missed conversion, falsely registering regret. Expected regret isolates the quality of the algorithmic decision from underlying environmental stochasticity.

## 5.2 Dual Evaluation Protocol: Online vs. Frozen Modes

A common pitfall in reinforcement learning evaluation is conflating the cost of active learning with post-convergence policy quality. To prevent this, we execute a rigorous dual-mode evaluation protocol:

1. **Online Evaluation (20,000 rounds):** Simulates live production deployment from a cold start. The bandit begins with uninformative priors ($A_a = I_d, b_a = 0$), actively explores actions, suffers exploration regret, and updates its model parameters after every single round. This measures learning speed, sample efficiency, and cumulative exploration cost.
2. **Frozen Evaluation (5,000 held-out rounds):** Evaluates operational policy performance after training is complete. The model weights ($\hat{\theta}_a, A_a^{-1}$) are locked. The agent selects actions greedily (or with frozen inference parameters) on a completely held-out sequence of 5,000 rounds without parameter updates. This measures the steady-state policy quality expected in production deployment.

## 5.3 Multi-Seed Experimental Protocol and Statistical Integrity

To eliminate seed selection bias and assess distributional stability, all experiments are replicated across **10 independent random seeds** (Seeds 42 through 51). 
- Every seed generates a distinct random sequence of customer arrivals, product inventory states, and competitor price fluctuations.
- For each seed, all five competing policies are executed on the **identical sequence of rounds**, enabling paired difference comparisons:
  $$\Delta \text{Profit}^{(s)} = \text{Profit}_{\text{Bandit}}^{(s)} - \text{Profit}_{\text{Baseline}}^{(s)}$$
- Performance is reported as sample mean $\pm$ sample standard deviation across the 10 seeds, complemented by **win counts** (the number of seeds in which the bandit strictly outperformed the comparator).

To preserve methodological integrity, we intentionally **omit uncalibrated $p$-values and formal significance tests**. The 10 evaluation seeds are independent algorithmic replications on a single underlying simulator; quoting asymptotic hypothesis tests would imply generalization to real-world retail chains that the simulated setup cannot mathematically support.

## 5.4 The Financial Transaction Ledger

In commercial enterprise, machine learning metrics like cumulative regret and ROC-AUC are insufficient to convince financial stakeholders. We implement an immutable transaction-level **financial ledger** (`results/ledger_seed42.csv`) that logs every microeconomic event across each round:
- Round index $t$ and customer loyalty tier.
- Context features: base price, competitor price, stock level.
- Policy chosen arm $a_t$, eligible arm set $\mathcal{A}_t$, and oracle arm $a_t^*$.
- Conversion outcome $Y_t$, basket units, and gross revenue.
- Net profit earned, discount offered, and discount conceded.
- Running cumulative totals for revenue, profit, and discount given.

The ledger is validated by hard accounting assertions: at every round, $\text{GrossRevenue} = \text{NetProfit} + \text{CostOfGoodsSold} + \text{DiscountGiven}$ must balance to zero rounding error.

## 5.5 Off-Policy Replay Evaluation Methodology

To test whether bandit policies can be evaluated on un-randomized observational data without simulator interaction, we implement the offline replay evaluation methodology of Li et al. [2].

### 5.5.1 The Rejection Sampling Replay Estimator
Let $\mathcal{D} = \{(x_1, a_1^{\text{hist}}, r_1), \dots, (x_N, a_N^{\text{hist}}, r_N)\}$ denote an offline stream of historical retail records. The replay evaluator streams through the records sequentially. For each row $k$:
1. The evaluated policy inspects context $x_k$ and recommends an action $a_k^{\text{eval}}$.
2. **Rejection Check:** If $a_k^{\text{eval}} == a_k^{\text{hist}}$, the event is a **match**. The historical reward $r_k$ is accepted into the evaluation pool, and the policy's internal parameters are updated.
3. If $a_k^{\text{eval}} \ne a_k^{\text{hist}}$, the event is rejected (dropped from evaluation), and no model update occurs.

Li et al. [2] proved that if historical actions $a^{\text{hist}}$ were logged uniformly at random (or independently of context), the average reward of matched events provides an **unbiased estimate** of the policy's true performance.

### 5.5.2 Independence Diagnostic and Observational Caveats
In our public retail dataset, historical discounts were **not logged uniformly at random**; they were set by historical store managers responding to inventory and demand. We run an empirical independence diagnostic (evaluating historical discount frequency conditional on demand tertiles) which reveals significant non-uniformity. Furthermore, 39.04% of historical records violate loyalty tier guardrails and can never match a guardrailed policy. Consequently, off-policy replay in this study serves as a diagnostic sanity check rather than an unbiased causal evaluation.

## 5.6 Hyperparameter Sensitivity and Reward Ablation Design

1. **Exploration Sensitivity Sweep ($\alpha$):** Evaluated across a 6-point grid $\alpha \in \{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\}$ using 5 separate tuning seeds (Seeds 101–105). This isolates the impact of uncertainty bonus scaling on early exploration vs. late exploitation.
2. **Reward Penalty Ablation ($\lambda$):** To test whether explicit discount penalty terms alter bandit behavior, we formulate a regularized reward:
   $$R_t^{\text{penalized}}(a_t) = R_t(a_t) - \lambda \cdot (\text{BasePrice} \times a_t)$$
   We sweep $\lambda \in \{0.0, 0.007, 0.073, 0.365, 0.730\}$, scaling the penalty from 0% to 100% of average unit profit, tracking policy collapse.

---

# 6. Empirical Results and Performance Analysis

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

---

# 7. The Decision Support Application

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

---

# 8. Discussion and Synthesis

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

---

# 9. Limitations and Threats to Validity

To ensure rigorous academic integrity, this section provides an unvarnished audit of the structural, empirical, and operational limitations of this study.

## 9.1 Data Limitations

1. **Absence of Customer Identifiers:** The underlying Kaggle dataset [10] comprises 73,100 store/product/day aggregated observational logs. It contains **no individual customer IDs**, no persistent cookie trackers, and no longitudinal shopping baskets. True individual personalization could not be learned from the raw table; individual customer profiles were instantiated via simulation.
2. **Dataset Provenance:** The dataset is designated strictly as a "public retail dataset" pending author confirmation of Kaggle provenance. If the source data was synthetically generated by its original author, statistical patterns reflect synthetic generator assumptions rather than genuine consumer habits.
3. **Assumed Cost Price Structure:** Wholesale procurement costs are proprietary and absent from the public data. We modeled unit cost under an **assumed uniform 40% margin rate** ($\text{CostPrice} = 0.60 \times \text{BasePrice}$). In physical retail, gross margins vary drastically across categories (e.g., 5% on milk vs. 60% on apparel). Distortions in this assumed margin directly alter calculated profit payoffs.

## 9.2 Simulation Limitations

1. **Synthetic Registry and Customer Behavior:** The 8,000-customer registry, customer loyalty tier assignments, recency days, lifetime visits, latent price sensitivities, transaction basket sizes, and conversion events were **simulated by design**.
2. **Modest Demand Calibration AUC (0.6322):** The purchase-probability formula is a stylized logistic sigmoid model calibrated via logistic regression against historical demand tertiles, yielding an empirical ROC-AUC of **0.6322**. While this confirms a modest predictive relationship, an AUC of 0.6322 indicates substantial unexplained variance. Simulated consumer choices represent stylized approximations of retail elasticity rather than exact human behavior.
3. **Internal Alignment Bias (Shared Simulator):** Both policy training and policy evaluation were conducted within the **identical simulation environment**. Because the simulator's generative rules remained stationary between training and evaluation, policies were evaluated under the exact behavioral assumptions they were trained on, flattering model convergence relative to real-world deployment.

## 9.3 Evaluation Limitations

1. **Violation of Replay Independence Condition:** Off-policy replay evaluation [2] mathematically requires that historical logging actions were chosen uniformly at random or independently of context. In the historical Kaggle data, discounts were set deliberately by human store managers, inducing confounding.
2. **Severe Rejection Bias in Historical Logs (39.04% Violation Rate):** **39.04% of historical records violate loyalty tier guardrails**. Because our guardrailed policies strictly respect tier ceilings, historical rows with non-compliant discounts could never match, dropping them from evaluation. Replay results reflect only the subset of historical situations where past human actions happened to coincide with guardrailed recommendations.
3. **Overlapping Confidence Intervals on Replay:** Replay per-round profit estimates (LinUCB ₹6.11 [₹5.84, ₹6.38] vs. Fixed 10% ₹5.87 [₹5.61, ₹6.13]) exhibit overlapping 95% confidence intervals. No statistically valid claim of real-world superiority can be asserted.
4. **Sample Size of Ten Seeds:** Experiments were evaluated across 10 random seeds. While sufficient to report means, standard deviations, and win counts, this sample size does not support formal asymptotic hypothesis testing.

## 9.4 Modeling Assumptions

1. **Linear Payoff on Non-Linear Demand:** LinUCB and linear Thompson Sampling assume a linear relationship $\mathbb{E}[R|x] = x^T \theta$, whereas the simulator generates purchases via a logistic sigmoid link. Because a linear model cannot fit a sigmoidal response surface across 20 dimensions, cumulative regret flattens to a non-zero constant slope ($\approx ₹0.05 - ₹0.09/\text{round}$) rather than converging asymptotically to zero.
2. **Single-Period Myopic Horizon:** Bandits treat each round as an isolated transaction. The model ignores long-term customer lifetime value (LTV), brand equity degradation, and customer conditioning (where shoppers learn to wait for discounts). Loyalty preservation was delegated to the deterministic tier guardrail rather than learned by the model.
3. **Offline Training Only:** The production deployment architecture evaluates pre-trained frozen models. Streaming online parameter updates were not implemented in the application layer.

## 9.5 Practical Deployment Barriers

1. **Store-Wide Promotional Budgets:** Physical retail operations operate under aggregate monthly promotional spend caps. The bandit optimizes per-transaction profit independently without tracking global budget burn rates.
2. **Competitor Dynamic Reactions:** The simulator assumes competitor prices fluctuate exogenously. In actual retail markets, competing chains dynamically undercut aggressive promotional discounting, potentially igniting destructive price wars.
3. **Regulatory and Fairness Audits:** Offering personalized prices or discounts based on customer profiling raises consumer protection and legal scrutiny. Retailers must submit dynamic pricing algorithms to formal fairness audits to ensure compliance with consumer protection laws.

---

# 10. Future Work

This section outlines technical and methodological extensions identified in the project specification to advance beyond the current implementation. *None of these extensions was implemented in the current codebase unless explicitly documented in project files.*

1. **Non-Linear Deep Contextual Bandits (NeuralUCB):** Implementing NeuralUCB [8] using deep neural network feature extractors would allow the bandit to approximate complex, non-linear sigmoidal demand surfaces directly, eliminating the persistent linear approximation error observed in regret curves.
2. **Streaming Online Model Updates:** Engineering an asynchronous streaming feedback pipeline (e.g., via Apache Kafka) would enable the bandit to continuously update its covariance matrices in real time as live retail point-of-sale transactions complete.
3. **Evaluation on Identity-Bearing Retail Datasets:** Transitioning from the aggregated Kaggle dataset to an identity-bearing transactional corpus (such as the UCI Online Retail dataset, which contains authentic customer IDs and multi-item invoices) would enable genuine empirical evaluation of longitudinal customer RFM dynamics without synthetic registry simulation.
4. **Reward-Formulated Loyalty Policies with Convergence Guarantees ("Option 2"):** Formulating customer retention value directly into the reward function—supported by theoretical regret proofs—would enable the agent to learn loyalty stewardship organically rather than relying entirely on hard guardrails.
5. **Hybrid Linear Contextual Bandits:** Deploying Hybrid LinUCB [1] with shared feature representations across arms would allow the model to pool customer-level parameters while maintaining arm-specific coefficients for product pricing attributes, accelerating early-stage convergence.
6. **Competitor-Reactive Dynamic Game Pricing:** Integrating multi-agent game-theoretic feedback loops where competitor prices respond endogenously to the bandit's promotional actions would simulate realistic oligopolistic retail competition.
7. **Production Microservice Architecture (FastAPI & PostgreSQL):** Packaging the bandit inference engine into a production-grade REST API using FastAPI, backed by a persistent PostgreSQL database for ledger logging and Redis for sub-millisecond context caching.

---

# 11. Conclusion

This project developed, evaluated, and deployed a contextual multi-armed bandit system for personalized dynamic discount recommendation in retail commerce, integrating linear decision policies (LinUCB and Thompson Sampling) with deterministic commercial guardrails. Across 10 independent random seeds evaluated over 20,000 online rounds, the heuristic benchmark **Fixed 10% earned the highest cumulative profit** (**₹80,265.41 ± 1,788.01**), outperforming LinUCB (**₹78,924.82 ± 788.32**, a -1.67% deficit) and Thompson Sampling (**₹78,856.21 ± 1,027.65**, a -1.76% deficit). However, LinUCB successfully generated the **highest gross revenue** (**₹234,276.84 ± 4,476.98**) by identifying price-sensitive opportunities and offering higher promotional discounts (8.45% vs. 8.09% for Fixed 10%), lifting conversion rates to **24.29% ± 0.58%** while outperforming the enterprise Rule-based baseline on profit (**+₹492.50 ± 1,022.62**, winning in 8 of 10 seeds). In frozen evaluation across 5,000 held-out rounds, relative policy hierarchies remained consistent. Programmatic guardrails achieved **100% compliance** across 200,000 online decisions, completely preventing below-cost transactions and tier breaches. Off-policy replay on 73,100 historical retail records yielded overlapping 95% confidence intervals across all policies (LinUCB ₹6.11 vs. Fixed 10% ₹5.87), demonstrating that historical observational logs cannot establish real-world policy superiority. The fundamental caveat is that all outcomes were derived **within a semi-synthetic simulation environment** calibrated against an aggregated public retail dataset lacking individual customer identifiers, with an assumed uniform 40% margin structure. Dynamic bandits offer genuine capabilities for revenue expansion, but their deployment demands disciplined guardrails and rigorous margin tracking.

---

# 12. References

[1] L. Li, W. Chu, J. Langford, and R. E. Schapire, "A contextual-bandit approach to personalized news article recommendation," in *Proceedings of the 19th International Conference on World Wide Web (WWW)*, 2010, pp. 661–670.

[2] L. Li, W. Chu, J. Langford, and X. Wang, "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms," in *Proceedings of the 4th ACM International Conference on Web Search and Data Mining (WSDM)*, 2011, pp. 297–306.

[3] W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, vol. 25, no. 3–4, pp. 285–294, 1933.

[4] S. Agrawal and N. Goyal, "Thompson Sampling for contextual bandits with linear payoffs," in *Proceedings of the 30th International Conference on Machine Learning (ICML)*, 2013, pp. 127–135.

[5] O. Chapelle and L. Li, "An empirical evaluation of Thompson Sampling," in *Advances in Neural Information Processing Systems (NeurIPS 24)*, 2011, pp. 2249–2257.

[6] T. Lattimore and C. Szepesvári, *Bandit Algorithms*. Cambridge, UK: Cambridge University Press, 2020.

[7] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.

[8] D. Zhou, L. Li, and Q. Gu, "Neural contextual bandits with UCB-based exploration," in *Proceedings of the 37th International Conference on Machine Learning (ICML)*, 2020, pp. 11492–11502.

[9] A. M. Hughes, *Strategic Database Marketing: The Masterplan for Starting and Managing a Profitable, Customer-Based Marketing Program*. Chicago, IL: Probus Publishing, 1994.

[10] Kaggle Retail Dataset: Retail Store Inventory and Demand Forecasting Dataset, Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets. [AUTHOR TO FILL IN: exact title, author, URL and access date of the corrected "Retail Store Inventory and Demand Forecasting" dataset and of the original by Anirudh Chauhan].

---

**NOTICE FOR AUTHOR: VERIFY EVERY REFERENCE AGAINST THE REAL SOURCE BEFORE SUBMITTING.**

---

# 13. Appendices

## Appendix A: Reproducibility Guide

### A.1 Hardware and Environment Specifications
All experiments, training runs, and evaluations were executed in the following local environment:
- **Operating System:** Microsoft Windows 11 Home / Pro (x86_64 architecture).
- **Python Runtime:** Python 3.14.3 (tags/v3.14.3, 64-bit AMD64).
- **Primary Package Dependencies:**
  - `numpy==2.2.3`
  - `pandas==2.2.3`
  - `scipy==1.15.2`
  - `scikit-learn==1.6.1`
  - `matplotlib==3.10.1`
  - `streamlit==1.42.0`
  - `plotly==6.0.0`
  - `pytest==8.3.4`
  - `python-pptx==1.0.2`
- **Hardware Resources:** Standard multi-core laptop CPU (no dedicated GPU acceleration was required or utilized).

### A.2 Execution Commands
All results, models, and application interfaces can be fully reproduced from the repository root:

```bash
# 1. Install required Python dependencies
pip install -r requirements-app.txt
pip install python-pptx

# 2. Re-run complete test suite (20 automated tests)
python -m pytest app/tests -v

# 3. Launch the operational Streamlit decision application
streamlit run app/streamlit_app.py

# 4. Re-generate comparison plots and slide figures
python scripts/generate_slide_figures.py

# 5. Build presentation slide deck (.pptx)
python scripts/build_slide_deck.py
```

## Appendix B: Design-Decision Log

**Table B.1: Architectural Design Decisions, Rationales, and Source Files**

| Decision ID | Plan Proposal | As-Built Implementation | Rationale and Context | Source File Reference |
| :---: | :--- | :--- | :--- | :--- |
| **DD-01** | Global random sequence per run | Synchronized per-round random seeds | Eliminates environmental variance between competing policies; all policies face identical customer arrivals. | `src/simulation/generator.py` |
| **DD-02** | Single online training run | Dual-mode evaluation (Online & Frozen) | Distinguishes the cost of exploratory learning (20k rounds) from mature operational policy quality (5k held-out rounds). | `src/evaluation/evaluator.py` |
| **DD-03** | Tune on evaluation seeds | Separate tuning seeds (101–105) | Prevents hyperparameter over-fitting on reported evaluation seeds (Seeds 42–51). | `results/alpha_sensitivity.csv` |
| **DD-04** | Soft penalty reward for loyalty | Hard programmatic guardrails | Soft penalties collapse discount volume globally; hard rules guarantee 100% policy compliance without distorting bandit learning. | `src/guardrails/rules.py` |
| **DD-05** | Real customer purchase logs | Synthetic 8,000-customer registry | Public Kaggle retail dataset lacks individual customer IDs; longitudinal RFM profiles required simulation. | `src/simulation/registry.py` |
| **DD-06** | Naive matrix inversion $\mathcal{O}(d^3)$ | Sherman-Morrison rank-1 update $\mathcal{O}(d^2)$ | Enables sub-millisecond inference and high-throughput training in pure Python/NumPy. | `src/policies/linucb.py` |
| **DD-07** | Empty set failure crash | Fallback to $\{0\%\}$ full price | Guarantees operational fault tolerance if commercial margin floors eliminate all promotional arms. | `src/guardrails/rules.py` |
| **DD-08** | Combined tier input in app | Decoupled Profile Tier vs. Context Tier | Preserves transparency when merchandisers test edge-case RFM inputs while inspecting specific tier rule envelopes. | `app/streamlit_app.py` |
| **DD-09** | Sole focus on net profit | Paired tracking of Profit, Revenue, and Conceded Discount | Exposes the critical business trade-off where bandits capture highest revenue by trading away net margin. | `src/evaluation/ledger.py` |

## Appendix C: Glossary of Terminology and Notation

- **Round ($t$):** A single atomic decision instance consisting of an observable customer context $x_t$, a product inventory state, a recommended discount arm $a_t$, and a stochastic conversion outcome $Y_t$.
- **Policy ($\pi$):** A decision-making mapping $\pi: \mathcal{X} \to \mathcal{A}$ that selects an action based on context.
- **Arm ($a$):** A discrete promotional discount choice from the action set $\mathcal{A} = \{0\%, 5\%, 10\%, 15\%, 20\%\}$.
- **Eligible Set ($\mathcal{A}_t$):** The subset of actions $\mathcal{A}_t \subseteq \mathcal{A}$ permitted after filtering context $x_t$ through loyalty tier caps and margin floor guardrails.
- **Guardrail:** A deterministic commercial rule enforced prior to algorithmic scoring that prevents illegal or below-cost recommendations.
- **Oracle ($a_t^*$):** The counterfactual arm within the eligible set $\mathcal{A}_t$ that maximizes true mathematical expected net profit under simulator ground-truth probabilities.
- **Expected Regret ($\Delta_t$):** The difference in expected profit between the oracle action and the policy's chosen action: $\Delta_t = \mu_t^* - \mu_t(a_t)$.
- **Cumulative Regret ($R(T)$):** The running summation of instantaneous expected regret: $R(T) = \sum_{t=1}^T \Delta_t$.
- **Financial Ledger:** An immutable transaction log recording per-round revenues, unit costs, realized profits, and discount concessions.
- **Discount Offered:** The nominal percentage discount presented to the consumer ($a_t$).
- **Discount Given (Conceded):** The actual currency value surrendered upon completed transaction: $Y_t \cdot (\text{BasePrice} \times a_t) \cdot \text{BasketSize}$.
- **Online Evaluation:** Protocol where the bandit begins from a cold start, actively explores, and updates model parameters after every transaction.
- **Frozen Evaluation:** Protocol where pre-trained model weights are locked, evaluating steady-state operational performance without updates on held-out rounds.
- **Off-Policy Replay:** Evaluation method using historical transaction logs via rejection sampling (Li et al. [2]).
- **Simulated Rupees (₹):** Simulated monetary units generated within the retail simulation environment.

## Appendix D: Complete Hyperparameters and Settings Table

**Table D.1: Master Algorithmic and Experimental Hyperparameters**

| Category | Hyperparameter Name | Symbol | Value | Source File |
| :--- | :--- | :---: | :---: | :--- |
| **Bandit** | Action Space | $\mathcal{A}$ | $\{0\%, 5\%, 10\%, 15\%, 20\%\}$ | `src/policies/base.py` |
| **Bandit** | Context Dimension | $d$ | 20 | `models/linucb_v1_meta.json` |
| **Bandit** | LinUCB Exploration Parameter | $\alpha$ | 0.5 | `models/linucb_v1_meta.json` |
| **Bandit** | Ridge Regularization Factor | $\lambda_2$ | 1.0 ($I_d$) | `src/policies/linucb.py` |
| **Bandit** | Thompson Sampling Noise Multiplier | $\sigma$ | 0.5 | `models/thompson_v1_meta.json` |
| **Simulation**| Online Horizon | $T_{\text{online}}$ | 20,000 rounds | `models/linucb_v1_meta.json` |
| **Simulation**| Frozen Evaluation Horizon | $T_{\text{frozen}}$ | 5,000 rounds | `src/evaluation/evaluator.py` |
| **Simulation**| Evaluation Random Seeds | $\mathcal{S}_{\text{eval}}$ | Seeds 42 through 51 | `results/comparison_online.csv` |
| **Simulation**| Hyperparameter Tuning Seeds | $\mathcal{S}_{\text{tune}}$ | Seeds 101 through 105 | `results/alpha_sensitivity.csv` |
| **Simulation**| Synthetic Customer Registry | $N_{\text{cust}}$ | 8,000 profiles | `src/simulation/registry.py` |
| **Simulation**| Assumed Cost Margin Rate | $m_{\text{cost}}$ | 40% of Base Price | `src/simulation/simulator.py` |
| **Simulation**| Demand Logistic Calibration AUC | $\text{AUC}$ | 0.6322 | `reports/evaluation_findings.md` |
| **Guardrails**| New Tier Discount Cap | $a_{\max}^{\text{New}}$ | 5% | `src/guardrails/rules.py` |
| **Guardrails**| Regular Tier Discount Cap | $a_{\max}^{\text{Reg}}$ | 15% | `src/guardrails/rules.py` |
| **Guardrails**| Loyal Tier Discount Cap | $a_{\max}^{\text{Loyal}}$ | 20% | `src/guardrails/rules.py` |
| **Guardrails**| Default Unit Margin Floor Buffer | $\Delta_{\text{floor}}$ | ₹0.00 | `src/guardrails/rules.py` |

## Appendix E: 50-Milestone Completion Map

**Table E.1: Complete Roadmap Completion Record (Phases 1 through 9)**

| Phase | Milestone Range | Primary Objective | Key Output Artifact | Status |
| :---: | :---: | :--- | :--- | :---: |
| **Phase 1** | M01 – M08 | Data foundation, cleaning, feature engineering, registry | `data/processed_retail.csv`, `src/simulation/registry.py` | **Complete** |
| **Phase 2** | M09 – M14 | Demand calibration, simulator core, counterfactuals, oracle | `src/simulation/simulator.py`, calibration logs | **Complete** |
| **Phase 3** | M15 – M19 | Programmatic guardrails, margin floor, historical audit | `src/guardrails/rules.py`, `reports/fairness_audit.json` | **Complete** |
| **Phase 4** | M20 – M22 | Baseline policies (Random, Fixed 10%, Rule-based) | `src/policies/baselines.py` | **Complete** |
| **Phase 5** | M23 – M28 | Bandit implementations (LinUCB, Thompson Sampling, Sherman-Morrison) | `src/policies/linucb.py`, `src/policies/thompson.py` | **Complete** |
| **Phase 6** | M29 – M31 | 20,000-round training, model serialization, logging | `models/linucb_v1.npz`, `models/thompson_v1.npz` | **Complete** |
| **Phase 7** | M32 – M40 | Multi-seed evaluation, regret, ledger, replay, sensitivity, ablation | `results/*.csv`, `outputs/plots/*.png`, `reports/evaluation_findings.md` | **Complete** |
| **Phase 8** | M41 – M47 | Frontend Streamlit app, test suite, dashboard UI | `app/streamlit_app.py`, `app/tests/test_app.py` (20/20 pass) | **Complete** |
| **Phase 9** | M48 – M50 | Final report, claims audit, slides & charts, viva preparation | `reports/final_report.md`, `slides/deck.pptx`, `viva/fact_sheet.md` | **Complete** |

---
