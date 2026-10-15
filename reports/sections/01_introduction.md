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
