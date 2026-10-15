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
