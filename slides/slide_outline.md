# Presentation Slide Outline

*Rules: 16 core slides + 6 backup slides. Every slide outline contains $\le 45$ words.*

---

### Slide 1: Title Slide
- **Title:** Personalized Dynamic Discount Recommendation
- **Subtitle:** Contextual Multi-Armed Bandits with Commercial Guardrails
- **Author:** Final Year Project Submission
- **Focus:** Reinforcement learning, pricing optimization, and margin preservation.

---

### Slide 2: The Retail Dilemma
- **The Trade-Off:** Discounts stimulate unit sales but erode profit margins.
- **Static Rules Fail:** Blanket 20% markdowns sacrifice margin on loyal shoppers.
- **The Objective:** Personalize discounts to price sensitivity while guaranteeing profitability.

---

### Slide 3: Why Contextual Bandits?
- **Beyond A/B Testing:** Bandits minimize opportunity cost during live exploration.
- **Context-Aware:** Recommends across 5 actions ($\{0\%, 5\%, 10\%, 15\%, 20\%\}$) using customer/product state.
- **Single-Step Efficiency:** Exogenous customer arrivals avoid complex multi-period state transitions.

---

### Slide 4: Algorithmic Formulations
- **LinUCB:** Ridge regression payoff estimate plus optimistic uncertainty bonus $\alpha \sqrt{x_t^T A_a^{-1} x_t}$.
- **Sherman-Morrison:** Updates inverse covariance directly in $\mathcal{O}(d^2)$ rather than $\mathcal{O}(d^3)$.
- **Thompson Sampling:** Bayesian posterior sampling over linear parameters ($\\sigma = 0.5$).

---

### Slide 5: The Commercial Safety Envelope
- **Tier Ceilings:** New (max 5%), Regular (max 15%), Loyal (max 20%).
- **Margin Floor:** $\text{DiscountedPrice} \ge \text{CostPrice} + \text{MarginFloor}$.
- **Inviolable Filtering:** Hard programmatic pruning prevents below-cost sales before model scoring.

---

### Slide 6: Data Pipeline & Strict Boundary
- **Public Retail Data:** 73,100 store/product/day logs from Kaggle.
- **Simulated Components:** 8,000-customer registry, RFM loyalty tiers, shopping baskets, conversion outcomes.
- **Cost Structure:** Modeled under an assumed 40% gross margin rate.

---

### Slide 7: Demand Calibration & Simulator
- **Purchase Probability:** Sigmoidal demand calibrated via logistic regression (ROC-AUC 0.6322).
- **Realistic Noise:** Modest AUC confirms natural retail variance without synthetic shortcuts.
- **Oracle Baseline:** Evaluates counterfactual expected profit across all 5 discount arms.

---

### Slide 8: Evaluation Methodology
- **Dual Modes:** Online (20,000 rounds learning) vs. Frozen (5,000 held-out rounds).
- **10 Seeds:** Paired comparisons across identical round sequences (Seeds 42–51).
- **Financial Ledger:** Complete microeconomic tracking of revenue, margin, and discount conceded.

---

### Slide 9: Headline Result: The Profit Paradox
- **Profit Winner:** Heuristic Fixed 10% earned ₹80,265.41 ± 1,788.01.
- **Bandit Deficit:** LinUCB achieved ₹78,924.82 (-1.67%); Thompson achieved ₹78,856.21 (-1.76%).
- **Elasticity Sweet Spot:** Flat 10% maximized conversion without triggering steep margin penalties.

---

### Slide 10: Headline Result: Revenue Leadership
- **Top-Line Winner:** LinUCB generated highest gross revenue (₹234,276.84 ± 4,476.98).
- **Conversion Lift:** LinUCB achieved 24.29% conversion vs. 23.81% for Fixed 10%.
- **Lift vs. Rule-based:** LinUCB beat Rule-based on profit (+₹492.50, won 8/10 seeds).

---

### Slide 11: Learning & Regret Convergence
- **Sublinear Profile:** Regret flattens from ₹0.229/round to ₹0.092/round.
- **Non-Zero Asymptote:** Irreducible gap persists because linear models cannot fit sigmoidal demand.
- **Final Regret (Seed 42):** LinUCB ₹1,754.14; Thompson Sampling ₹1,075.28.

---

### Slide 12: Financial Ledger Dynamics
- **Exploration Tax:** LinUCB trailed Rule-based during initial 1,500 rounds.
- **Break-Even Crossover:** Surpassed Rule-based at Round 3,200 on Seed 42.
- **Margin Concession:** LinUCB conceded ₹24,642 in discounts vs. ₹20,955 for Fixed 10%.

---

### Slide 13: Tier Breakdown & Guardrail Audit
- **100% Compliance:** Zero margin floor breaches across 200,000 decisions.
- **Tier Utilization:** LinUCB averaged 3.11% on New, 9.84% on Regular, 11.02% on Loyal.
- **Historical Contrast:** 39.04% of historical Kaggle records violated tier rules.

---

### Slide 14: Off-Policy Replay Sanity Check
- **73,100 Records:** Replay match rates averaged ~19.8% to 20.5%.
- **Replay Profit:** LinUCB ₹6.11/round vs. Fixed 10% ₹5.87/round.
- **No Causal Claim:** Overlapping 95% confidence intervals and non-random logging prevent superiority claims.

---

### Slide 15: Operational Streamlit Dashboard
- **Live Decision Tool:** Modular frontend decoupled from frozen inference backend.
- **Explainability:** Decomposes UCB score into mean payoff and uncertainty bonus.
- **100% Offline:** Zero external network calls; sub-millisecond local NumPy execution.

---

### Slide 16: Conclusions & Takeaways
- **Key Finding:** Bandits excel at revenue expansion; simple heuristics can win on profit.
- **Guardrails Required:** Hard constraints are mandatory to prevent margin destruction.
- **Commercial Reality:** Evaluated within calibrated simulator; real stores require global spend budgets.

---

## Backup Slides (6)

---

### Slide 17: Hyperparameter Sensitivity ($\alpha$)
- **Exploration Sweep:** Tested $\alpha \in \{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\}$.
- **Sweet Spot:** $\alpha \in [0.1, 1.0]$ delivers robust convergence; $\alpha = 0.5$ optimal.
- **Tuning Integrity:** Evaluated on separate tuning seeds (Seeds 101–105).

---

### Slide 18: Reward Penalty Ablation ($\lambda$)
- **Penalty Formulation:** Tested regularized reward $R - \lambda \cdot (\text{BasePrice} \times a)$.
- **Policy Collapse:** Higher $\lambda$ throttled discounts (4.86%), destroying revenue (-₹20k).
- **Validation:** Proves hard guardrails outperform soft reward penalties.

---

### Slide 19: Comprehensive Performance Tables
- **Online & Frozen:** Full side-by-side metrics across all 5 policies.
- **Consistent Hierarchy:** Fixed 10% leads profit; LinUCB leads revenue across both modes.

---

### Slide 20: Mathematical Derivations
- **Ridge Solution:** $\hat{\theta}_a = (D_a^T D_a + I_d)^{-1} D_a^T c_a = A_a^{-1} b_a$.
- **Sherman-Morrison:** $M \leftarrow M - \frac{M x x^T M}{1 + x^T M x}$, reducing complexity to $\mathcal{O}(d^2)$.

---

### Slide 21: Full Threats to Validity
- **Data:** No customer IDs; public retail logs; assumed 40% margin.
- **Simulation:** Calibrated at AUC 0.6322; shared environment for training/evaluation.
- **Evaluation:** Replay non-randomness; single retail simulator.

---

### Slide 22: Technical Future Scope
- **NeuralUCB:** Deep non-linear contextual bandits for sigmoidal elasticity.
- **Streaming Pipeline:** Continuous real-time updates via streaming architecture.
- **Real Identifiers:** Validation on longitudinal transaction data (UCI Online Retail).
