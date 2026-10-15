# Section-by-Section Viva Questions and Report Pointers

*This question bank maps potential examiner questions directly to the corresponding sections and paragraphs of reports/final_report.md.*

## Section 1: Introduction
1. **Q: Why is promotional discounting formulated as a bandit problem instead of supervised regression?**  
   *Pointer:* Report §1.2 ("The Exploration-Exploitation Dilemma in Dynamic Pricing"). Supervised models only observe historical pricing outcomes and suffer from selection bias; they cannot explore counterfactual price elasticities.
2. **Q: What is the financial risk of unconstrained price experimentation in physical retail?**  
   *Pointer:* Report §1.1 ("Motivation and Business Problem"). Unconstrained discounting erodes net unit margins, cannibalizes full-price transactions, and conditions consumers to wait for sales.
3. **Q: What are the four research questions this project answers?**  
   *Pointer:* Report §1.4 ("Research Questions": RQ1 Convergence, RQ2 Financial Lift, RQ3 Guardrail Safety, RQ4 Replay Viability).

## Section 2: Background and Theoretical Foundations
4. **Q: Walk me through the mathematical derivation of the LinUCB upper confidence bound.**  
   *Pointer:* Report §2.2.2. The UCB score decomposes into ridge regression mean payoff $x_t^T \hat{	heta}_a$ plus uncertainty bonus $lpha \sqrt{x_t^T A_a^{-1} x_t}$ derived from matrix concentration bounds.
5. **Q: How does Sherman-Morrison rank-1 update improve computational performance?**  
   *Pointer:* Report §2.2.3. It updates the inverse covariance matrix directly in $\mathcal{O}(d^2)$ rather than re-inverting $A_a$ from scratch at $\mathcal{O}(d^3)$ per round.
6. **Q: Why use a contextual bandit rather than full Reinforcement Learning?**  
   *Pointer:* Report §2.4. Customer transaction events are exogenous; customer arrivals do not exhibit Markovian state transitions ($s_{t+1} \sim P(s_t, a_t)$), making bandits far more sample-efficient.

## Section 3: Data Pipeline and Feature Formulation
7. **Q: What part of your dataset is real and what part is simulated?**  
   *Pointer:* Report §3.5 ("What is Real and What is Simulated"). Store/product/day logs are from Kaggle (73,100 rows); customer registry, baskets, conversion outcomes, and 40% margin cost structure are simulated by design.
8. **Q: What does the PriceGapPct feature represent and why is it important?**  
   *Pointer:* Report §3.3. Relative difference vs. competitor shelf price; positive gaps increase price sensitivity.

## Section 4: Methodology and System Architecture
9. **Q: How was the purchase probability formula calibrated and what does the AUC of 0.6322 imply?**  
   *Pointer:* Report §4.1.2. Calibrated via logistic regression on historical demand tertiles. Modest AUC reflects realistic retail noise without artificial shortcuts.
10. **Q: Why enforce loyalty tiers as hard guardrails instead of soft reward penalties?**  
    *Pointer:* Report §4.2 and §8.3. Soft penalties cause global discount collapse; hard programmatic filtering achieves 100% compliance while preserving optimization freedom.
11. **Q: What did the 39.04% historical tier violation rate reveal?**  
    *Pointer:* Report §4.2.3. Historical retail discounts were uncoordinated, proving why automated guardrails are necessary.

## Section 5: Evaluation Design
12. **Q: Why must cumulative regret be computed on expected profit rather than realized profit?**  
    *Pointer:* Report §5.1. Realized binary purchase outcomes ($Y_t \in \{0, 1\}$) introduce severe binomial noise; expected regret isolates algorithmic policy quality.
13. **Q: What is the difference between Online and Frozen evaluation modes?**  
    *Pointer:* Report §5.2. Online measures in-flight learning and exploration cost; Frozen measures post-training operational policy quality on held-out rounds.

## Section 6: Results
14. **Q: Why did the heuristic Fixed 10% beat both LinUCB and Thompson Sampling on profit?**  
    *Pointer:* Report §6.1 and §8.1. Fixed 10% hit the empirical elasticity sweet spot; bandits offered 15-20% discounts to price-sensitive customers, lifting revenue to ₹234k but sacrificing unit net margin.
15. **Q: Why does the regret curve flatten to a non-zero slope rather than zero?**  
    *Pointer:* Report §6.2. A linear model ($\mathbb{E}[R|x] = x^T 	heta$) cannot perfectly replicate a multi-dimensional logistic sigmoid demand surface.
16. **Q: Did off-policy replay prove your model works on real-world retail data?**  
    *Pointer:* Report §6.5. No; overlapping confidence intervals, non-random historical logging, and a 39% tier violation rate make replay a sanity check, not a causal proof.

## Section 7 & 8: Application and Discussion
17. **Q: Does your Streamlit web application learn from user interactions?**  
    *Pointer:* Report §7.1. No; the app operates in frozen forward inference mode using pre-trained weights to ensure deterministic recommendations.
18. **Q: How does the app explain recommendations to a store manager?**  
    *Pointer:* Report §7.3. Decomposes LinUCB score into expected mean payoff and uncertainty bonus, with clear text blocking reasons for disqualified arms.
