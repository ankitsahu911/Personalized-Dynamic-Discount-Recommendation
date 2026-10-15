# Speaker Notes and Oral Presentation Script

*Spoken delivery: approximately 60–90 seconds per slide. Natural spoken style with exact numbers, transitions, and examiner defense notes.*

---

### Slide 1: Title Slide (Time: 0:45)
"Good morning, members of the committee. Today I am presenting my final-year project: *Personalized Dynamic Discount Recommendation Using Contextual Multi-Armed Bandits with Commercial Guardrails*.

Promotional discounting is one of the most critical operational levers in retail, but when executed poorly, it destroys millions in margin. In this work, I formulate discount recommendation as a contextual bandit, bound it with hard business guardrails, and evaluate it across twenty thousand simulated rounds and seventy-three thousand historical retail records. Let's examine the commercial problem."

*Transition:* "To understand why bandits are necessary, consider how physical retailers currently discount."

---

### Slide 2: The Retail Dilemma (Time: 1:15)
"In retail, every price promotion involves a delicate trade-off. Price cuts stimulate sales volume, clear seasonal inventory, and acquire customers. But every percentage point conceded directly reduces unit margin. 

Most retailers rely on blunt, category-wide heuristics—like 'twenty percent off apparel.' This is financially wasteful. A loyal shopper buying an everyday essential will gladly convert at full price; giving them twenty percent off is deadweight margin loss. Meanwhile, a price-sensitive prospect might abandon their cart without a five percent incentive. Our goal is to personalize discounts dynamically to customer and market context while guaranteeing that no discount ever causes a financial loss."

*Transition:* "Traditional machine learning struggles with this problem. Here is why we chose contextual bandits."

---

### Slide 3: Why Contextual Bandits? (Time: 1:15)
"Standard supervised learning only tells us what happened at the historical prices we offered—it cannot tell us counterfactuals. Classical A/B testing can, but it forces retailers to allocate static traffic to failing price points for weeks, suffering high opportunity costs.

Contextual bandits solve this exploration-exploitation dilemma. At each transaction, the bandit observes a twenty-dimensional context—customer loyalty, competitor price gap, inventory levels—and selects among five discrete discount arms: zero, five, ten, fifteen, or twenty percent. Unlike full reinforcement learning, retail transactions are exogenous: discounting item A this morning does not alter the feature vector of unrelated customer B this afternoon. This single-step formulation eliminates MDP complexity and achieves high sample efficiency."

*Transition:* "Now let's examine the specific bandit algorithms we implemented."

---

### Slide 4: Algorithmic Formulations (Time: 1:30)
"We implemented two distinct contextual policies: Linear Upper Confidence Bound, or LinUCB, and Bayesian contextual Thompson Sampling.

LinUCB models expected payoff as a linear function of context. For each arm, it computes an optimistic score consisting of the expected payoff plus an exploration bonus: alpha times the standard deviation of our estimate. The hyperparameter alpha governs the exploration appetite. Crucially, a naive LinUCB inverts the covariance matrix at every round in O of d-cubed time. We implemented the rank-one Sherman-Morrison update directly on the inverse covariance matrix, reducing update complexity to O of d-squared.

In parallel, we implemented contextual Thompson Sampling with Gaussian priors, which samples parameters from the posterior distribution and executes the greedy action, naturally balancing exploration through parameter dispersion."

*Transition:* "However, letting an algorithm explore prices unconstrained is commercial suicide. That brings us to our safety guardrails."

---

### Slide 5: The Commercial Safety Envelope (Time: 1:15)
"In physical retail, an autonomous learning agent might discover that offering twenty percent discounts to price-sensitive customers maximizes immediate conversions. But doing so degrades brand equity and risks selling below cost.

We engineered a hard, programmatic safety envelope. First, customer loyalty tier ceilings: New customers can receive at most five percent, Regular customers fifteen percent, and Loyal customers twenty percent. Second, a unit margin floor: discounted price must exceed wholesale cost plus a margin buffer.

Crucially, these guardrails are implemented as deterministic filters in `get_eligible_actions`. Forbidden arms are pruned before the bandit scores are evaluated. If all promotional arms are disqualified, the system falls back safely to zero percent full retail price."

*Transition:* "Let us clearly examine the data foundations and where real data ends and simulation begins."

---

### Slide 6: Data Pipeline & Strict Boundary (Time: 1:15)
"I want to be completely honest about our data foundations. We utilized a public retail dataset from Kaggle comprising seventy-three thousand one hundred store-product-day records across multiple categories and pricing regimes.

However, public retail logs lack individual customer identifiers. Therefore, we establish a strict boundary: the historical store, product, and competitor prices are from the public dataset. But the customer registry—eight thousand synthetic profiles with RFM tiers, shopping baskets, and conversion outcomes—is simulated by design. Furthermore, wholesale cost is modeled under an assumed forty percent gross margin rate. All financial figures are simulated rupees."

*Transition:* "Let's see how the simulator generates demand."

---

### Slide 7: Demand Calibration & Simulator (Time: 1:15)
"To train and evaluate policies offline, we built a demand simulation environment. Purchase probability is modeled via a logistic sigmoid link incorporating offered discount, customer price sensitivity, competitor price gap, and inventory scarcity.

To prevent arbitrary elasticity assumptions, we calibrated the logistic coefficients against historical demand tertiles in the dataset, achieving an empirical ROC-AUC of 0.6322. A modest AUC of 0.63 reflects noisy, realistic retail demand—it confirms that the simulator embeds genuine trade-offs without trivial deterministic patterns. The simulator also computes counterfactual outcomes across all five arms simultaneously, giving us an omniscient Oracle for regret tracking."

*Transition:* "Now let's examine our evaluation protocol."

---

### Slide 8: Evaluation Methodology (Time: 1:15)
"We designed a dual-mode evaluation protocol to avoid conflating learning costs with mature policy quality.

First, Online Mode: twenty thousand sequential rounds from a cold start. The bandit learns in-flight, suffering exploration regret. Second, Frozen Mode: five thousand held-out rounds where model weights are locked, measuring operational steady-state performance.

We replicated all runs across ten independent random seeds, Seeds 42 through 51. All five competing policies face identical customer arrival sequences with synchronized per-round random seeds. Finally, we track an immutable financial ledger recording every rupee of revenue, profit, and conceded discount."

*Transition:* "Here is our most surprising empirical result."

---

### Slide 9: Headline Result: The Profit Paradox (Time: 1:30)
"Our most important headline finding is that the heuristic benchmark, Fixed 10%, earned the highest total profit in both Online and Frozen modes.

Across ten seeds in Online mode, Fixed 10% achieved eighty thousand two hundred sixty-five rupees of cumulative profit. LinUCB achieved seventy-eight thousand nine hundred twenty-five rupees, suffering a deficit of one thousand three hundred forty-one rupees, or minus 1.67%. Thompson Sampling trailed by minus 1.76%. 

Why did a static ten percent rule beat adaptive bandits? Because in our calibrated retail market, ten percent discount sits at the exact microeconomic sweet spot. It provides enough price stimulus to convert shoppers without incurring the severe margin loss of fifteen or twenty percent markdowns. The bandits explored fifteen and twenty percent discounts, but the incremental volume gained could not offset the halved unit margin."

*Transition:* "However, when we look at top-line revenue, the picture flips entirely."

---

### Slide 10: Headline Result: Revenue Leadership (Time: 1:15)
"LinUCB generated the highest gross revenue in both Online mode—two hundred thirty-four thousand two hundred seventy-seven rupees—and Frozen mode—fifty-eight thousand four hundred sixty-five rupees.

LinUCB learned customer price elasticity and offered higher discounts—averaging 8.45% versus 8.09% for Fixed 10%. This lifted conversion rates to 24.29% compared to 23.81% for Fixed 10%. Furthermore, benchmarked against the standard commercial Rule-based policy, LinUCB achieved a profit lift of plus four hundred ninety-two rupees, winning in eight out of ten evaluation seeds. LinUCB proved superior at expanding market share and top-line volume."

*Transition:* "Let's examine the mathematical convergence of regret."

---

### Slide 11: Learning & Regret Convergence (Time: 1:15)
"This slide displays the cumulative expected regret curves over twenty thousand rounds. Notice the characteristic sublinear profile. In the first five hundred rounds, active exploration produces regret of nearly twenty-three paise per round. By the final two thousand rounds, regret flattens to nine paise per round for LinUCB and under five paise per round for Thompson Sampling.

Importantly, neither curve flattens to zero. In theory, linear bandit regret flattens asymptotically. Here, a steady residual slope persists because the simulator generates demand via a non-linear logistic sigmoid, whereas LinUCB assumes a linear payoff. This non-zero asymptote reflects an irreducible functional approximation gap."

*Transition:* "Now let's examine the money view through our financial ledger."

---

### Slide 12: Financial Ledger Dynamics (Time: 1:15)
"Our transaction ledger exposes cash flow dynamics that regret curves hide. During the initial fifteen hundred rounds, LinUCB incurs an 'exploration tax,' trailing the Rule-based baseline by up to two hundred eighty rupees.

However, as parameter estimates mature, LinUCB begins to selectively exploit profitable elasticities. It crosses break-even against the Rule-based policy at Round 3,200 on Seed 42, expanding its profit lead to four hundred ninety-two rupees by the end. The ledger also shows that LinUCB conceded twenty-four thousand six hundred rupees in discounts versus twenty thousand nine hundred for Fixed 10%, illustrating the exact cash trade-off between volume acquisition and margin preservation."

*Transition:* "Next, let's verify how our commercial guardrails performed."

---

### Slide 13: Tier Breakdown & Guardrail Audit (Time: 1:15)
"Across all ten evaluation seeds and two hundred thousand cumulative online rounds, our programmatic guardrails achieved exactly one hundred percent compliance: zero margin floor breaches and zero loyalty tier violations.

In the New customer tier, LinUCB respected the five percent ceiling, averaging 3.11% discount. In the Loyal tier, it leveraged its higher twenty percent ceiling to average 11.02%. 

In stark contrast, when we audited the raw historical Kaggle dataset against our tier rules, 39.04% of historical records violated loyalty tier policies. This proves that historical human discounting was undisciplined, reinforcing why algorithmic systems require deterministic guardrails."

*Transition:* "We also evaluated policies on real historical logs via off-policy replay."

---

### Slide 14: Off-Policy Replay Sanity Check (Time: 1:15)
"Using the rejection sampling methodology of Li et al., we evaluated policies against seventy-three thousand one hundred historical records. Replay match rates were consistent across policies at roughly twenty percent.

LinUCB achieved a replay profit of six rupees and eleven paise per round, compared to five rupees and eighty-seven paise for Fixed 10%. However, we do NOT claim this proves real-world superiority. First, ninety-five percent confidence intervals overlap completely across all policies. Second, 39% of historical rows violated guardrails and could never match. Third, historical discounts were logged non-randomly by store managers. Replay serves as a valuable sanity check, but not a causal proof."

*Transition:* "To make these models useful to store managers, we built an interactive application."

---

### Slide 15: Operational Streamlit Dashboard (Time: 1:15)
"We built an operational decision-support tool in Streamlit. The architecture strictly decouples the presentation layer from the inference engine. 

The application loads frozen, pre-trained models—it performs forward inference only, with zero parameter mutation during user sessions. Store managers can input customer profiles, inspect recommended discounts, view expected conversion curves, and run counterfactual what-if simulations.

Crucially, the app provides complete explainability: it decomposes the LinUCB score into expected mean payoff and uncertainty bonus, and explicitly states why forbidden arms were blocked by guardrails. It runs one hundred percent offline with zero cloud or API dependencies."

*Transition:* "Let me summarize our primary contributions and conclusions."

---

### Slide 16: Conclusions & Takeaways (Time: 1:15)
"To conclude:
First, contextual bandits successfully learn price elasticity and maximize gross revenue, generating two hundred thirty-four thousand rupees and outperforming commercial rule-based systems in eighty percent of seeds.

Second, sophisticated algorithms do not automatically beat simple heuristics on profit. Fixed 10% won because it avoided excessive margin concessions. Merchandisers must align reward functions with net cash margin rather than top-line volume.

Third, deterministic guardrails are non-negotiable. Hard programmatic filtering achieved one hundred percent operational safety without disrupting learning.

All results are derived within our calibrated simulator. Thank you, and I look forward to your questions."

---

## Defense & Backup Notes (For Q&A)

### If asked about Slide 17 (Alpha Sensitivity):
"We evaluated alpha from zero to five on separate tuning seeds to prevent evaluation bias. Alpha between 0.1 and 1.0 is extremely robust; alpha equal to 0.5 balanced early exploration without destabilizing late exploitation."

### If asked about Slide 18 (Reward Penalty Ablation):
"We tested adding a penalty lambda to the reward to discourage discounting. As lambda increased, the policy didn't become more nuanced—it collapsed toward zero discount everywhere, destroying revenue. This proves that hard guardrails are superior to soft penalty terms."

### If asked about Slide 20 (Sherman-Morrison Inversion):
"Inverting a twenty-by-twenty matrix at every round takes O(d-cubed) operations, or about eight thousand flops per update. Sherman-Morrison updates the inverse directly via an outer product in O(d-squared), or four hundred operations, enabling microsecond training on standard CPUs."

### If asked about Slide 21 (Threats to Validity):
"The three main limitations are: first, public data lacks individual customer keys, so the RFM registry is simulated; second, wholesale cost is modeled under an assumed 40% margin; third, training and evaluation share the same simulator. These are fully documented in Section 9 of our report."
