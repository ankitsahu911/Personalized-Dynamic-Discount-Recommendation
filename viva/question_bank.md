# Comprehensive Viva Voce Question Bank (50 Questions)

*Contains 40 examiner questions across 6 categories with complete answer skeletons, plain summaries, trap warnings, and follow-ups, plus 10 self-examination questions.*

---

## Category 1: Why the Guardrail Exists (Core Topic 1)

### Q01: Why did you enforce loyalty tiers as a hard rule instead of letting the bandit learn them?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* Hard rules provide an absolute commercial guarantee against brand damage and margin forfeiture.
  - *Evidence:* Programmatic guardrails achieved 100% compliance across 200,000 decisions (`multiseed_online_by_tier.csv`), whereas soft reward penalties collapsed revenue by ₹20k (`reward_ablation.csv`).
  - *Caveat:* Hard rules restrict the bandit from testing deep acquisition discounts on New customers.
- **Say it Simply:** "In retail, you can't afford to let an algorithm learn through trial and error that giving big discounts to new users hurts brand trust. A hard rule guarantees safety from day one."
- **Trap to Avoid:** Do not say "the bandit couldn't learn it." It could learn, but the commercial risk during learning is unacceptable.
- **Follow-up:** "What would happen if you removed the guardrails completely?"

### Q02 [TRAP]: Your guardrail caps New customers at 5%. Doesn't that hurt customer acquisition?
- **Difficulty:** Trap
- **Skeleton Answer:**
  - *Claim:* Yes, it slightly limits acquisition conversion, but it protects profitability and prevents conditioning shoppers to expect markdowns.
  - *Evidence:* In the New tier, LinUCB achieved ₹19,410 profit with an average discount of 3.11%, while Fixed 10% offered 5% and earned ₹19,652 (`multiseed_online_by_tier.csv`).
  - *Caveat:* Retailers focused purely on volume might accept lower margins to acquire users.
- **Say it Simply:** "It does trade off some volume, but giving steep discounts to unverified shoppers creates margin bleed. The rule ensures new customers convert on product value, not pure price slashing."
- **Trap to Avoid:** Do not deny that it hurts acquisition volume; acknowledge the trade-off openly.
- **Follow-up:** "How would you adjust the rule for a high-inventory clearance product?"

### Q03: What does the unit cost margin floor do and how is it implemented?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* It programmatically disqualifies any discount arm that would price an item below wholesale cost plus a buffer.
  - *Evidence:* Evaluated in `get_eligible_actions(context)` as $	ext{DiscountedPrice} \ge 	ext{CostPrice} + 	ext{Floor}$; achieved 0 breaches across 200,000 rounds (`rules.py`).
  - *Caveat:* In our benchmark with 40% margin and 20% max discount, all 5 arms remained above cost by default; the floor activates when users configure tighter buffers in the app.
- **Say it Simply:** "It's a code-level safety check that says: 'Never sell an item below what it cost us to buy.' If an arm breaches the floor, it gets thrown out before the algorithm even scores it."
- **Trap to Avoid:** Do not say "the bandit knows not to sell below cost." The bandit has no concept of cost unless constrained.
- **Follow-up:** "What happens if all promotional arms are blocked by the margin floor?"

### Q04: What happens when the eligible set is empty?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* The system falls back cleanly and safely to $\{0\%\}$ full retail price.
  - *Evidence:* Hard-coded in `get_eligible_actions` in `src/guardrails/rules.py`.
  - *Caveat:* Zero discount may reduce conversion probability for price-sensitive shoppers, but preserves 100% unit margin.
- **Say it Simply:** "If every discount is blocked because margins are too thin, the system safely falls back to selling the item at its full shelf price."
- **Trap to Avoid:** Do not say "the system throws an error or skips the round."
- **Follow-up:** "Could the fallback be dynamic, like 2.5%?"

### Q05: Why did 39.04% of historical records violate your loyalty tier rules?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* Historical store managers applied promotional discounts opportunistically without structured loyalty governance.
  - *Evidence:* Documented in `reports/fairness_and_business_audit.json` across 73,100 Kaggle rows.
  - *Caveat:* The public dataset records store-level promotional markdowns, not individualized loyalty transactions.
- **Say it Simply:** "In the historical data, store managers handed out steep discounts to anyone to hit sales targets. It proves that human retail discounting was undisciplined, which is why automated guardrails are needed."
- **Trap to Avoid:** Do not say "the historical dataset was corrupted or full of bugs."
- **Follow-up:** "How did that 39% violation rate affect your off-policy replay evaluation?"

### Q06: What happens when a user loosens guardrails in your Streamlit application?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* The app permits wider action spaces but issues an extrapolation and margin risk warning.
  - *Evidence:* Implemented in `app/streamlit_app.py:render_tab_whatif`.
  - *Caveat:* Model parameters were trained under historical tier constraints; predicting unconstrained behavior involves statistical extrapolation.
- **Say it Simply:** "The app lets merchandisers test what-if scenarios with loosened rules, but it displays an amber warning showing that predicting outside the guardrails carries higher uncertainty."
- **Trap to Avoid:** Do not say "the model handles relaxed rules with 100% confidence."
- **Follow-up:** "Can store managers override the guardrail in real time?"

---

## Category 2: What Regret & Ledger Prove (Core Topic 2)

### Q07: Explain what cumulative regret is and why you computed it on expected rather than realized profit.
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* Regret measures the cumulative expected profit forfeited by choosing suboptimal actions relative to an omniscient Oracle.
  - *Evidence:* Formulated in Report §5.1 following Lattimore & Szepesvari [6].
  - *Caveat:* Computing regret on realized binary outcomes ($Y_t \in \{0, 1\}$) injects severe binomial noise, masking algorithmic convergence.
- **Say it Simply:** "Regret is the difference between what we expected to make with the best possible discount and what we made with the one we picked. We use expected profit because if a customer randomly decides not to buy, that's bad luck, not a bad algorithm."
- **Trap to Avoid:** Never say "regret is lost revenue" or "regret was computed on real sales."
- **Follow-up:** "How do you know what the best possible discount was (the Oracle)?"

### Q08: Why does your cumulative regret curve flatten to a non-zero slope instead of zero?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* A linear contextual bandit cannot perfectly approximate a non-linear logistic sigmoid demand surface across 20 dimensions.
  - *Evidence:* LinUCB window regret stabilizes at ₹0.0919/round in the final 2,000 rounds (`regret_checkpoints.csv`).
  - *Caveat:* Regret flattens dramatically (sublinear convergence), but the residual slope represents an irreducible functional approximation gap.
- **Say it Simply:** "Our simulator generates purchases using a curved S-shaped formula, but LinUCB uses a flat linear formula. A flat line can never perfectly match a curve in 20 dimensions, so there is always a small, permanent gap."
- **Trap to Avoid:** Do not say "the model needed more rounds to reach zero." It will never reach zero with a linear model.
- **Follow-up:** "How would NeuralUCB address this?"

### Q09: What does the financial ledger show that cumulative regret curves hide?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* Regret measures theoretical algorithmic distance from an oracle; the ledger measures real-world commercial cash flows and trade-offs.
  - *Evidence:* The ledger (`results/ledger_seed42.csv`) reveals the initial ₹280 exploration tax, the Round 3,200 break-even crossover, and LinUCB's gross revenue leadership (₹234k).
  - *Caveat:* Ledger cash flows represent simulated rupees within the calibrated simulator.
- **Say it Simply:** "Regret tells you how smart the algorithm is compared to perfection; the ledger tells you whether you're making or losing actual cash in the cash register."
- **Trap to Avoid:** Do not treat regret and profit as interchangeable metrics.
- **Follow-up:** "What were the accounting assertions you used to validate the ledger?"

### Q10 [TRAP]: Your report says Fixed 10% beat LinUCB. Doesn't that mean your machine learning project failed?
- **Difficulty:** Trap
- **Skeleton Answer:**
  - *Claim:* No; it demonstrated that LinUCB successfully learned to optimize revenue and conversion, while exposing the exact price elasticity mechanics of retail pricing.
  - *Evidence:* LinUCB generated the highest gross revenue (₹234,276.84), achieved highest conversion (24.29%), and beat commercial Rule-based policies in 8 of 10 seeds (+₹492.50) (`comparison_lift.csv`).
  - *Caveat:* Fixed 10% earned higher profit because the simulator's margin structure severely penalized discounts above 10%.
- **Say it Simply:** "Not at all. In research, discovering that a simple rule hits the profit sweet spot while machine learning expands top-line revenue is a major, honest insight. LinUCB proved superior at driving volume and beat standard enterprise retail rules in 80% of runs."
- **Trap to Avoid:** Do not try to spin the numbers or pretend LinUCB won on profit. Own the result proudly.
- **Follow-up:** "How would you modify the reward function to make LinUCB beat Fixed 10% on profit?"

### Q11: What is the difference between Online and Frozen evaluation, and why do you report both?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* Online evaluation measures learning speed and exploration cost from a cold start; Frozen evaluation measures steady-state operational policy performance.
  - *Evidence:* Online evaluates 20,000 rounds with updates; Frozen evaluates 5,000 held-out rounds with locked weights (`comparison_online.csv` vs. `comparison_frozen.csv`).
  - *Caveat:* Policies preserved their relative rankings across both modes.
- **Say it Simply:** "Online is testing a student while they are studying; Frozen is testing them after graduation. You need both to know what learning costs and how well the finished model performs."
- **Trap to Avoid:** Do not mix up the round counts (20k online vs. 5k frozen).
- **Follow-up:** "Did LinUCB beat Fixed 10% in Frozen mode?"

### Q12: Why did LinUCB trail the Rule-based baseline in early rounds?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* LinUCB begins with uninformative priors ($A = I_d, b = 0$) and pays an active "exploration tax" while learning context-payoff relationships.
  - *Evidence:* Trailed Rule-based by up to -₹280 in the first 1,500 rounds before crossing break-even at Round 3,200 (`ledger_seed42.csv`).
  - *Caveat:* In a real store, cold-start exploration can be mitigated by warm-starting LinUCB with offline logistic regression weights.
- **Say it Simply:** "At the start, the bandit knows nothing, so it has to try different discounts to see what works. That trial-and-error costs money early on, but once it learns, it overtakes the rule-based policy."
- **Trap to Avoid:** Do not say "it had a bug in early rounds."
- **Follow-up:** "How could you warm-start LinUCB to eliminate that early deficit?"

---

## Category 3: Dataset and Simulation Limitations (Core Topic 3)

### Q13: What part of your dataset is real and what part is simulated?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* Historical store, product, pricing, and inventory features are from a public Kaggle dataset; customer registry, shopping baskets, and conversion outcomes are simulated.
  - *Evidence:* Documented in Report §3.5 and Table 3.1.
  - *Caveat:* Public dataset contains 73,100 store/product/day logs but no individual customer identifiers.
- **Say it Simply:** "The products, prices, and stock counts come from a public Kaggle retail dataset. But the individual customers, their shopping visits, and their buying decisions were created inside our simulator."
- **Trap to Avoid:** Never say "we used real customer transaction data."
- **Follow-up:** "Why didn't you use a dataset with real customer IDs?"

### Q14: What does your demand calibration AUC of 0.6322 tell you about your simulator?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* An AUC of 0.6322 indicates a modest, directionally valid predictive signal with realistic retail noise.
  - *Evidence:* Calibrated via logistic regression on historical demand tertiles (`reports/evaluation_findings.md`).
  - *Caveat:* An AUC of 0.6322 means the simulator embeds substantial unexplained variance; outcomes are stylized approximations of human elasticity.
- **Say it Simply:** "It means our simulator has a realistic amount of noise. A 0.99 AUC would mean the simulation is trivial and fake. An AUC of 0.63 means the bandit has to work hard to find subtle patterns under genuine uncertainty."
- **Trap to Avoid:** Do not say "0.6322 is a great predictive accuracy." In supervised learning it is modest; in retail simulation it is realistic.
- **Follow-up:** "How would your results change if demand elasticity were much steeper?"

### Q15 [TRAP]: Your off-policy replay on 73,100 records proves this will work in a real store, doesn't it?
- **Difficulty:** Trap
- **Skeleton Answer:**
  - *Claim:* Absolutely not. Replay serves as an observational sanity check, not a causal proof.
  - *Evidence:* Overlapping 95% confidence intervals across all policies (LinUCB ₹6.11 vs. Fixed 10% ₹5.87) (`replay_results.csv`).
  - *Caveat:* Historical discounts were logged non-randomly by store managers, and 39% of historical rows violated guardrails, violating replay's theoretical validity conditions.
- **Say it Simply:** "No, definitely not. The confidence intervals overlap, store managers didn't pick discounts at random, and 39% of past discounts broke our rules. Replay just shows our model behaves sensibly on past data—it doesn't prove live lift."
- **Trap to Avoid:** Never claim replay validates the model or proves real-world performance.
- **Follow-up:** "Under what conditions would replay provide an unbiased estimate?"

### Q16: Why did you assume a uniform 40% margin rate for wholesale cost?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* True wholesale procurement invoices are proprietary trade secrets absent from public datasets.
  - *Evidence:* Formulated in Report §3.3 as $	ext{CostPrice} = 0.60 	imes 	ext{BasePrice}$.
  - *Caveat:* Real physical retail margins vary widely by category (e.g., 5% on groceries vs. 60% on apparel); uniform assumptions distort cross-category comparisons.
- **Say it Simply:** "Real wholesale supplier costs are confidential and not in the Kaggle dataset. We used an assumed 40% gross margin as an industry benchmark, meaning a 20% discount cuts unit profit in half."
- **Trap to Avoid:** Do not say "the dataset had wholesale costs."
- **Follow-up:** "How would category-specific margins affect arm selection?"

### Q17 [TRAP]: Are your 10-seed evaluation results statistically significant?
- **Difficulty:** Trap
- **Skeleton Answer:**
  - *Claim:* We intentionally did not run hypothesis tests or claim statistical significance.
  - *Evidence:* Documented in Report §5.3 and §9.3.
  - *Caveat:* 10 random seeds represent algorithmic replications on a single simulator, not a sample of independent retail stores. Quoting $p$-values would be scientifically misleading.
- **Say it Simply:** "We don't claim statistical significance, and we didn't run p-value tests on purpose. Running 10 seeds in one simulator shows algorithmic stability, but claiming statistical significance would falsely imply we tested across 10 real retail chains."
- **Trap to Avoid:** Never cite a $p$-value or say "results were significant at $p < 0.05$."
- **Follow-up:** "How do you report confidence across seeds without p-values? (Answer: Paired differences, standard deviations, and win counts)."

### Q18: What is internal alignment bias in your evaluation?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* Both policy training and policy evaluation were conducted within the identical simulation environment.
  - *Evidence:* Documented in Report §9.2 ("Simulation Limitations").
  - *Caveat:* Because the simulator's generative rules did not shift, the model was evaluated under the exact behavioral assumptions it learned, flattering convergence.
- **Say it Simply:** "It's the fact that our model was tested on the same simulator it practiced on. In a real store, customer tastes and competitor prices change unexpectedly, which didn't happen in our static simulator."
- **Trap to Avoid:** Do not deny the existence of alignment bias.
- **Follow-up:** "How could you test policy robustness against non-stationarity?"

---

## Category 4: Methods and Algorithms

### Q19: Explain the LinUCB decision rule and its score decomposition in plain English.
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* LinUCB computes an optimistic score for each arm: expected payoff plus an exploration bonus proportional to parameter uncertainty.
  - *Evidence:* $p_{t,a} = x_t^T \hat{	heta}_a + lpha \sqrt{x_t^T A_a^{-1} x_t}$ (Report §2.2.2).
  - *Caveat:* The score assumes a linear payoff relationship, which creates approximation error on sigmoidal demand.
- **Say it Simply:** "For every discount, LinUCB asks two things: 'How much profit do I think this will make?' and 'How unsure am I about this choice?' It adds them together and picks the highest score, ensuring it explores unfamiliar discounts."
- **Trap to Avoid:** Do not confuse $lpha$ with a learning rate; it is an exploration confidence multiplier.
- **Follow-up:** "What happens when $lpha = 0$?"

### Q20: How does the Sherman-Morrison formula work and why did you use it?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* It computes the rank-1 update to the inverse covariance matrix directly in $\mathcal{O}(d^2)$ operations instead of inverting $A_a$ from scratch in $\mathcal{O}(d^3)$.
  - *Evidence:* $M_a \leftarrow M_a - rac{M_a x x^T M_a}{1 + x^T M_a x}$ (Report §2.2.3 and `src/policies/linucb.py`).
  - *Caveat:* Requires maintaining the inverse matrix directly, which must be guarded against numerical drift via regularized initialization ($M = I_d$).
- **Say it Simply:** "Inverting a 20x20 matrix from scratch takes 8,000 math operations every single round. Sherman-Morrison updates the existing inverse in just 400 operations, making the model run twenty times faster."
- **Trap to Avoid:** Do not say "it changes the mathematical prediction." The result is mathematically identical to full inversion.
- **Follow-up:** "What is the numerical stability risk of Sherman-Morrison?"

### Q21: How does contextual Thompson Sampling differ from LinUCB?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* LinUCB explores deterministically via upper confidence bounds; Thompson Sampling explores stochastically by sampling from a Bayesian posterior.
  - *Evidence:* Draws $	ilde{	heta}_a \sim \mathcal{N}(\hat{	heta}_a, v^2 B_a^{-1})$ and selects greedy action (Report §2.3).
  - *Caveat:* Thompson Sampling achieved lower cumulative regret on Seed 42 (₹1,075 vs. ₹1,754) and higher oracle match rate (62.2% vs. 16.9%), but has higher variance across seeds.
- **Say it Simply:** "LinUCB always picks the highest optimistic ceiling. Thompson Sampling rolls dice based on its probability distribution—if it's unsure, the dice roll can land anywhere, giving other discounts a chance."
- **Trap to Avoid:** Do not say "Thompson Sampling is non-contextual." Our implementation conditions on the 20D context vector.
- **Follow-up:** "Why did Thompson Sampling have higher variance across seeds in your results?"

### Q22: What does the hyperparameter $lpha$ do and how did you choose $lpha = 0.5$?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* $lpha$ scales the exploration uncertainty bonus $\sqrt{x_t^T A_a^{-1} x_t}$.
  - *Evidence:* Swept across $\{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\}$ on separate tuning seeds (101–105); $lpha = 0.5$ minimized regret while ensuring steady exploitation (`results/alpha_sensitivity.csv`).
  - *Caveat:* $lpha \in [0.1, 1.0]$ is broad and robust; performance degrades only at $lpha \ge 2.0$.
- **Say it Simply:** "Alpha is the curiosity dial. At zero, it never explores; at five, it's too curious and wastes money on bad discounts. We tested six values on separate tuning seeds and found 0.5 was the sweet spot."
- **Trap to Avoid:** Do not say you tuned $lpha$ on Seeds 42–51 (that would be test-set leakage).
- **Follow-up:** "Why didn't you use $lpha = 0.0$ if it had lower early regret?"

### Q23: Why optimize net profit instead of conversion rate or revenue?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* Maximizing conversion rate or revenue incentivizes offering the maximum allowable discount (20%) on every transaction, causing massive margin destruction.
  - *Evidence:* LinUCB offered 8.45% discount and won on revenue (₹234k), but conceded ₹24.6k in discounts and lost on profit to Fixed 10% (`comparison_online.csv`).
  - *Caveat:* In corporate settings, merchandisers occasionally run loss-leader campaigns to gain market share, but bottom-line solvency requires profit optimization.
- **Say it Simply:** "If you optimize for conversions, the model will give everyone 20% off because free stuff sells faster. But you go bankrupt selling at zero margin. Optimizing profit forces the model to protect cash flow."
- **Trap to Avoid:** Do not dismiss revenue; acknowledge that revenue expansion has strategic value.
- **Follow-up:** "How could a retailer balance both revenue and profit goals?"

### Q24: What are the 20 features in your context vector?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* A 20-dimensional vector combining standardized product pricing, stock availability, simulated customer RFM profile, derived price gaps, and temporal indicators.
  - *Evidence:* Detailed in Table 3.1 of Report §3.4.
  - *Caveat:* Categorical features (tiers, demand tertiles) are one-hot encoded; numerical features are standardized to zero mean and unit variance.
- **Say it Simply:** "It's 20 numbers describing the situation: product price, competitor price, stock level, customer visit history, days since last visit, day of the week, and customer tier."
- **Trap to Avoid:** Do not recite all 20 from memory; group them into Product, Customer, Market, and Time.
- **Follow-up:** "How did you prevent data leakage during feature standardization?"

---

## Category 5: The Application and Deployment

### Q25: Does your Streamlit web application learn from user interactions in real time?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* No. The application executes purely forward inference using pre-trained, frozen model weights (`linucb_v1.npz`).
  - *Evidence:* Documented in Report §7.1 and `app/streamlit_app.py`.
  - *Caveat:* Forward inference guarantees deterministic, reproducible recommendations without risk of model corruption from rogue user inputs.
- **Say it Simply:** "No, the app uses a frozen model. It takes what the user inputs, calculates the recommendation using pre-trained weights, and gives the answer in milliseconds. It does not update its brain during the session."
- **Trap to Avoid:** Do not say "yes, it learns every time you click a button."
- **Follow-up:** "What would be required to enable real-time learning in the app?"

### Q26: How does the application explain recommendations to non-technical store managers?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* Decomposes the LinUCB score into expected mean payoff and uncertainty bonus, and displays explicit plain-English blocking reasons for disqualified arms.
  - *Evidence:* Implemented in `app/streamlit_app.py:render_tab_recommend` (Report §7.3).
  - *Caveat:* Explaining uncertainty bonuses requires careful visual labeling so store managers do not confuse epistemic uncertainty with financial risk.
- **Say it Simply:** "It breaks the recommendation into two simple bars: 'How much profit we expect' and 'How curious the model is.' If a discount is forbidden, it says clearly: 'Blocked: Exceeds 5% cap for New Customers.'"
- **Trap to Avoid:** Do not say "we use SHAP values." We use explicit UCB score decomposition.
- **Follow-up:** "Why didn't you use SHAP or tree-based feature importance?"

### Q27: What is the What-If Simulator in your application showing?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* An interactive slider that sweeps discount percentages from 0% to 20%, plotting projected conversion probability against net profit margin.
  - *Evidence:* Implemented in `app/streamlit_app.py:render_tab_whatif`.
  - *Caveat:* Surfaces the exact profit-maximizing apex and triggers amber callouts if the selected discount breaches the unit cost margin floor.
- **Say it Simply:** "It lets store managers play with a discount slider and watch the profit curve rise and fall in real time, so they can see why giving 15% instead of 10% throws money away."
- **Trap to Avoid:** Do not say it re-trains the model; it queries the simulator demand curve.
- **Follow-up:** "What happens if a user tests a shelf price of ₹10,000 in the what-if tool?"

### Q28: What is the difference between Profile Tier and Context Tier in your application?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* Profile Tier governs the legal guardrail discount ceiling (e.g., Loyal max 20%); Contextual RFM features (visits, recency) feed the numerical bandit vector.
  - *Evidence:* Documented in Report §7.4 and `app/about_text.md`.
  - *Caveat:* Decoupling them allows merchandisers to test edge cases (e.g., a customer marked 'Loyal' who has only visited once) while the app displays an advisory note.
- **Say it Simply:** "Profile tier is the legal policy ceiling—what discount category they belong to. Context tier is their actual mathematical purchase history. Decoupling them lets managers test edge cases safely."
- **Trap to Avoid:** Do not say they are the same thing; they serve different architectural roles.
- **Follow-up:** "Why wouldn't you derive the tier strictly from the purchase count automatically?"

### Q29: Does your application require an internet connection or external cloud APIs?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* No; the application is 100% self-contained and operates completely offline with zero external network calls.
  - *Evidence:* All model parameters and dataset tables are bundled locally; matrix operations run in pure local NumPy (Report §7.1).
  - *Caveat:* Cloud deployment via Docker/AWS is straightforward, but local execution guarantees zero latency and data privacy.
- **Say it Simply:** "It runs 100% offline on a laptop. You can turn off Wi-Fi completely, and the app runs in milliseconds with zero external API calls."
- **Trap to Avoid:** Do not say "it uses OpenAI or Google Cloud APIs."
- **Follow-up:** "How would you package this for deployment across 500 retail branches?"

---

## Category 6: Process, Engineering, and Next Steps

### Q30: What was the single most difficult engineering challenge in this project?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* Building a calibrated simulation environment that exhibited realistic non-linear price elasticity without creating artificial shortcuts for the bandit.
  - *Evidence:* Requiring logistic calibration on historical demand tertiles (AUC 0.6322) and counterfactual generation across all 5 arms (`src/simulation/simulator.py`).
  - *Caveat:* A simulator that is too simple makes bandits learn trivially; a simulator that is too noisy prevents learning entirely.
- **Say it Simply:** "Building the simulator. Bandits need an environment to practice on, but if you make the simulator too predictable, the bandit cheats. Calibrating it to real retail data with realistic noise took weeks of careful tuning."
- **Trap to Avoid:** Do not say "writing the Streamlit frontend" or "installing packages."
- **Follow-up:** "How did you verify the simulator was working before running full experiments?"

### Q31: What did you change from your original project specification and why?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* We prioritized Sherman-Morrison matrix updates, dual online/frozen evaluation, and comprehensive financial ledgers over complex multi-period deep RL.
  - *Evidence:* Documented in Appendix B (Design-Decision Log) of Report §13.
  - *Caveat:* The original plan considered NeuralUCB and PostgreSQL; we focused on linear bandit mathematical rigor and production auditability first.
- **Say it Simply:** "We focused on linear algebra efficiency and financial ledgers rather than chasing deep neural networks. In retail pricing, having an O(d^2) model with 100% auditable accounting is far more valuable than a black-box neural net."
- **Trap to Avoid:** Do not say "we ran out of time." Say "we made deliberate engineering trade-offs for auditability."
- **Follow-up:** "Would you recommend NeuralUCB for production?"

### Q32: If you had another month, what is the first technical upgrade you would implement?
- **Difficulty:** Easy
- **Skeleton Answer:**
  - *Claim:* Validating the bandit on an authentic identity-bearing transaction corpus (such as the UCI Online Retail dataset) and implementing NeuralUCB [8].
  - *Evidence:* Documented in Report §10 ("Future Work").
  - *Caveat:* A real identity dataset would eliminate the need for a synthetic customer registry, enabling authentic empirical evaluation of customer journey dynamics.
- **Say it Simply:** "I would test this on the UCI Online Retail dataset, which has real customer invoice IDs. That would eliminate the need for our simulated customer registry and prove how well bandits personalize on genuine shopper journeys."
- **Trap to Avoid:** Do not propose cosmetic UI changes; propose deep methodological advancements.
- **Follow-up:** "How would you handle non-stationary customer preferences over time?"

### Q33: What would a real-world physical retail deployment require beyond this codebase?
- **Difficulty:** Medium
- **Skeleton Answer:**
  - *Claim:* Global store-wide promotional spend budgets, competitor dynamic reaction modeling, real-time POS streaming integration, and legal fairness audits.
  - *Evidence:* Documented in Report §9.5 ("Practical Deployment Barriers").
  - *Caveat:* An algorithm cannot be deployed in isolation without enterprise resource planning (ERP) integration.
- **Say it Simply:** "In a real store, you need monthly budget caps so you don't overspend on promotions, real-time integration with the cash register, and a legal review to ensure dynamic pricing complies with consumer laws."
- **Trap to Avoid:** Do not say "it's ready to deploy tomorrow as-is."
- **Follow-up:** "What legal or fairness issues arise with personalized discounts?"

### Q34: What is the price of restraint you identified in your reward ablation study?
- **Difficulty:** Hard
- **Skeleton Answer:**
  - *Claim:* Imposing an additive discount penalty term ($-\lambda \cdot a_t$) in the reward function causes the policy to collapse toward zero discounting globally, destroying revenue without improving profit.
  - *Evidence:* At $\lambda = 0.730$, average discount offered collapsed to 4.86% and revenue dropped by ₹20k from ₹235k to ₹214k (`results/reward_ablation.csv`).
  - *Caveat:* Demonstrates that soft reward shaping is economically destructive compared to hard programmatic guardrails.
- **Say it Simply:** "If you try to punish discounts by putting a penalty in the reward, the model gets scared and stops offering discounts to anyone. Sales collapse, revenue drops by ₹20,000, and profit doesn't improve."
- **Trap to Avoid:** Do not say "penalty terms work well." The data proved they caused policy collapse.
- **Follow-up:** "Why didn't the model learn to penalize only unprofitable discounts?"

---

## 10 Questions I Should Ask MYSELF (Self-Examination Check)

1. *Can I write the Sherman-Morrison update formula on a whiteboard from memory and explain why it is $O(d^2)$?*
2. *Can I explain why Fixed 10% beat LinUCB on profit without checking my notes?*
3. *Can I state the exact reason why LinUCB regret flattens to a non-zero slope ($pprox ₹0.09/	ext{round}$)?*
4. *Do I know where the number 39.04% comes from and what it means?*
5. *Can I explain why expected regret is scientifically superior to realized regret?*
6. *Can I explain why off-policy replay confidence intervals overlap?*
7. *Do I know the exact difference between discount offered (nominal %) and discount given (rupees conceded)?*
8. *Can I start the Streamlit app cold, navigate to the What-If tab, and trigger a margin floor alert in under 45 seconds?*
9. *Can I calmly say: "Within our calibrated simulation environment..." instead of "In the real world..."?*
10. *Can I recite my safety sentence without hesitation when asked an unknown detail?*
