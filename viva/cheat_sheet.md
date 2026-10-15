# Viva Voce 1-Page Quick Reference Cheat Sheet

### 1. The Project in 3 Sentences
We built a contextual multi-armed bandit system (LinUCB and Thompson Sampling) to recommend personalized retail discounts across five discrete levels ({0%, 5%, 10%, 15%, 20%}). The bandit is constrained by hard commercial guardrails: customer loyalty tier ceilings and an absolute unit cost margin floor. Evaluated over 10 seeds across 20,000 online rounds in a calibrated simulator, LinUCB generated the highest gross revenue (₹234,277) and beat Rule-based policies in 8 of 10 seeds, while the heuristic Fixed 10% earned the highest net profit (₹80,265).

### 2. The 10 Most Important Numbers
- **₹80,265.41 ± 1,788:** Fixed 10% Online profit (Profit Winner).
- **₹78,924.82 ± 788:** LinUCB Online profit (-1.67% vs. Fixed 10%).
- **+₹492.50 ± 1,022:** LinUCB profit lift vs. Rule-based (won in 8 of 10 seeds).
- **₹234,276.84 ± 4,477:** LinUCB Online revenue (Revenue Winner).
- **24.29% vs. 23.81%:** LinUCB vs. Fixed 10% conversion rate.
- **8.45% vs. 8.09%:** LinUCB vs. Fixed 10% average discount offered.
- **Round 3,200:** Break-even round where LinUCB surpassed Rule-based on Seed 42.
- **100% (0 breaches):** Guardrail compliance across 200,000 online decisions.
- **39.04%:** Historical Kaggle dataset tier rule violation rate.
- **0.6322:** Demand calibration ROC-AUC (realistic noisy retail signal).

### 3. The 3 Core Topics (Claim → Evidence → Caveat)
1. **Guardrails:** Hard rules prevent below-cost sales and loyalty violations → 100% compliance across 200k rounds vs. 39% historical violations → Restricts discounts on New customers.
2. **Regret & Ledger:** Regret proves convergence; ledger proves LinUCB led revenue but trailed Fixed 10% on profit → Regret flattens to ₹0.09/round; LinUCB conceded ₹24.6k vs. ₹20.9k discounts → Results are within the simulator.
3. **Limitations:** Evaluated in a semi-synthetic simulator → 73,100 Kaggle rows have no customer IDs; 8k registry is synthetic; assumed 40% margin → Replay is a sanity check, not causal proof.

### 4. Examination Safety Sentence
*If asked an unexpected technical question you cannot answer immediately, state calmly:*  
**"I'm not certain of that specific number off-hand — here is how I would check it in my transaction ledger and codebase."**
