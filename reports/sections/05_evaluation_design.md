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
