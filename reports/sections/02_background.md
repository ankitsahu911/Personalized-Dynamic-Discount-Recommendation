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
