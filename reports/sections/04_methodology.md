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
