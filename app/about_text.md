# About the Dynamic Discount Recommendation Engine

### System Overview
This web application is a decision-support prototype that recommends personalized retail discounts for e-commerce and store merchandising.
Rather than applying blanket store-wide markdowns, the engine pairs contextual multi-armed bandit algorithms (**Disjoint LinUCB** and **Thompson Sampling**) with hard business margin guardrails.
When an item context and customer profile are provided, the system scores eligible discount arms by balancing expected profit margin against epistemic uncertainty.
The system protects retail profitability by refusing to recommend discounts below a configurable margin floor or outside tier-specific loyalty menus.
All financial models deployed here are completely **frozen** and evaluated offline.

---

### Architectural Flowchart

```text
[73,100 Real Retail Transactions]
               │
               ▼
   [Feature Standardization & Encoding] (Z-score + One-Hot)
               │
               ▼
       [Customer Registry] (New, Regular, Loyal Tiers)
               │
               ▼
     [Simulator & Guardrails] (40% Cost Margin + Floor + Tier Menus)
               │
               ▼
[Contextual Bandit Policies] (Disjoint LinUCB & Thompson Sampling)
               │
               ▼
   [Offline Vectorized Evaluation] (10 Seeds × 20k Rounds + Ledgers + Replay)
               │
               ▼
  [Interactive Streamlit Application] (Decision Simulator + What-If + Ledgers)
```

---

### What is Real vs. What is Simulated

| Dimension | Status | Description & Source |
| :--- | :--- | :--- |
| **Product Catalogs & Pricing** | **Real** | Calibrated against 73,100 Kaggle store-inventory records (Price, Competitor Pricing, Inventory, Region, Category). |
| **Category & Season Distributions**| **Real** | Frequency distributions for 5 categories (Electronics, Groceries, Furniture, Toys, Clothing) and 4 seasons match observed dataset shares. |
| **Customer Identities & Recency** | **Simulated** | Synthetic customer registry modeling purchase frequencies and return intervals. |
| **Purchase Decisions** | **Simulated** | Non-linear logistic probability function calibrated against real historical demand elasticity ($\text{AUC} \approx 0.63$). |
| **Cost Basis** | **Assumed** | Standard retail 40% gross margin assumption ($\text{CostPrice} = 0.60 \times \text{Price}$). |
| **Financial Values (₹)** | **Simulated** | All displayed rupees represent simulated currency units within the offline environment. |

---

### Core System Limitations
1. **In-Simulator Evaluation:** Model training and evaluation were conducted within the calibrated simulator; field performance in physical retail stores may differ due to unmodeled omni-channel dynamics.
2. **Historical Confounding:** Over 39% of historical dataset transactions violated loyalty guardrails, and promotions were heavily confounded with discount assignment ($\text{Cramér's } V = 0.4755$). Replay results represent a directional sanity check.
3. **Stationary Offline Model:** The LinUCB model is frozen at Milestone 31; this web application does not update model weights online.
4. **Uniform Cost Model:** Real retail environments experience supplier volume rebates, freight surcharges, and perishable holding costs not reflected in the uniform 40% margin model.

---

### Model Artifact Specifications
* **Active Model Version:** `linucb_v1`
* **Algorithm:** Disjoint LinUCB with Sherman-Morrison rank-1 matrix inversion
* **Context Dimension:** 26 features (7 z-standardized numeric + 1 binary promo + 16 one-hot + 1 bias)
* **Exploration Parameter ($\alpha$):** 1.0 (calibrated against profit standard deviation $r_{\text{scale}} = ₹7.99$)
* **Training Rounds:** 20,000 rounds (Seed 42)

> **Notice:** Prototype developed for research and demonstration purposes. Not intended as direct financial or merchandising advice.
