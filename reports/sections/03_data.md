# 3. Data Pipeline and Feature Formulation

## 3.1 Public Retail Dataset Overview

The empirical foundation of this study utilizes a public retail dataset sourced from Kaggle [10], containing store inventory and demand tracking logs. The dataset comprises 73,100 clean records after filtering, capturing store-product-day level commercial activity across multiple product categories, pricing regimes, inventory balances, and historical demand levels.

Each record represents an aggregated observational snapshot rather than an individual shopping cart transaction. Key primary fields include:
- `Store_ID`: Categorical identifier for retail physical branches.
- `Product_ID`: Unique stock keeping unit (SKU) identifier.
- `Category`: High-level merchandise category (e.g., Groceries, Electronics, Apparel).
- `Base_Price`: Standard un-discounted retail shelf price in rupees (₹).
- `Discount_Applied`: Historical percentage discount recorded for that product-day.
- `Historical_Demand`: Past baseline unit velocity recorded for the product.
- `Current_Stock`: Units available in store inventory at the time of observation.
- `Competitor_Price`: Prevailing market price observed for identical or substitute goods.

## 3.2 Cleaning and Preprocessing Validation

Data cleaning followed a strict validation protocol to ensure mathematical consistency:
1. **Invalid Price Removal:** Verified that all `Base_Price` and `Competitor_Price` entries were strictly positive ($> 0$). No negative or zero prices were identified.
2. **Missing Value Imputation:** Categorical fields had zero missing values. Numerical outliers in stock and demand were capped at the 99.9th percentile to prevent leverage distortion in linear ridge regression.
3. **Discount Discretization:** Historical recorded discounts were mapped to their nearest nominal evaluation arm: $\{0\%, 5\%, 10\%, 15\%, 20\%\}$.

## 3.3 Derived Indicators and Feature Engineering

To provide rich contextual signals for the bandit policies, several domain-specific features were engineered from the raw table:

- **Demand Indicator (Tertile Split):** A categorical demand classification was derived by computing category-specific tertile boundaries on `Historical_Demand`. Products are partitioned into `Low`, `Medium`, and `High` demand bands.
- **Discount Amount & Discounted Price:**
  $$\text{DiscountAmount} = \text{BasePrice} \times a_t$$
  $$\text{DiscountedPrice} = \text{BasePrice} - \text{DiscountAmount}$$
- **Price Gap Percentage:** Relative price positioning versus local competition:
  $$\text{PriceGapPct} = \frac{\text{BasePrice} - \text{CompetitorPrice}}{\text{CompetitorPrice}}$$
  A positive gap indicates that the store is more expensive than the competitor, elevating customer price sensitivity.
- **Demand-to-Inventory Ratio & Stock Coverage Days:**
  $$\text{DemandToInventoryRatio} = \frac{\text{HistoricalDemand}}{\max(1, \text{CurrentStock})}$$
  $$\text{StockCoverageDays} = \frac{\text{CurrentStock}}{\max(1, \text{HistoricalDemand})}$$
- **Inventory Status:** Categorical binning of stock coverage into `Critical Low`, `Balanced`, and `Overstocked`.
- **Cost Price and Unit Margin:** Because real supplier procurement invoices are proprietary and absent from the public dataset, unit cost is modeled under an **assumed 40% margin rate**:
  $$\text{CostPrice} = \text{BasePrice} \times (1 - 0.40) = 0.60 \times \text{BasePrice}$$
  $$\text{UnitMargin} = \text{DiscountedPrice} - \text{CostPrice} = \text{BasePrice} \times (0.40 - a_t)$$
  Under this structure, offering a 20% discount halves the percentage margin from 40% to 20%, illustrating the steep financial cost of excessive discounting.
- **Temporal Calendar Features:** Extracted `DayOfWeek`, `Month`, and binary `IsWeekend` indicators to capture cyclical shopping variations.

## 3.4 Context Vector Formulation

At each decision round $t$, the bandit observes a 20-dimensional numerical context vector $x_t \in \mathbb{R}^{20}$. Table 3.1 details the constituent features, their provenance, and their preprocessing transformations.

**Table 3.1: Complete 20-Dimensional Context Vector Formulation**

| Dimension | Feature Name | Description | Provenance | Data Type / Encoding |
| :---: | :--- | :--- | :--- | :--- |
| $x^{(1)}$ | `BasePrice_scaled` | Standardized retail shelf price | Public Dataset | Continuous (StandardScaler) |
| $x^{(2)}$ | `CompetitorPrice_scaled` | Standardized competitor shelf price | Public Dataset | Continuous (StandardScaler) |
| $x^{(3)}$ | `PriceGapPct` | Relative price gap vs. competitor | Derived | Continuous (StandardScaler) |
| $x^{(4)}$ | `CurrentStock_scaled` | Current store inventory count | Public Dataset | Continuous (StandardScaler) |
| $x^{(5)}$ | `HistoricalDemand_scaled` | Baseline historical unit velocity | Public Dataset | Continuous (StandardScaler) |
| $x^{(6)}$ | `DemandToInventoryRatio` | Ratio of velocity to current inventory | Derived | Continuous (StandardScaler) |
| $x^{(7)}$ | `StockCoverageDays` | Estimated days of inventory remaining | Derived | Continuous (StandardScaler) |
| $x^{(8)}$ | `PriceSensitivity` | Synthetic latent customer elasticity | Simulated | Continuous ($[0.2, 1.0]$) |
| $x^{(9)}$ | `PurchaseCount_scaled` | Customer lifetime transaction frequency | Simulated | Continuous (StandardScaler) |
| $x^{(10)}$ | `RecencyDays_scaled` | Days elapsed since customer's last visit | Simulated | Continuous (StandardScaler) |
| $x^{(11)}$ | `CustomerTier_New` | Indicator: customer lifetime visits $\le 2$ | Simulated | Binary $\{0, 1\}$ |
| $x^{(12)}$ | `CustomerTier_Regular` | Indicator: customer lifetime visits $3 - 9$ | Simulated | Binary $\{0, 1\}$ |
| $x^{(13)}$ | `CustomerTier_Loyal` | Indicator: customer lifetime visits $\ge 10$ | Simulated | Binary $\{0, 1\}$ |
| $x^{(14)}$ | `DemandLevel_Low` | Category demand tertile: Low | Derived | Binary $\{0, 1\}$ |
| $x^{(15)}$ | `DemandLevel_Med` | Category demand tertile: Medium | Derived | Binary $\{0, 1\}$ |
| $x^{(16)}$ | `DemandLevel_High` | Category demand tertile: High | Derived | Binary $\{0, 1\}$ |
| $x^{(17)}$ | `IsWeekend` | Saturday or Sunday indicator | Derived | Binary $\{0, 1\}$ |
| $x^{(18)}$ | `Month_sin` | Cyclical month representation | Derived | Continuous ($[-1, 1]$) |
| $x^{(19)}$ | `Month_cos` | Cyclical month representation | Derived | Continuous ($[-1, 1]$) |
| $x^{(20)}$ | `Bias_Intercept` | Constant unit intercept | Architecture | Constant ($1.0$) |

Continuous numerical features are standardized to zero mean and unit variance using scaling parameters fitted strictly on training observations, preventing data leakage.

## 3.5 What is Real and What is Simulated (Strict Boundary)

To maintain absolute scientific integrity and avoid misrepresenting results, we establish an explicit, unequivocal boundary between real empirical data and simulated components:

1. **The Public Retail Dataset:** The historical dataset consists of 73,100 store/product/day rows from Kaggle [10]. It contains **no individual customer IDs**, no longitudinal user shopping carts, and no counterfactual conversion outcomes at alternative prices. Per project guidelines, this dataset is designated strictly as a "public retail dataset" pending author verification of Kaggle provenance.
2. **Simulated Customer Registry:** The 8,000 synthetic customer profiles—including their persistent customer IDs, assigned loyalty tiers (`New`, `Regular`, `Loyal`), historical purchase counts, recency days, and latent price sensitivity values—are **simulated by design**.
3. **Simulated Basket Value and Purchase Outcomes:** Individual customer shopping sessions, transaction basket sizes (units per purchase), and the binary conversion outcome $Y_t \in \{0, 1\}$ are **simulated by design** via the calibrated simulator.
4. **Assumed Cost Structure:** CostPrice is derived under an **assumed 40% margin rate**. No real wholesale supplier cost data was present in the public dataset.
5. **Simulated Currency:** All financial metrics (revenue, profit, discount conceded) reported in rupees (₹) are **simulated rupees** generated within the simulation environment.
