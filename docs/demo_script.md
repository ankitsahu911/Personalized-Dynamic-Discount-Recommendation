# Live Demo Walkthrough Script (3-Minute Presentation)
### Project: NEXUS Commerce — Personalized Dynamic Retail Discount Recommendation Engine

This script provides an exact, step-by-step walkthrough of the Streamlit application for presentations, technical interviews, and stakeholder demos. Every value, slider setting, and metric in this guide has been mathematically pre-computed and verified against the model checkpoint (`linucb_v1`).

---

## ⏱️ At-A-Glance Timeline

| Minute | Stage | Core Concept Shown | Console Page |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:45** | **Merchant Authentication & Store Catalog** | 1-click login, store health counters, interactive product catalog | **Sign In & 🏠 Store Home & Catalog** |
| **0:45 – 1:30** | **Live Inference & UCB Scoring** | 1-click product load, tier restrictions, UCB dot-and-whisker decomposition | **🎯 Live Pricing Simulator** |
| **1:30 – 2:15** | **Market Sensitivity & Guardrails** | Competitor price shock, flip points, margin floor protection | **🔮 Sensitivity Lab & 🛡️ Guardrails** |
| **2:15 – 3:00** | **Offline Audit & Governance** | Multi-seed benchmarks, break-even rounds, transparency | **📊 Performance Audits & ℹ️ Methodology** |

---

## 🚀 Pre-Demo Checklist

1. Open your terminal in the project root:
   ```bash
   cd "C:\Users\anknk\Desktop\Personalized Dynamic Discount Recommendation"
   ```
2. Verify all 20 automated tests pass:
   ```bash
   python -m pytest app/tests -q
   ```
3. Launch the web application:
   ```bash
   streamlit run app/streamlit_app.py
   ```
4. Verify the browser opens at `http://localhost:8501`.

---

## 🎙️ Step 0: Merchant Authentication (0:00 – 0:15)

1. Show the clean Wix/Shopify-style **Merchant Sign In Portal**.
2. Point to the **1-Click Demo Evaluation Box**:
   > *"Notice the enterprise security portal. For live review, we feature a 1-click demo login that provisions instant role-based access as Pricing Operations Director."*
3. Click **"🚀 1-Click Demo Login as Pricing Director"**.
4. The page instantly re-routes into the **Store Home & Catalog**.

---

## 🎙️ Step 1: Store Homepage & Interactive Catalog (0:15 – 0:45)

1. Point to the **Store Health Metric Counters**:
   - Gross Simulated Sales: `₹2,32,591` across 20,000 rounds.
   - Total Conversions: `4,931 Orders` (highest conversion volume across all tested policies).
   - Margin Retained: `84.8%` under hard unit floor enforcement.
2. Scroll to the **Featured Store Catalog Showcase**:
   - Show the 6 real retail products across categories (Sony Headphones, Apple Watch, Herman Miller Task Chair, Blue Tokai Coffee, Patagonia Jacket, LEGO Porsche).
3. **Click "⚡ Optimize Discount for Electronics" on Product 2 (Apple Watch Series 9)**:
   > *"Rather than manually entering parameters, merchandising managers can click any catalog SKU to automatically load its price (₹60.00), competitor benchmark (₹58.00), stock level (500 units), and shopper history into the contextual bandit."*
4. The application seamlessly navigates to **`🎯 Live Pricing Simulator`** with all parameters loaded.

---

## 🎙️ Step 2: Live Pricing Simulator & UCB Decomposition (0:45 – 1:30)

1. **Review Winning Recommendation**:
   - **Shopper Tier**: `Loyal Customer` (12 past orders, 10d recency).
   - **Eligible Menu**: `[10%, 15%, 20%]`. Lower discounts (`0%`, `5%`) are automatically blocked to honor loyalty tiers.
   - **Winning Action**: **`15% Discount`** (Selling Price: `₹51.00`, Unit Margin: `₹15.00`).
   - **UCB Score Breakdown**:
     - `15% Arm`: Total Score **`4.27`** (Point Estimate: `3.00`, Uncertainty Bonus: `1.28`).
     - `10% Arm`: Total Score **`4.23`** (Point Estimate: `2.95`, Uncertainty Bonus: `1.28`).
     - `20% Arm`: Total Score **`0.78`** (Point Estimate: `-0.52`, Uncertainty Bonus: `1.30`).
2. **Point to the Dot-and-Whisker Plotly Chart**:
   - Explain the solid dot (point estimate of expected unit profit) vs the whisker length ($\alpha \sqrt{x^T A^{-1} x}$, the uncertainty credit).
   - Show the red crosses at `y=0` for arms blocked by business rules.
3. **Presenter Script**:
   > *"The bandit scores all eligible actions simultaneously. It selects 15% with a score of 4.27, narrowly beating 10% (4.23). The deterministic narrative confirms this recommendation is driven by expected unit profitability exploitation rather than cold exploration."*

---

## 🎙️ Step 3: Market Sensitivity & What-If Stress Testing (1:30 – 2:00)

1. In the Sidebar, click navigation to **`🔮 Market Sensitivity & What-If Lab`**.
2. Under **Competitor Price Shift**, move the slider to **`-25%`** (simulating a rival flash sale dropping competitor price from `₹58.00` to `₹43.50`).
3. **Point to the Screen**:
   - **Base Case**: `15% Discount` (Score `4.27`).
   - **What-If Scenario**: Dynamically flips to **`10% Discount`** (Score **`7.18`**, Point Estimate jumps to `5.88`).
   - **Look at the Continuous Response Curve**:
     - Trace along the x-axis to show where the transition boundary occurs.
4. **Presenter Script**:
   > *"When a competitor slashes prices by 25%, our relative price gap widens. Rather than over-conceding margin with a 20% cut, LinUCB recognizes the shift in price elasticity and optimizes for 10%, capturing higher net profit (₹7.18)."*

---

## 🎙️ Step 4: Margin Guardrail Solvency Protection (2:00 – 2:30)

1. In the Sidebar, click navigation to **`🛡️ Margin Guardrails Studio`**.
2. Under **Unit Margin Floor**, slide the value from `₹10.00` to **`₹22.00`**.
3. Point to the **Comprehensive 5-Arm Compliance Matrix**:
   - Catalogue price `₹60.00`, Cost `₹36.00`.
   - Post-discount unit margins are:
     - `10% Discount`: `₹18.00` (< ₹22.00 floor) → ❌ **Disqualified**
     - `15% Discount`: `₹15.00` (< ₹22.00 floor) → ❌ **Disqualified**
     - `20% Discount`: `₹12.00` (< ₹22.00 floor) → ❌ **Disqualified**
   - **System Response**: Intercepts the bandit and cleanly defaults to catalog price (`0%`), preventing loss-making sales.
4. **Presenter Script**:
   > *"This demonstrates our safety-critical guardrail. When corporate finance sets a ₹22 margin floor, every discount option is blocked. The hard rule overrides model optimization, guaranteeing financial solvency."*

---

## 🎙️ Step 5: Multi-Seed Offline Audits & Governance (2:30 – 3:00)

1. In the Sidebar, switch to **`📊 Policy Performance & ROI Audits`**.
2. Point to the **Audited Benchmarks**:
   - Total Profit: **`₹76,985`** across 20,000 rounds.
   - Win Rate vs Rule-Based: **`8 out of 10 random seeds`** won by LinUCB.
   - Break-Even Curve: Displays the exact round where LinUCB recoups its initial exploration cost and pulls ahead of baseline rules.
   - Click **Download Online Comparison CSV** to show auditability.
3. Switch to **`ℹ️ Model Architecture & Transparency`** to show the mathematical specifications ($x^T \theta_a + \alpha \sqrt{x^T A_a^{-1} x}$) and decoupled software architecture.

---

## 📸 Presentation Deck Screenshot Targets

1. **Merchant Portal Login Card**: Clean authentication with 1-click demo entry button.
2. **Store Homepage & Catalog Grid**: 6 Wix-style product showcase cards with 1-click optimization buttons.
3. **Recommendation Banner & Dot-and-Whisker Chart**: Selected discount card (`15%`) and Plotly UCB error-bar visualization.
4. **What-If Sensitivity Side-by-Side Cards**: Base vs What-If cards displaying decision flip from 15% to 10%.
5. **Continuous Flip-Point Response Curve**: Crossover curve displaying decision transitions across competitor price shocks.
6. **20k-Round Multi-Seed Benchmark Trajectory**: Multi-policy cumulative profit curves with break-even markers.
