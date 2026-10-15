# Dynamic Retail Discount Recommendation — Web Application

A production-grade Streamlit web interface demonstrating contextual multi-armed bandits (**Disjoint LinUCB**) paired with strict financial margin guardrails for retail discount recommendation.

---

## 🌟 Key Features

1. **🎯 Live Recommendation Simulator (Tab 1):**
   - Live context inputs: Retail price, competitor pricing, inventory stock, demand velocity, seasonality, customer order frequency, and recency.
   - **Disjoint LinUCB Scoring:** Real-time calculation of expected profit point estimates and uncertainty bonuses ($\alpha \sqrt{x^T A^{-1} x}$).
   - **Plotly Dot-and-Whisker Visualization:** Distinctly plots expected profit with upward exploration whiskers and clearly demarcates blocked arms.
   - **Deterministic Explanations:** Explains why the winning discount was selected over the runner-up and whether it was driven by margin exploitation or uncertainty exploration.
   - **Strict Guardrails:** Blocks options violating loyalty tier menus or falling below unit margin floors.

2. **🔮 What-If Market Sensitivity Simulator (Tab 2):**
   - Relative sliders for competitor price cuts (-30% to +30%), inventory depletion multipliers (0.25x to 3.0x), and seasonal shifts.
   - Side-by-side comparison cards and per-arm score delta tables.
   - Continuous Plotly response curves showing exactly where recommendations flip.

3. **📊 Audited Performance Dashboard (Tab 3):**
   - High-level KPIs from 20,000-round simulations (LinUCB profit, revenue, discount concession, and lift vs baselines).
   - Interactive Plotly trajectories with policy filters and round-range zoom sliders.
   - Gain/Loss vs comparator curves with computed break-even round annotations.
   - Multi-seed comparison tables with CSV downloads.

4. **ℹ️ Methodology & Transparency (Tab 4):**
   - Complete technical overview, architecture flow, data calibration sources, and stated simulator limitations.

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements-app.txt
```

### 2. Verify / Refresh Application Assets
If running for the first time or refreshing model checkpoints:
```bash
python scripts/build_app_config.py
python scripts/sync_app_assets.py
```

### 3. Run Automated Tests
```bash
python -m pytest app/tests -v
```
*Executes all 18 unit tests, guardrail property tests, golden regression tests, and headless Streamlit AppTest smoke tests.*

### 4. Launch the Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Architecture & File Layout

```text
app/
├── streamlit_app.py      # UI ONLY: Layout, widgets, Plotly figures. Zero model math.
├── recommender.py        # PURE LOGIC: Scoring, guardrails, what-if deltas. Zero Streamlit imports.
├── config.py             # Configuration loader, input schema, currency helpers.
├── app_config.json       # Calibrated ranges, defaults, and model constants.
├── about_text.md         # Embedded methodology documentation.
├── assets/               # Self-contained runtime assets (models & dashboard CSVs).
│   ├── models/           # linucb_v1.npz and linucb_v1_meta.json
│   └── dashboard/        # Precomputed simulation time-series & ledgers
└── tests/                # Automated pytest & AppTest suites.
    ├── test_config.py
    ├── test_recommender.py
    ├── test_guardrails.py
    ├── test_app_smoke.py
    └── test_golden.py
```

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

1. Ensure all changes and `app/assets/` are committed to your GitHub repository.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository, branch `main`, and set Main file path to:
   ```text
   app/streamlit_app.py
   ```
4. Under Advanced Settings, confirm Python version 3.10+ and deploy!
