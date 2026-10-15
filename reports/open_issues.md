# Open Issues and Human Verification Checkpoints

*This document records items that require author-level human review before final report submission, in accordance with the Phase 9 Wrap-Up protocol.*

## 1. Dataset Provenance Verification
- **Current Text Representation:** "public retail dataset"
- **Required Action:** Author must visit the Kaggle source page for the dataset (by original author Anirudh Chauhan or the corrected version) and inspect the description.
- **Decision:** If the dataset states it is synthetically generated, update the description to *"a public synthetically generated retail-style dataset"* in the report and slides. If real observational logs, maintain *"public retail dataset"*.

## 2. Bibliographic Citation Details for Reference [10]
- **Current Citation in reports/sections/12_references.md:**
  `[10] Kaggle Retail Dataset: Retail Store Inventory and Demand Forecasting Dataset, Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets. [AUTHOR TO FILL IN: exact title, author, URL and access date...]`
- **Required Action:** Fill in the exact URL, author name, specific dataset title, and download date before compiling final thesis/coursework submission.

## 3. Institution-Specific Formatting & Disclosure
- **Title Page & Declaration:** Insert university-specific declaration of originality and cover page.
- **AI Assistance Disclosure:** If required by your university policy, disclose AI pair-programming assistance (e.g., using Antigravity AI assistant for code scaffolding, automated testing, and LaTeX formatting).
- **Word Count Compliance:** Main technical report body (excluding appendices and references) is approximately 7,200 words, conforming to the default 7,000-word academic target.

## 4. Five Key Numbers to Spot-Check by Hand
1. **LinUCB vs. Fixed 10% Online Profit Deficit:** `-Rs 1,340.59 +/- 1,554.26` (-1.67%) in `results/comparison_lift.csv` (Row 0).
2. **LinUCB vs. Rule-based Online Profit Lift:** `+Rs 492.50 +/- 1,022.62` (+0.63%, 8/10 seeds) in `results/comparison_lift.csv` (Row 1).
3. **LinUCB Online Gross Revenue:** `Rs 234,276.84 +/- 4,476.98` in `results/comparison_online.csv` (Row LinUCB).
4. **Seed 42 Final Cumulative Regret:** `Rs 1,754.14` (LinUCB) vs. `Rs 1,075.28` (Thompson) in `results/regret_checkpoints.csv`.
5. **Historical Loyalty Tier Violation Rate:** `39.04%` in `reports/fairness_and_business_audit.json`.
