# -*- coding: utf-8 -*-
"""
Generates course-specific submission artifacts for BCSE209L (Machine Learning):
1. reports/DA3_Project_Review_Evaluation_Form.docx (Printable hardcopy form)
2. reports/BCSE209L_DA3_Project_Report.docx (Formatted project report)
3. reports/BCSE209L_DA3_Project_Report.md (Markdown version)
4. slides/DA3_Presentation.pptx (Deck strictly matching the 7 professor headings)
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from pptx import Presentation
from pptx.util import Inches as PptInches, Pt as PptPt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor as PptRGBColor
from pptx.enum.shapes import MSO_SHAPE

os.makedirs("reports", exist_ok=True)
os.makedirs("slides", exist_ok=True)

# -------------------------------------------------------------
# 1. reports/DA3_Project_Review_Evaluation_Form.docx
# -------------------------------------------------------------
def build_evaluation_form():
    print("Generating reports/DA3_Project_Review_Evaluation_Form.docx...")
    doc = docx.Document()

    # Set narrow margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.5)
        s.bottom_margin = Inches(0.5)
        s.left_margin = Inches(0.6)
        s.right_margin = Inches(0.6)

    # Header title
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = p_inst.add_run("VELLORE INSTITUTE OF TECHNOLOGY\nSCHOOL OF COMPUTER SCIENCE AND ENGINEERING / CNI")
    run_inst.bold = True
    run_inst.font.size = Pt(13)

    p_form = doc.add_paragraph()
    p_form.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_form = p_form.add_run("DA3 Project Review Evaluation Form")
    run_form.bold = True
    run_form.font.size = Pt(15)
    run_form.font.color.rgb = RGBColor(15, 23, 42)

    # Meta Table
    meta_table = doc.add_table(rows=4, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    data_meta = [
        [("Course Code & Subject:", True), ("BCSE209L / Machine Learning", False), ("Class Number / Slot:", True), ("CH2026270100894 / B2", False)],
        [("Faculty Name:", True), ("Dr. S. Sridevi, Associate Professor", False), ("School:", True), ("CNI", False)],
        [("Assessment Type:", True), ("Digital Assessment 3 (DA3)", False), ("Total Marks Allocated:", True), ("10 Marks", False)],
        [("Review & Presentation Dates:", True), ("15.10.2026 to 19.10.2026", False), ("Mode of Submission:", True), ("Project Report & Demo Link in MS Teams", False)],
    ]

    for r_idx, row_vals in enumerate(data_meta):
        for c_idx, (text, is_bold) in enumerate(row_vals):
            cell = meta_table.cell(r_idx, c_idx)
            cell.text = text
            p = cell.paragraphs[0]
            p.runs[0].bold = is_bold
            p.runs[0].font.size = Pt(9.5)
            if is_bold:
                shading = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls('w')))
                cell._tc.get_or_add_tcPr().append(shading)

    doc.add_paragraph().paragraph_format.space_before = Pt(4)

    # Project Details Section
    p_proj = doc.add_paragraph()
    p_proj.paragraph_format.space_before = Pt(6)
    r1 = p_proj.add_run("Project Team No: ")
    r1.bold = True
    r1.font.size = Pt(10.5)
    p_proj.add_run("________________________                 ")
    r2 = p_proj.add_run("Project Title: ")
    r2.bold = True
    r2.font.size = Pt(10.5)
    r_title = p_proj.add_run("Personalized Dynamic Discount Recommendation Using Contextual Multi-Armed Bandits")
    r_title.bold = True
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(37, 99, 235)

    doc.add_paragraph().paragraph_format.space_before = Pt(4)

    # Evaluation Rubrics Table
    rubric_desc = doc.add_paragraph()
    rubric_desc.paragraph_format.space_before = Pt(4)
    r_rub = rubric_desc.add_run("Rubric Breakdown:")
    r_rub.bold = True
    r_rub.font.size = Pt(10)
    rubric_desc.add_run("  • Rubric 1 (3 Marks): Appropriate choice of problem statement where ML is needed. Application is need of the hour.\n"
                    "  • Rubric 2 (7 Marks): Specifications of Hardware and Software modules completely given (Report, PPT, Code, & Demo).")

    # Student Scoring Table
    score_table = doc.add_table(rows=5, cols=8)
    score_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = [
        "S.No",
        "Team Member\nReg. No",
        "Team Member\nName",
        "Role in\nProject",
        "Report as per\nFormat",
        "Rubric 1\nML App (3M)",
        "Rubric 2\nModules (7M)",
        "Total Marks\n(10M)"
    ]

    for c_idx, h in enumerate(headers):
        cell = score_table.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        shading = parse_xml(r'<w:shd {} w:fill="E2E8F0"/>'.format(nsdecls('w')))
        cell._tc.get_or_add_tcPr().append(shading)

    for r_idx in range(1, 5):
        score_table.cell(r_idx, 0).text = str(r_idx)
        score_table.cell(r_idx, 0).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for c in range(8):
            p = score_table.cell(r_idx, c).paragraphs[0]
            if len(p.runs) > 0:
                p.runs[0].font.size = Pt(9.5)

    # Checkboxes / Deliverables Checklist
    doc.add_paragraph().paragraph_format.space_before = Pt(8)
    p_check = doc.add_paragraph()
    p_check.add_run("Deliverables Checklist (Tick as Verified):\n").bold = True
    p_check.add_run("[   ] Project Report soft copy submitted in MS Teams (with Hardware & Software Module Specifications)\n")
    p_check.add_run("[   ] Presentation PPT submitted (Objectives, Literature Review, Methodology, Novel Aspects, Results, Conclusion, References)\n")
    p_check.add_run("[   ] Short Video Recording of Demo submitted in MS Teams (1.5 to 2 minutes duration)\n")
    p_check.add_run("[   ] Complete Code (GitHub Repository Link or Zip file)\n")
    p_check.add_run("[   ] Hardcopy of this Evaluation Form submitted to Faculty")

    # Signatures
    doc.add_paragraph().paragraph_format.space_before = Pt(16)
    p_sig = doc.add_paragraph()
    p_sig.add_run("Student Signature: _______________________                  Faculty Signature: _______________________\n")
    p_sig.add_run("Date: _______________________                                      (Dr. S. Sridevi, Associate Professor)")

    doc.save("reports/DA3_Project_Review_Evaluation_Form.docx")
    print("  -> Saved reports/DA3_Project_Review_Evaluation_Form.docx")

# -------------------------------------------------------------
# 2. reports/BCSE209L_DA3_Project_Report.docx & .md
# -------------------------------------------------------------
def build_da3_project_report():
    print("Generating reports/BCSE209L_DA3_Project_Report.docx...")
    doc = docx.Document()

    # Title & Metadata
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_top = p_top.add_run("BCSE209L — MACHINE LEARNING (DA3 PROJECT REPORT)\nCLASS: CH2026270100894 | SLOT: B2 | SCHOOL: CNI\nFACULTY: DR. S. SRIDEVI, ASSOCIATE PROFESSOR\n")
    r_top.bold = True
    r_top.font.size = Pt(11)
    r_top.font.color.rgb = RGBColor(71, 85, 105)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t = p_title.add_run("PERSONALIZED DYNAMIC DISCOUNT RECOMMENDATION USING CONTEXTUAL MULTI-ARMED BANDITS")
    r_t.bold = True
    r_t.font.size = Pt(18)
    r_t.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Reinforcement Learning Optimization with Commercial Safety Guardrails, Financial Ledgers & Offline Observational Replay")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(37, 99, 235)

    doc.add_paragraph().paragraph_format.space_before = Pt(6)

    # Sections mapped directly to Rubrics & Presentation structure
    sections = [
        ("1. Problem Statement & Why ML is Needed (Need of the Hour)", [
            "Promotional discounting is among the most powerful yet commercially risky operational levers in modern retail. While price discounts stimulate customer conversion rates and accelerate inventory velocity, blanket indiscriminate discounting (e.g., '20% off store-wide') severely erodes net unit profit margins and forfeits revenue on shoppers who would gladly convert at full price.",
            "Why Machine Learning is the Need of the Hour: Traditional retail pricing relies on static categorical rules (e.g., department markdowns) or classical A/B testing. Static rules cannot adapt dynamically to multivariate customer price elasticity, real-time competitor price gaps, or fluctuating inventory coverage. Standard supervised ML models predict purchase probability but suffer from selection bias—they only observe historical prices and cannot evaluate counterfactuals. Classical A/B testing incurs immense financial opportunity costs by forcing 50% of traffic to suboptimal prices for prolonged evaluation periods.",
            "Contextual Multi-Armed Bandits (MAB) formulate dynamic pricing as an active, online reinforcement learning problem that optimally balances exploration (learning customer price responsiveness) and exploitation (recommending the profit-maximizing discount), minimizing cumulative regret while preserving business margins."
        ]),
        ("2. Objectives of the Project", [
            "1. Formulate dynamic retail discount recommendation across five discrete promotional actions ({0%, 5%, 10%, 15%, 20%}) as a contextual multi-armed bandit problem.",
            "2. Implement Linear Upper Confidence Bound (LinUCB) with O(d^2) Sherman-Morrison rank-1 matrix updates and Bayesian contextual Thompson Sampling.",
            "3. Enforce deterministic commercial safety guardrails: customer loyalty tier discount ceilings and a strict unit cost margin floor, eliminating below-cost sales by construction.",
            "4. Perform dual-mode evaluation over 10 independent random seeds across 20,000 online rounds (in-flight learning) and 5,000 frozen held-out rounds (operational steady state).",
            "5. Maintain an immutable microeconomic financial transaction ledger tracking per-round revenues, unit margins, and discount concessions.",
            "6. Conduct off-policy replay evaluation across 73,100 historical retail records and deploy an explainable, 100% offline Streamlit decision dashboard."
        ]),
        ("3. Literature Review", [
            "1. Contextual Bandits in Recommendation (Li et al., WWW 2010): Introduced LinUCB with ridge regression parameter estimation and confidence bound derivation via matrix concentration inequalities, establishing sublinear regret bounds.",
            "2. Offline Replay Evaluation (Li et al., WSDM 2011): Established the rejection sampling replay estimator, proving that evaluating candidate policies on un-randomized logged data is asymptotically unbiased under independent logging.",
            "3. Thompson Sampling for Linear Bandits (Agrawal & Goyal, ICML 2013; Chapelle & Li, NeurIPS 2011): Formulated contextual Bayesian posterior sampling with Gaussian priors, establishing near-optimal empirical and theoretical regret.",
            "4. Bandit Theory & Regret Foundations (Lattimore & Szepesvari, Cambridge 2020): Defined cumulative pseudo-regret on expected payoffs rather than noisy realized outcomes to isolate policy quality from stochastic binomial variance.",
            "5. Customer RFM Segmentation Principles (Hughes, 1994): Formalized the Recency, Frequency, Monetary framework for tiered customer lifecycle stewardship, providing the theoretical basis for our loyalty tier guardrails."
        ]),
        ("4. Proposed Methodology", [
            "The proposed architecture consists of four interconnected layers:",
            "A. Context Vector Pipeline: Constructs a 20-dimensional feature vector combining standardized product price, competitor shelf price, price gap percentage, current inventory, stock coverage days, and simulated customer RFM metrics.",
            "B. Commercial Guardrail Envelope: Programmatically filters the full action set A to an eligible subset A_t = get_eligible_actions(context). Enforces New tier (max 5%), Regular tier (max 15%), Loyal tier (max 20%), and unit margin floor (DiscountedPrice >= CostPrice). Defaults safely to {0%} if empty.",
            "C. LinUCB with Sherman-Morrison Inversion: Computes optimistic UCB score p_{t,a} = x_t^T theta_a + alpha * sqrt(x_t^T A_a^-1 x_t). Updates inverse covariance directly in O(d^2) operations using M_a <- M_a - (M_a x x^T M_a)/(1 + x^T M_a x).",
            "D. Contextual Thompson Sampling: Draws parameter samples tilde_theta_a ~ N(theta_hat_a, v^2 B_a^-1) and pulls the greedy arm, providing stochastic Bayesian exploration.",
            "E. Demand Simulation & Oracle: Purchase probability is modeled via a calibrated logistic link P(Y=1|x,a) = sigma(z(x,a)) achieving an ROC-AUC of 0.6322 on historical demand tertiles. Evaluates counterfactuals across all 5 arms simultaneously to compute exact expected regret."
        ]),
        ("5. Specifications of Hardware and Software Modules (Rubric 2 Focus)", [
            "To completely satisfy the specification requirements of Rubric 2, the table below provides the full architectural breakdown of all hardware, system software, and functional software modules implemented in this project."
        ])
    ]

    for title, paras in sections:
        h = doc.add_heading(title, level=1)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        for p_text in paras:
            p = doc.add_paragraph(p_text)
            p.paragraph_format.space_after = Pt(4)

    # Table of Hardware and Software Modules
    mod_table = doc.add_table(rows=10, cols=5)
    mod_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    mod_headers = ["Module / Component", "Type", "Key Technical Specifications", "Inputs & Dependencies", "Outputs & Artifacts"]
    for c_idx, h in enumerate(mod_headers):
        cell = mod_table.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        shading = parse_xml(r'<w:shd {} w:fill="0F172A"/>'.format(nsdecls('w')))
        cell._tc.get_or_add_tcPr().append(shading)
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    specs_data = [
        ["Hardware Environment", "Hardware", "x86_64 Multi-core CPU (Intel/AMD), 16GB RAM, No GPU required", "Standard local machine", "Execution environment"],
        ["Data Processing (`src/simulation`)", "Software", "NumPy 2.2.3, Pandas 2.2.3; 73,100 Kaggle rows cleaning, tertile splits", "Raw Kaggle retail logs", "`processed_retail.csv`, 20D context vector"],
        ["Customer Registry (`registry.py`)", "Software", "8,000 synthetic longitudinal customer profiles; RFM tier allocation", "Customer transaction rules", "Customer profiles (New, Regular, Loyal)"],
        ["Demand Simulator (`simulator.py`)", "Software", "Logistic sigmoid demand calibrated at ROC-AUC 0.6322; counterfactuals", "20D Context, Discount arm", "P(Purchase), Basket, Realized Profit, Oracle"],
        ["Safety Guardrails (`guardrails/rules.py`)", "Software", "Deterministic constraint filter; tier ceilings & unit margin floor", "20D Context, Action space", "Eligible action set A_t (100% compliance)"],
        ["LinUCB Engine (`policies/linucb.py`)", "Software", "Disjoint linear bandit; O(d^2) Sherman-Morrison rank-1 inverse update", "A_t, Context x_t, Reward r_t", "Optimal arm a_t, Score decomposition, Uncertainty"],
        ["Thompson Engine (`policies/thompson.py`)", "Software", "Contextual Bayesian regression; Gaussian posterior sampling (sigma=0.5)", "A_t, Context x_t, Reward r_t", "Sampled arm a_t, Posterior dispersion"],
        ["Evaluation Suite (`src/evaluation`)", "Software", "10-seed paired simulator; Online (20k rounds) & Frozen (5k rounds) modes", "Model checkpoints, Seeds 42-51", "Regret curves, `comparison_online.csv`, ledgers"],
        ["Frontend UI (`app/streamlit_app.py`)", "Software", "Streamlit 1.42.0, Plotly 6.0.0; Decoupled, 100% offline, zero-mutation", "Model .npz, config.json", "Interactive dashboard, What-If simulator, Audits"]
    ]

    for r_idx, r_vals in enumerate(specs_data, start=1):
        for c_idx, val in enumerate(r_vals):
            cell = mod_table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            if len(p.runs) > 0:
                p.runs[0].font.size = Pt(8.5)
            if r_idx % 2 == 1:
                shading = parse_xml(r'<w:shd {} w:fill="F8FAFC"/>'.format(nsdecls('w')))
                cell._tc.get_or_add_tcPr().append(shading)

    # Remaining Sections
    post_sections = [
        ("6. Novel Aspects of the Project", [
            "• Sherman-Morrison Rank-1 Optimization: Bypasses naive O(d^3) matrix inversion, computing exact inverse covariance updates in O(d^2) time to achieve microsecond inference latency.",
            "• Inviolable Programmatic Safety Envelope: Enforces loyalty tier ceilings and unit margin floors prior to model scoring, achieving 100% operational safety across 200,000 online rounds without the catastrophic policy collapse induced by soft reward penalties.",
            "• Dual-Mode Evaluation Protocol: Separates the short-term exploration tax (Online, 20k rounds) from post-training steady-state deployment quality (Frozen, 5k held-out rounds).",
            "• Double-Entry Financial Ledger: Complements theoretical regret metrics with an immutable transaction-level ledger tracking revenue, net margin, and discount concessions.",
            "• Unvarnished Commercial Insight: Discovered that simple heuristics (Fixed 10%) can win on net profit by hitting price elasticity sweet spots, while contextual bandits excel at revenue expansion and conversion lift."
        ]),
        ("7. Results and Discussion", [
            "A. Headline Performance (10 Seeds x 20,000 Rounds Online):",
            "  - Fixed 10% (Heuristic): Earned the highest cumulative profit: Rs 80,265.41 +/- 1,788.01.",
            "  - LinUCB (Contextual Bandit): Achieved Rs 78,924.82 +/- 788.32 (-1.67% deficit vs. Fixed 10%), but generated highest gross revenue (Rs 234,276.84 +/- 4,476.98) and highest conversion rate (24.29% +/- 0.58%).",
            "  - Lift vs. Enterprise Rule-Based: LinUCB achieved +Rs 492.50 +/- 1,022.62 (+0.63%) profit lift, winning in 8 of 10 seeds.",
            "  - Thompson Sampling: Achieved Rs 78,856.21 +/- 1,027.65 (-1.76% vs. Fixed 10%; +Rs 423.88 vs. Rule-based in 6 of 10 seeds).",
            "B. Regret Convergence & Non-Zero Asymptote:",
            "  - LinUCB instantaneous regret dropped by over 58%, flattening to Rs 0.0919/round in final rounds.",
            "  - The regret curve does not flatten to zero because a linear model cannot fit a non-linear sigmoidal demand surface, leaving an irreducible functional approximation gap.",
            "C. Financial Ledger & Break-Even Dynamics:",
            "  - On Seed 42, LinUCB suffered an early exploration deficit of -Rs 280, crossed break-even against Rule-based at Round 3,200, and expanded its profit lead to the end.",
            "D. Guardrail Compliance & Replay Sanity Check:",
            "  - 100% compliance across 200,000 rounds (zero margin floor breaches and zero tier violations).",
            "  - Off-policy replay on 73,100 historical records yielded overlapping 95% confidence intervals (LinUCB Rs 6.11 vs. Fixed 10% Rs 5.87), demonstrating that historical data cannot prove real-world superiority."
        ]),
        ("8. Conclusion", [
            "This project successfully designed, implemented, and validated a complete contextual multi-armed bandit framework for personalized retail discount recommendation. While LinUCB proved superior at top-line revenue expansion and demand stimulation, the heuristic Fixed 10% achieved higher profit by avoiding excessive margin loss on deep markdowns. Integrating deterministic business guardrails is essential to eliminate financial downside risk during autonomous exploration. All results were evaluated within our calibrated simulation environment."
        ]),
        ("9. References", [
            "[1] L. Li, W. Chu, J. Langford, and R. E. Schapire, 'A contextual-bandit approach to personalized news article recommendation,' in Proc. 19th Int. Conf. World Wide Web (WWW), 2010, pp. 661-670.",
            "[2] L. Li, W. Chu, J. Langford, and X. Wang, 'Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms,' in Proc. 4th ACM Int. Conf. Web Search and Data Mining (WSDM), 2011, pp. 297-306.",
            "[3] W. R. Thompson, 'On the likelihood that one unknown probability exceeds another in view of the evidence of two samples,' Biometrika, vol. 25, no. 3-4, pp. 285-294, 1933.",
            "[4] S. Agrawal and N. Goyal, 'Thompson Sampling for contextual bandits with linear payoffs,' in Proc. 30th Int. Conf. Machine Learning (ICML), 2013, pp. 127-135.",
            "[5] O. Chapelle and L. Li, 'An empirical evaluation of Thompson Sampling,' in Adv. Neural Inf. Process. Syst. (NeurIPS 24), 2011, pp. 2249-2257.",
            "[6] T. Lattimore and C. Szepesvari, Bandit Algorithms. Cambridge University Press, 2020.",
            "[7] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.",
            "[8] D. Zhou, L. Li, and Q. Gu, 'Neural contextual bandits with UCB-based exploration,' in Proc. 37th Int. Conf. Machine Learning (ICML), 2020, pp. 11492-11502.",
            "[9] A. M. Hughes, Strategic Database Marketing. Probus Publishing, 1994.",
            "[10] Kaggle Retail Store Inventory and Demand Forecasting Dataset, 2023. [Online]. Available: https://www.kaggle.com/datasets."
        ])
    ]

    for title, paras in post_sections:
        h = doc.add_heading(title, level=1)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        for p_text in paras:
            p = doc.add_paragraph(p_text)
            p.paragraph_format.space_after = Pt(4)

    doc.save("reports/BCSE209L_DA3_Project_Report.docx")
    print("  -> Saved reports/BCSE209L_DA3_Project_Report.docx")

# -------------------------------------------------------------
# 3. slides/DA3_Presentation.pptx (Strict 7-Section Structure)
# -------------------------------------------------------------
def build_da3_ppt():
    print("Generating slides/DA3_Presentation.pptx...")
    prs = Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)
    blank_layout = prs.slide_layouts[6]

    NAVY = PptRGBColor(15, 23, 42)
    BLUE = PptRGBColor(37, 99, 235)
    EMERALD = PptRGBColor(16, 185, 129)
    SLATE = PptRGBColor(71, 85, 105)
    LIGHT_BG = PptRGBColor(248, 250, 252)
    WHITE = PptRGBColor(255, 255, 255)
    BORDER_CLR = PptRGBColor(226, 232, 240)
    RED = PptRGBColor(239, 68, 68)

    def add_da3_header(slide, title, section_num_label):
        cat_box = slide.shapes.add_textbox(PptInches(0.8), PptInches(0.4), PptInches(11.5), PptInches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = section_num_label.upper()
        p_cat.font.size = PptPt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BLUE

        title_box = slide.shapes.add_textbox(PptInches(0.8), PptInches(0.7), PptInches(11.5), PptInches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = PptPt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    # Slide 1: Title Slide (With Course & Review Metadata)
    slide1 = prs.slides.add_slide(blank_layout)
    tbox = slide1.shapes.add_textbox(PptInches(1.0), PptInches(1.2), PptInches(11.3), PptInches(4.5))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    p0 = tf1.paragraphs[0]
    p0.text = "PERSONALIZED DYNAMIC DISCOUNT RECOMMENDATION"
    p0.font.size = PptPt(32)
    p0.font.bold = True
    p0.font.color.rgb = NAVY

    p1 = tf1.add_paragraph()
    p1.text = "Contextual Multi-Armed Bandits with Commercial Guardrails & Financial Ledgers"
    p1.font.size = PptPt(20)
    p1.font.color.rgb = BLUE
    p1.font.bold = True
    p1.space_before = PptPt(8)

    p2 = tf1.add_paragraph()
    p2.text = "Course: BCSE209L — Machine Learning | Digital Assessment 3 (DA3 Project Review)\n" \
              "Class: CH2026270100894 | Slot: B2 | School: CNI\n" \
              "Faculty: Dr. S. Sridevi, Associate Professor\n" \
              "Evaluation Dates: 15.10.2026 to 19.10.2026 | Marks Allocated: 10 Marks"
    p2.font.size = PptPt(14)
    p2.font.color.rgb = SLATE
    p2.space_before = PptPt(20)

    # Slide 2: 1. Objectives
    slide2 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide2, "1. Project Objectives", "RUBRIC SECTION 1")
    c1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(0.8), PptInches(1.8), PptInches(11.7), PptInches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BLUE
    tf2 = c1.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "PRIMARY OBJECTIVES (Need of the Hour Problem)"
    p.font.size = PptPt(18)
    p.font.bold = True
    p.font.color.rgb = NAVY
    objs = [
        "1. Dynamic Pricing Personalization: Personalize discounts across 5 arms ({0%, 5%, 10%, 15%, 20%}) based on 20D context.",
        "2. Solve Exploration vs. Exploitation: Deploy LinUCB and Thompson Sampling to minimize cumulative regret vs. static A/B testing.",
        "3. Guaranteed Safety via Guardrails: Programmatically enforce loyalty tier ceilings and unit margin floors (100% compliance).",
        "4. Dual-Mode Evaluation: Benchmark Online (20,000 rounds learning) vs. Frozen (5,000 rounds operational) over 10 random seeds.",
        "5. Auditable Accounting: Maintain an immutable microeconomic transaction ledger tracking profit, revenue, and discount concessions.",
        "6. Frontline Explainability: Deliver an interactive, 100% offline Streamlit decision dashboard for store merchandisers."
    ]
    for b in objs:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = PptPt(14)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(10)

    # Slide 3: 2. Literature Review
    slide3 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide3, "2. Literature Review", "RUBRIC SECTION 2")
    c1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(0.8), PptInches(1.8), PptInches(11.7), PptInches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf3 = c1.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "KEY FOUNDATIONAL LITERATURE"
    p.font.size = PptPt(18)
    p.font.bold = True
    p.font.color.rgb = NAVY
    lit = [
        "• Li et al. (WWW 2010) — LinUCB: Formulated contextual bandits with linear ridge regression and matrix concentration confidence bounds.",
        "• Li et al. (WSDM 2011) — Replay Evaluation: Proved that rejection sampling on logged data produces an unbiased offline estimator.",
        "• Agrawal & Goyal (ICML 2013) & Chapelle & Li (NeurIPS 2011) — Contextual Thompson Sampling: Evaluated Bayesian posterior sampling with linear payoffs.",
        "• Lattimore & Szepesvari (Cambridge 2020) — Bandit Theory: Grounded cumulative expected regret formulations to isolate policy quality from noise.",
        "• Hughes (1994) — RFM Framework: Customer segmentation by Recency, Frequency, and Monetary value for commercial loyalty tiers."
    ]
    for b in lit:
        p = tf3.add_paragraph()
        p.text = b
        p.font.size = PptPt(14)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(12)

    # Slide 4: 3. Proposed Methodology
    slide4 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide4, "3. Proposed Methodology: Algorithmic Architecture", "RUBRIC SECTION 3")
    c1 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(0.8), PptInches(1.8), PptInches(5.6), PptInches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BLUE
    tf4 = c1.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "LinUCB & Thompson Sampling"
    p.font.size = PptPt(16)
    p.font.bold = True
    p.font.color.rgb = BLUE
    pts_m1 = [
        "LinUCB Upper Confidence Bound: p_{t,a} = x_t^T theta_a + alpha * sqrt(x_t^T A_a^-1 x_t)",
        "Sherman-Morrison Update: M_a <- M_a - (M_a x x^T M_a)/(1 + x^T M_a x), updating inverse directly in O(d^2) operations.",
        "Contextual Thompson Sampling: Samples theta_tilde_a ~ N(theta_hat_a, v^2 B_a^-1) for Bayesian exploration.",
        "Action Space: 5 discrete discount arms {0%, 5%, 10%, 15%, 20%}."
    ]
    for b in pts_m1:
        p = tf4.add_paragraph()
        p.text = "• " + b
        p.font.size = PptPt(13)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(10)

    c2 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(6.8), PptInches(1.8), PptInches(5.7), PptInches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = EMERALD
    tf4b = c2.text_frame
    tf4b.word_wrap = True
    p = tf4b.paragraphs[0]
    p.text = "Safety Guardrails & Simulator"
    p.font.size = PptPt(16)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    pts_m2 = [
        "Inviolable Guardrails: Programmatic filtering restricts action set before scoring: New (max 5%), Regular (max 15%), Loyal (max 20%).",
        "Unit Cost Margin Floor: DiscountedPrice >= CostPrice + MarginFloor.",
        "Calibrated Demand Simulator: Logistic purchase link P(Y=1|x,a) calibrated at ROC-AUC 0.6322 on historical demand tertiles.",
        "Omniscient Oracle: Evaluates expected profit across all 5 arms simultaneously for exact regret computation."
    ]
    for b in pts_m2:
        p = tf4b.add_paragraph()
        p.text = "• " + b
        p.font.size = PptPt(13)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(10)

    # Slide 5: Hardware & Software Module Specifications (Rubric 2 focus)
    slide5 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide5, "Hardware and Software Module Specifications", "RUBRIC 2 COMPLIANCE")
    
    rows, cols = 6, 4
    t_shape = slide5.shapes.add_table(rows, cols, PptInches(0.8), PptInches(1.8), PptInches(11.7), PptInches(4.8))
    t = t_shape.table
    hdrs = ["Module Name", "Type & Technology", "Inputs / Dependencies", "Outputs & Artifacts"]
    for i, h in enumerate(hdrs):
        cell = t.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = PptPt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
            
    mod_data = [
        ["Hardware Platform", "Multi-core x86_64 CPU, 16GB RAM, No GPU", "Local PC environment", "Fast CPU execution (<18s training)"],
        ["Data & Simulator (`src/simulation`)", "Python 3.14, NumPy, Pandas, Scikit-learn", "73,100 Kaggle rows, 8k registry", "20D context vectors, P(Purchase), Oracle"],
        ["Guardrails (`src/guardrails`)", "Deterministic Python filter (`rules.py`)", "20D Context, Action space", "Eligible action set A_t (100% compliance)"],
        ["Bandit Policies (`src/policies`)", "LinUCB (Sherman-Morrison O(d^2)), Thompson", "Eligible set, Context, Reward", "Recommended discount, Uncertainty score"],
        ["Dashboard (`app/streamlit_app.py`)", "Streamlit 1.42.0, Plotly 6.0.0, 100% Offline", "Pre-trained .npz models, config", "Live recommendation, What-If, Audits"]
    ]
    for r_idx, r in enumerate(mod_data, start=1):
        for c_idx, val in enumerate(r):
            cell = t.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = PptPt(11)
                p.font.color.rgb = NAVY
                p.alignment = PP_ALIGN.CENTER

    # Slide 6: 4. Novel Aspects
    slide6 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide6, "4. Novel Aspects of the Project", "RUBRIC SECTION 4")
    cards = [
        ("Sherman-Morrison O(d^2) Updates", "High-Throughput Linear Algebra", "Bypasses naive O(d^3) matrix inversion by updating inverse covariance directly via rank-1 outer products, slashing latency to microseconds.", BLUE),
        ("Inviolable Safety Envelope", "100% Operational Compliance", "Eliminates below-cost sales and loyalty breaches programmatically (0 breaches in 200k decisions), avoiding policy collapse of soft penalty terms.", EMERALD),
        ("Dual-Mode & Ledger Protocol", "Microeconomic Auditability", "Pairs theoretical regret with double-entry ledgers, revealing early exploration tax, Round 3,200 break-even, and gross revenue leadership.", NAVY)
    ]
    c_left = PptInches(0.8)
    for title, subtitle, desc, col in cards:
        c = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, PptInches(1.8), PptInches(3.6), PptInches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = LIGHT_BG
        c.line.color.rgb = col
        c.line.width = PptPt(2)
        tf = c.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = PptPt(17)
        p.font.bold = True
        p.font.color.rgb = col
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.size = PptPt(12)
        p_sub.font.bold = True
        p_sub.font.color.rgb = SLATE
        p_sub.space_before = PptPt(4)
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = PptPt(14)
        p_desc.font.color.rgb = NAVY
        p_desc.space_before = PptPt(14)
        c_left += PptInches(3.9)

    # Slide 7: 5. Results & Discussion: Headline Benchmarks
    slide7 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide7, "5. Results & Discussion: Policy Benchmarks", "RUBRIC SECTION 5")
    fig_path = "slides/figures/cumulative_profit_slide.png"
    if os.path.exists(fig_path):
        slide7.shapes.add_picture(fig_path, PptInches(0.8), PptInches(1.8), width=PptInches(7.2))
    c2 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(8.3), PptInches(1.8), PptInches(4.2), PptInches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = EMERALD
    tf7 = c2.text_frame
    tf7.word_wrap = True
    p = tf7.paragraphs[0]
    p.text = "HEADLINE FINDINGS (10 Seeds)"
    p.font.size = PptPt(16)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    res_pts = [
        "Profit Winner: Fixed 10% earned Rs 80,265.41 +/- 1,788.01 (hit elasticity sweet spot).",
        "Revenue Winner: LinUCB captured highest gross revenue: Rs 234,276.84 +/- 4,476.98.",
        "Conversion Lift: LinUCB achieved 24.29% conversion vs. 23.81% for Fixed 10%.",
        "Lift vs. Retail Rules: LinUCB beat enterprise Rule-based by +Rs 492.50 (won in 8/10 seeds)."
    ]
    for b in res_pts:
        p = tf7.add_paragraph()
        p.text = "• " + b
        p.font.size = PptPt(13)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(10)

    # Slide 8: 5. Results & Discussion: Regret & Break-Even
    slide8 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide8, "5. Results & Discussion: Convergence & Cash Flows", "RUBRIC SECTION 5")
    fig_path = "slides/figures/regret_curves_slide.png"
    if os.path.exists(fig_path):
        slide8.shapes.add_picture(fig_path, PptInches(0.8), PptInches(1.8), width=PptInches(7.2))
    c2 = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(8.3), PptInches(1.8), PptInches(4.2), PptInches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = BLUE
    tf8 = c2.text_frame
    tf8.word_wrap = True
    p = tf8.paragraphs[0]
    p.text = "REGRET & LEDGER DYNAMICS"
    p.font.size = PptPt(16)
    p.font.bold = True
    p.font.color.rgb = BLUE
    dyn_pts = [
        "Sublinear Convergence: Regret drops from Rs 0.229 to Rs 0.092/round (learning flattens).",
        "Non-Zero Asymptote: Linear model cannot fit sigmoidal demand link across 20 dimensions.",
        "Break-Even Crossover: Crossed break-even vs. Rule-based at Round 3,200 on Seed 42.",
        "Volume-Margin Trade-Off: LinUCB conceded Rs 24.6k discounts vs. Rs 20.9k for Fixed 10%."
    ]
    for b in dyn_pts:
        p = tf8.add_paragraph()
        p.text = "• " + b
        p.font.size = PptPt(13)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(10)

    # Slide 9: 6. Conclusion
    slide9 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide9, "6. Conclusion & Commercial Insights", "RUBRIC SECTION 6")
    c1 = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(0.8), PptInches(1.8), PptInches(11.7), PptInches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf9 = c1.text_frame
    tf9.word_wrap = True
    p = tf9.paragraphs[0]
    p.text = "SUMMARY OF CONTRIBUTIONS"
    p.font.size = PptPt(18)
    p.font.bold = True
    p.font.color.rgb = NAVY
    concls = [
        "1. ML Application: Contextual bandits successfully personalize retail discounts, driving top-line revenue to Rs 234.3k and outperforming rule-based baselines in 80% of runs.",
        "2. The Profit Lesson: Heuristic Fixed 10% earned highest profit (Rs 80.3k) by hitting the price elasticity sweet spot without suffering steep unit margin penalties on 15-20% discounts.",
        "3. Hard Guardrails are Mandatory: Deterministic programmatic filtering achieved 100% operational safety (0 breaches across 200k rounds) without the policy collapse of soft penalties.",
        "4. Accounting Integrity: Double-entry financial ledgers revealed cash-flow dynamics that theoretical regret curves hide.",
        "5. Scientific Integrity: Evaluated within our calibrated simulation environment; physical retail deployment requires global store-wide promotional spend budgets."
    ]
    for b in concls:
        p = tf9.add_paragraph()
        p.text = b
        p.font.size = PptPt(15)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(14)

    # Slide 10: 7. References
    slide10 = prs.slides.add_slide(blank_layout)
    add_da3_header(slide10, "7. References", "RUBRIC SECTION 7")
    c1 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(0.8), PptInches(1.8), PptInches(11.7), PptInches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf10 = c1.text_frame
    tf10.word_wrap = True
    p = tf10.paragraphs[0]
    p.text = "BIBLIOGRAPHIC REFERENCES"
    p.font.size = PptPt(18)
    p.font.bold = True
    p.font.color.rgb = NAVY
    refs = [
        "[1] L. Li, W. Chu, J. Langford, and R. E. Schapire, 'A contextual-bandit approach to personalized news article recommendation,' in Proc. 19th Int. Conf. World Wide Web (WWW), 2010, pp. 661-670.",
        "[2] L. Li, W. Chu, J. Langford, and X. Wang, 'Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms,' in Proc. 4th ACM WSDM, 2011, pp. 297-306.",
        "[3] W. R. Thompson, 'On the likelihood that one unknown probability exceeds another in view of the evidence of two samples,' Biometrika, vol. 25, pp. 285-294, 1933.",
        "[4] S. Agrawal and N. Goyal, 'Thompson Sampling for contextual bandits with linear payoffs,' in Proc. 30th ICML, 2013, pp. 127-135.",
        "[5] O. Chapelle and L. Li, 'An empirical evaluation of Thompson Sampling,' in Adv. Neural Inf. Process. Syst. (NeurIPS 24), 2011.",
        "[6] T. Lattimore and C. Szepesvari, Bandit Algorithms. Cambridge University Press, 2020.",
        "[7] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.",
        "[8] D. Zhou, L. Li, and Q. Gu, 'Neural contextual bandits with UCB-based exploration,' in Proc. 37th ICML, 2020.",
        "[9] A. M. Hughes, Strategic Database Marketing. Probus Publishing, 1994.",
        "[10] Kaggle Retail Store Inventory and Demand Forecasting Dataset, 2023. [Online]. Available: https://www.kaggle.com/datasets."
    ]
    for b in refs:
        p = tf10.add_paragraph()
        p.text = b
        p.font.size = PptPt(12)
        p.font.color.rgb = NAVY
        p.space_before = PptPt(6)

    prs.save("slides/DA3_Presentation.pptx")
    print("  -> Saved slides/DA3_Presentation.pptx (Strict 7-section alignment)")

if __name__ == "__main__":
    build_evaluation_form()
    build_da3_project_report()
    build_da3_ppt()
    print("All DA3 course-specific artifacts successfully created.")
