# -*- coding: utf-8 -*-
"""
Builds a high-impact, professional 22-slide widescreen PowerPoint deck (slides/deck.pptx)
using python-pptx. Embeds native tables, slide figures, clean cards, and speaker notes.
"""

import os
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

os.makedirs("slides", exist_ok=True)

def create_deck():
    print("Building slides/deck.pptx...")
    prs = Presentation()
    # Configure 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # completely blank layout

    # Color palette
    NAVY = RGBColor(15, 23, 42)      # #0F172A
    BLUE = RGBColor(37, 99, 235)     # #2563EB
    EMERALD = RGBColor(16, 185, 129) # #10B981
    SLATE = RGBColor(71, 85, 105)    # #475569
    LIGHT_BG = RGBColor(248, 250, 252)# #F8FAFC
    WHITE = RGBColor(255, 255, 255)
    BORDER_CLR = RGBColor(226, 232, 240) # #E2E8F0
    RED = RGBColor(239, 68, 68)

    # Load speaker notes
    with open("slides/speaker_notes.md", "r", encoding="utf-8") as f:
        notes_raw = f.read()
    
    # Parse notes by slide
    notes_dict = {}
    import re
    slide_sections = re.findall(r"### Slide (\d+):[^\n]*\n([\s\S]*?)(?=### Slide|\Z)", notes_raw)
    for s_num, s_text in slide_sections:
        notes_dict[int(s_num)] = s_text.strip()

    def add_header(slide, title, category="RESEARCH PRESENTATION"):
        # Category pill/tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BLUE

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def set_notes(slide, slide_num):
        if slide_num in notes_dict:
            notes_slide = slide.notes_slide
            tf = notes_slide.notes_text_frame
            tf.text = notes_dict[slide_num]

    # -------------------------------------------------------------
    # Slide 1: Title Slide
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 1)
    
    # Title Box
    tbox = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "PERSONALIZED DYNAMIC DISCOUNT RECOMMENDATION"
    p0.font.size = Pt(36)
    p0.font.bold = True
    p0.font.color.rgb = NAVY
    
    p1 = tf.add_paragraph()
    p1.text = "Contextual Multi-Armed Bandits with Commercial Guardrails & Financial Ledgers"
    p1.font.size = Pt(22)
    p1.font.color.rgb = BLUE
    p1.font.bold = True
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Final Year Technical Presentation • Machine Learning, Bandits & Pricing Optimization\nEvaluated over 20,000 Online Rounds, 5,000 Frozen Rounds & 73,100 Retail Records"
    p2.font.size = Pt(15)
    p2.font.color.rgb = SLATE
    p2.space_before = Pt(24)

    # Bottom pill badges
    badges = [("LinUCB & Thompson Sampling", BLUE), ("Hard Margin & Tier Guardrails", EMERALD), ("10-Seed Paired Benchmarks", NAVY)]
    left_start = Inches(1.0)
    for b_text, b_col in badges:
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_start, Inches(5.8), Inches(3.4), Inches(0.55))
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BG
        shape.line.color.rgb = b_col
        shape.line.width = Pt(1.5)
        tf_b = shape.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = b_text
        p_b.font.size = Pt(12)
        p_b.font.bold = True
        p_b.font.color.rgb = b_col
        p_b.alignment = PP_ALIGN.CENTER
        left_start += Inches(3.6)

    # -------------------------------------------------------------
    # Slide 2: The Retail Dilemma
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 2)
    add_header(slide, "The Retail Dilemma: Revenue Volume vs. Profit Margin", "BUSINESS MOTIVATION")
    
    # 2 Cards: Left Dilemma, Right Objective
    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "THE PROMOTIONAL DILEMMA"
    p1.font.size = Pt(16)
    p1.font.bold = True
    p1.font.color.rgb = RED
    
    bullets1 = [
        "Blanket Promotions Destroy Margin: Uniform 20% markdowns give discounts to shoppers who would happily pay full price.",
        "Deadweight Loss: Conceding margin on loyal customers generates zero incremental volume.",
        "Cart Abandonment: Denying discounts to price-sensitive shoppers forfeits sales.",
        "Static Rules Fail: Category-level rules cannot react to real-time competitor prices or inventory velocity."
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = BORDER_CLR
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "PROJECT OBJECTIVE"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = EMERALD

    bullets2 = [
        "Personalized Recommendation: Learn price elasticity dynamically across 5 discount levels: {0%, 5%, 10%, 15%, 20%}.",
        "Hard Commercial Guardrails: Programmatically guarantee zero below-cost sales and respect loyalty tier ceilings.",
        "Net Profit Reward: Optimize strictly for net profit cash flow rather than deceptive vanity revenue.",
        "Frontline Explainability: Provide merchandisers with transparent score decompositions and what-if simulation."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    # -------------------------------------------------------------
    # Slide 3: Why Contextual Bandits?
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 3)
    add_header(slide, "Why Contextual Bandits? Solving Exploration vs. Exploitation", "METHODOLOGY")

    cards = [
        ("Static A/B Testing", "High Opportunity Cost", "Forces 50% traffic to underperforming prices for weeks to achieve significance; millions lost in margin.", RED),
        ("Full Reinforcement Learning", "Unnecessary Complexity", "Full RL assumes Markovian state transitions (s_t+1 ~ P(s_t, a_t)). Retail transactions are exogenous; customer arrivals do not depend on past discounts.", SLATE),
        ("Contextual Bandits", "The Optimal Balance", "Condition actions on context x_t, explore price elasticity optimistically, bound cumulative regret, and achieve high sample efficiency.", BLUE)
    ]
    c_left = Inches(0.8)
    for title, subtitle, desc, col in cards:
        c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, Inches(1.8), Inches(3.6), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = LIGHT_BG
        c.line.color.rgb = col
        c.line.width = Pt(2)
        tf = c.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = col
        
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.size = Pt(13)
        p_sub.font.bold = True
        p_sub.font.color.rgb = SLATE
        p_sub.space_before = Pt(4)
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(14)
        p_desc.font.color.rgb = NAVY
        p_desc.space_before = Pt(16)
        c_left += Inches(3.9)

    # -------------------------------------------------------------
    # Slide 4: Algorithmic Architecture
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 4)
    add_header(slide, "Algorithmic Architecture: LinUCB & Thompson Sampling", "ALGORITHMS")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BLUE
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "LinUCB (with Sherman-Morrison)"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = BLUE

    lin_pts = [
        "Optimistic Score: p_{t,a} = x_t^T theta_a + alpha * sqrt(x_t^T A_a^-1 x_t)",
        "Exploration vs. Exploitation: Expected profit estimate + uncertainty bonus.",
        "Sherman-Morrison Inversion: Updates inverse covariance directly in O(d^2) instead of O(d^3), cutting update latency to microseconds.",
        "Hyperparameters: alpha = 0.5, context dimension d = 20."
    ]
    for b in lin_pts:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = RGBColor(139, 92, 246)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Contextual Thompson Sampling"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(139, 92, 246)

    ts_pts = [
        "Bayesian Formulation: Maintains conjugate Gaussian posterior over arm parameters theta_a ~ N(theta_hat_a, v^2 B_a^-1).",
        "Probability Matching: Draws random parameter vector tilde_theta_a and selects greedy action.",
        "Natural Exploration: Wide posterior dispersion explores uncertain arms; tight posterior exploits confident arms.",
        "Hyperparameters: Noise multiplier sigma = 0.5, spherical prior."
    ]
    for b in ts_pts:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    # -------------------------------------------------------------
    # Slide 5: The Commercial Safety Envelope
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 5)
    add_header(slide, "The Commercial Safety Envelope: Hard Guardrails", "SYSTEM SAFETY")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "GUARDRAIL ARCHITECTURE"
    p1.font.size = Pt(16)
    p1.font.bold = True
    p1.font.color.rgb = EMERALD

    pts1 = [
        "Deterministic Pruning: Filter action space A to eligible subset A_t before bandit scoring.",
        "Loyalty Tier Caps: New customer max 5%; Regular max 15%; Loyal max 20%.",
        "Unit Cost Margin Floor: DiscountedPrice >= CostPrice + MarginFloor.",
        "Safe Fallback: If all promotional arms are disqualified, defaults cleanly to 0% full price."
    ]
    for b in pts1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = BORDER_CLR
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "HISTORICAL AUDIT & COMPLIANCE"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = NAVY

    pts2 = [
        "100% Online Compliance: Exactly 0 margin floor breaches and 0 tier violations across 200,000 rounds.",
        "39.04% Historical Violation Rate: 39% of historical Kaggle records violate tier rules.",
        "Why Hard Rules Win: Historical discounts were uncoordinated; machine learning without hard guardrails will replicate bad historical practices."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    # -------------------------------------------------------------
    # Slide 6: Data Pipeline & Strict Boundary
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 6)
    add_header(slide, "Data Pipeline & Strict Scientific Boundary", "DATA FOUNDATION")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BLUE
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "PUBLIC RETAIL DATASET (Kaggle)"
    p1.font.size = Pt(16)
    p1.font.bold = True
    p1.font.color.rgb = BLUE

    pts1 = [
        "73,100 Clean Records: Store/product/day observations across multiple retail categories.",
        "Observable Contexts: Shelf BasePrice, CompetitorPrice, Stock levels, Historical Demand.",
        "Derived Features: PriceGapPct, StockCoverageDays, DemandToInventoryRatio, Cyclical month/day.",
        "No Customer Identifiers: Public dataset lacks individual shopper tracking."
    ]
    for b in pts1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = RED
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "SIMULATED COMPONENTS (Strict Honesty)"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = RED

    pts2 = [
        "8,000 Synthetic Customers: Longitudinal RFM profiles (New, Regular, Loyal) simulated by design.",
        "Purchase Outcomes: Stochastic conversion and basket sizes generated via simulator.",
        "Assumed 40% Margin Rate: CostPrice = 0.60 * BasePrice; no real wholesale supplier invoices.",
        "Simulated Rupees (Rs): All financial figures represent simulated currency units."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = NAVY
        p.space_before = Pt(12)

    # -------------------------------------------------------------
    # Slide 7: Demand Calibration & Simulator
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 7)
    add_header(slide, "Demand Calibration & Simulation Environment", "SIMULATION")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "LOGISTIC DEMAND CALIBRATION (ROC-AUC = 0.6322)"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    pts1 = [
        "Purchase Probability Equation: P(Y=1|x, a) = sigma( beta_0 + beta_disc * a + beta_sens * Sensitivity + beta_gap * PriceGapPct + beta_stock * StockRatio )",
        "Empirically Calibrated: Logistic regression fitted to historical demand tertiles yields ROC-AUC of 0.6322.",
        "Realistic Residual Noise: A modest AUC of 0.63 confirms realistic retail variance without artificial deterministic shortcuts.",
        "Omniscient Counterfactual Oracle: Simulator evaluates expected profit across all 5 arms simultaneously, providing exact regret ground truth.",
        "Reproducible Synchronized Sequences: Pre-generated 20,000-round sequences with per-round seeds guarantee identical context arrivals for all policies."
    ]
    for b in pts1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

    # -------------------------------------------------------------
    # Slide 8: Evaluation Methodology
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 8)
    add_header(slide, "Dual Evaluation Protocol: Online vs. Frozen Modes", "EVALUATION")

    cards = [
        ("Online Mode (20k Rounds)", "In-Flight Learning", "Cold-start deployment; actively explores, incurs exploration regret, updates parameters after each round.", BLUE),
        ("Frozen Mode (5k Rounds)", "Operational Deployment", "Pre-trained weights locked; evaluates operational policy on held-out rounds with zero model mutation.", EMERALD),
        ("Multi-Seed Paired Protocol", "Distributional Robustness", "10 independent random seeds (Seeds 42–51). Identical arrivals allow paired lift comparisons and win counts.", NAVY)
    ]
    c_left = Inches(0.8)
    for title, subtitle, desc, col in cards:
        c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, Inches(1.8), Inches(3.6), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = LIGHT_BG
        c.line.color.rgb = col
        c.line.width = Pt(2)
        tf = c.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = col
        
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.size = Pt(13)
        p_sub.font.bold = True
        p_sub.font.color.rgb = SLATE
        p_sub.space_before = Pt(4)
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(14)
        p_desc.font.color.rgb = NAVY
        p_desc.space_before = Pt(16)
        c_left += Inches(3.9)

    # -------------------------------------------------------------
    # Slide 9: Headline Result: The Profit Paradox
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 9)
    add_header(slide, "Headline Result: The Profit Paradox (Fixed 10% Won)", "RESULTS")

    # Add Figure: cumulative_profit_slide.png
    fig_path = "slides/figures/cumulative_profit_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = EMERALD
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "PROFIT OUTCOMES (10 Seeds)"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = EMERALD

    pts2 = [
        "Fixed 10%: Rs 80,265.41 +/- 1,788 (Profit Winner).",
        "LinUCB: Rs 78,924.82 +/- 788 (-1.67% deficit, won 1/10 seeds).",
        "Thompson: Rs 78,856.21 +/- 1,028 (-1.76% deficit, won 2/10 seeds).",
        "Why Heuristic Won: Flat 10% hit the elasticity sweet spot—stimulated conversions without paying the steep margin penalty of 15% or 20% markdowns."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 10: Headline Result: Revenue Leadership
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 10)
    add_header(slide, "Headline Result: LinUCB Led Top-Line Gross Revenue", "RESULTS")

    # Add Figure: profit_vs_discount_scatter_slide.png
    fig_path = "slides/figures/profit_vs_discount_scatter_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = BLUE
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "REVENUE & CONVERSION"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = BLUE

    pts2 = [
        "Revenue Winner: LinUCB achieved Rs 234,276.84 +/- 4,477.",
        "Conversion Lift: LinUCB achieved 24.29% conversion vs. 23.81% for Fixed 10%.",
        "Higher Discount Offered: LinUCB offered 8.45% avg discount vs. 8.09% for Fixed 10%.",
        "Lift vs. Rule-Based: LinUCB beat Rule-based on profit (+Rs 492.50, won in 8 of 10 seeds)."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 11: Learning & Regret Convergence
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 11)
    add_header(slide, "Learning Dynamics & Regret Convergence", "CONVERGENCE")

    # Add Figure: regret_curves_slide.png
    fig_path = "slides/figures/regret_curves_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = NAVY
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "REGRET CONVERGENCE"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = NAVY

    pts2 = [
        "Sublinear Flattening: Regret drops from Rs 0.229/round (early) to Rs 0.092/round (late).",
        "Final Regret (Seed 42): LinUCB Rs 1,754.14; Thompson Sampling Rs 1,075.28.",
        "Oracle Alignment: Thompson matched oracle 62.2% of rounds vs. 16.9% for LinUCB.",
        "Non-Zero Asymptote: Irreducible functional approximation gap because linear models cannot fit sigmoidal demand."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 12: Financial Ledger Dynamics
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 12)
    add_header(slide, "Financial Ledger: Exploration Tax & Break-Even Dynamics", "FINANCIAL VIEW")

    # Add Figure: gain_loss_slide.png
    fig_path = "slides/figures/gain_loss_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = BLUE
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "LEDGER CASH FLOWS"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = BLUE

    pts2 = [
        "Exploration Tax: LinUCB trailed Rule-based during initial 1,500 rounds while testing arms.",
        "Break-Even Crossover: Crossed break-even at Round 3,200 on Seed 42, finishing +Rs 492 ahead.",
        "Cash Trade-Off: LinUCB conceded Rs 24,642 in discounts vs. Rs 20,955 for Fixed 10%.",
        "Ledger Value: Proves financial cash flows divergence that regret curves hide."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 13: Tier Breakdown & Guardrail Audit
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 13)
    add_header(slide, "Loyalty Tier Breakdown & Guardrail Audit", "TIER BREAKDOWN")

    # Create Native Table for Slide 13
    rows = 4
    cols = 6
    left = Inches(0.8)
    top = Inches(1.8)
    width = Inches(11.7)
    height = Inches(3.2)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    t = table_shape.table
    
    headers = ["Loyalty Tier", "Share", "Guardrail Cap", "LinUCB Avg Disc", "LinUCB Profit", "Fixed 10% Profit"]
    for i, h in enumerate(headers):
        cell = t.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    data = [
        ["New", "25%", "Max 5%", "3.11%", "Rs 19,410.22", "Rs 19,652.18"],
        ["Regular", "50%", "Max 15%", "9.84%", "Rs 39,411.50", "Rs 40,188.42"],
        ["Loyal", "25%", "Max 20%", "11.02%", "Rs 20,103.10", "Rs 20,424.81"],
    ]
    for row_idx, r_data in enumerate(data, start=1):
        for col_idx, val in enumerate(r_data):
            cell = t.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13)
                p.font.color.rgb = NAVY
                p.alignment = PP_ALIGN.CENTER

    # Bottom summary box
    c_bot = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5))
    c_bot.fill.solid()
    c_bot.fill.fore_color.rgb = LIGHT_BG
    c_bot.line.color.rgb = EMERALD
    tf_bot = c_bot.text_frame
    tf_bot.word_wrap = True
    p = tf_bot.paragraphs[0]
    p.text = "100% GUARDRAIL COMPLIANCE AUDIT"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p2 = tf_bot.add_paragraph()
    p2.text = "• Exactly 0 margin floor breaches and 0 tier cap violations across 200,000 decisions (100% compliance).\n• In contrast, 39.04% of historical Kaggle records violated tier rules, highlighting past lack of discipline."
    p2.font.size = Pt(13)
    p2.font.color.rgb = NAVY
    p2.space_before = Pt(4)

    # -------------------------------------------------------------
    # Slide 14: Off-Policy Replay Sanity Check
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 14)
    add_header(slide, "Off-Policy Replay on 73,100 Historical Records", "OFF-POLICY REPLAY")

    # Add Figure: replay_comparison_slide.png
    fig_path = "slides/figures/replay_comparison_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))

    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = RED
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "CRITICAL REPLAY CAVEATS"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = RED

    pts2 = [
        "Replay Profit: LinUCB Rs 6.11/round vs. Fixed 10% Rs 5.87/round.",
        "Overlapping CIs: 95% confidence intervals overlap across all policies—no statistical superiority.",
        "Non-Random Logging: Historical discounts were set deliberately by managers, violating independence.",
        "Conclusion: Replay confirms coherence within historical bounds; it does NOT prove real-world lift."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Slide 15: Operational Streamlit Dashboard
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 15)
    add_header(slide, "Frontline Decision Support: The Streamlit Web App", "APPLICATION")

    cards = [
        ("Live Recommendation", "Explainable Inference", "Displays optimal discount, expected profit, and UCB uncertainty bonus. Explains why disqualified arms were blocked.", BLUE),
        ("What-If Simulator", "Margin Sensitivity Curve", "Interactive slider projecting conversion and profit. Surfaces profit-maximizing apex with margin floor warnings.", EMERALD),
        ("Guardrail Audit & Dashboard", "100% Offline Architecture", "Full rule inspector and multi-seed performance charts. Zero network calls, runs locally in sub-millisecond latency.", NAVY)
    ]
    c_left = Inches(0.8)
    for title, subtitle, desc, col in cards:
        c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, Inches(1.8), Inches(3.6), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = LIGHT_BG
        c.line.color.rgb = col
        c.line.width = Pt(2)
        tf = c.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = col
        
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.size = Pt(13)
        p_sub.font.bold = True
        p_sub.font.color.rgb = SLATE
        p_sub.space_before = Pt(4)
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(14)
        p_desc.font.color.rgb = NAVY
        p_desc.space_before = Pt(16)
        c_left += Inches(3.9)

    # -------------------------------------------------------------
    # Slide 16: Key Takeaways & Commercial Conclusion
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 16)
    add_header(slide, "Summary Conclusions & Commercial Takeaways", "CONCLUSIONS")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "KEY TAKEAWAYS FOR RETAIL MERCHANDISERS"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    pts1 = [
        "1. Bandits Drive Top-Line Revenue: LinUCB captured Rs 234.3k revenue and 24.29% conversion, beating standard retail rules in 80% of seeds.",
        "2. Heuristics Can Win on Profit: Fixed 10% earned highest profit (Rs 80.3k) by avoiding excessive margin loss on 15% and 20% markdowns.",
        "3. Hard Guardrails are Essential: Programmatic constraints delivered 100% safety, preventing algorithmic margin destruction.",
        "4. Pair Regret with Financial Ledgers: Theoretical regret proves convergence; double-entry ledgers reveal actual commercial cash flows.",
        "5. Honest Caveats: All results derived within calibrated simulator; physical stores require store-wide discount budgets and competitor reaction modeling."
    ]
    for b in pts1:
        p = tf1.add_paragraph()
        p.text = b
        p.font.size = Pt(15)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

    # -------------------------------------------------------------
    # Backup Slide 17: Alpha Sensitivity
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 17)
    add_header(slide, "Backup: LinUCB Exploration Sensitivity (alpha)", "BACKUP SLIDES")
    fig_path = "slides/figures/alpha_regret_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))
    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = BLUE
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "ALPHA SWEEP INSIGHTS"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = BLUE
    pts2 = [
        "Grid: alpha in {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}.",
        "Separate Tuning Seeds: Swept on Seeds 101–105 to avoid headline seed contamination.",
        "Optimal Region: alpha in [0.1, 1.0] is highly stable; alpha=0.5 chosen.",
        "Over-Exploration: alpha >= 2.0 inflates regret to Rs 1,463."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Backup Slide 18: Reward Penalty Ablation
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 18)
    add_header(slide, "Backup: Reward Penalty Ablation (lambda)", "BACKUP SLIDES")
    fig_path = "slides/figures/ablation_tradeoff_slide.png"
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path, Inches(0.8), Inches(1.8), width=Inches(7.2))
    c2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.8), Inches(4.2), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = RED
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "PRICE OF RESTRAINT"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = RED
    pts2 = [
        "Formulation: R - lambda * (BasePrice * a).",
        "Policy Collapse: lambda = 0.73 collapses avg discount to 4.86% and revenue to Rs 214.5k.",
        "Key Finding: Soft reward penalties act as crude blunt throttles; hard guardrails are structurally superior."
    ]
    for b in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = NAVY
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # Backup Slide 19: Comprehensive Performance Tables
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 19)
    add_header(slide, "Backup: Complete Online & Frozen Benchmark Tables", "BACKUP SLIDES")
    
    rows = 6
    cols = 5
    t_shape = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    t = t_shape.table
    hdrs = ["Policy", "Online Profit (Rs)", "Online Revenue (Rs)", "Frozen Profit (Rs)", "Frozen Revenue (Rs)"]
    for i, h in enumerate(hdrs):
        cell = t.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
            
    tbl_data = [
        ["Fixed 10%", "Rs 80,265 +/- 1,788", "Rs 232,095 +/- 5,212", "Rs 20,137 +/- 645", "Rs 58,299 +/- 1,932"],
        ["LinUCB", "Rs 78,925 +/- 788", "Rs 234,277 +/- 4,477", "Rs 19,737 +/- 670", "Rs 58,465 +/- 2,464"],
        ["Thompson", "Rs 78,856 +/- 1,028", "Rs 232,651 +/- 5,585", "Rs 19,730 +/- 620", "Rs 58,159 +/- 2,116"],
        ["Rule-based", "Rs 78,432 +/- 1,177", "Rs 232,086 +/- 3,581", "Rs 19,667 +/- 884", "Rs 58,313 +/- 2,499"],
        ["Random", "Rs 78,371 +/- 986", "Rs 232,030 +/- 3,355", "Rs 19,376 +/- 651", "Rs 57,513 +/- 2,060"],
    ]
    for r_idx, r in enumerate(tbl_data, start=1):
        for c_idx, val in enumerate(r):
            cell = t.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = NAVY
                p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # Backup Slide 20: Mathematical Derivations
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 20)
    add_header(slide, "Backup: Mathematical Formulations & Updates", "BACKUP SLIDES")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BORDER_CLR
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "CORE MATHEMATICAL FORMULATIONS"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    m_pts = [
        "1. Ridge Regression Payoff: theta_hat_a = (D_a^T D_a + I_d)^-1 D_a^T c_a = A_a^-1 b_a",
        "2. LinUCB Optimistic Score: p_{t,a} = x_t^T theta_hat_a + alpha * sqrt(x_t^T A_a^-1 x_t)",
        "3. Sherman-Morrison Update: M_a <- M_a - (M_a x x^T M_a) / (1 + x^T M_a x), updating inverse directly in O(d^2).",
        "4. Thompson Sampling Posterior: theta_tilde_a ~ N(theta_hat_a, v^2 B_a^-1), where B_a = I_d + sum(x x^T).",
        "5. Cumulative Regret: R(T) = sum_{t=1}^T ( mu_t^* - mu_t(a_t) ), tracking expected profit gap vs. Oracle."
    ]
    for b in m_pts:
        p = tf1.add_paragraph()
        p.text = b
        p.font.size = Pt(15)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

    # -------------------------------------------------------------
    # Backup Slide 21: Full Threats to Validity
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 21)
    add_header(slide, "Backup: Comprehensive Threats to Validity", "BACKUP SLIDES")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = RED
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "LIMITATIONS & VALIDITY AUDIT"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = RED

    v_pts = [
        "Data Limitations: Public Kaggle dataset lacks customer keys; customer profiles are synthetic; CostPrice assumes 40% margin.",
        "Simulation Assumptions: Demand formula is a stylized logistic link with modest AUC (0.6322); same simulator used for training and evaluation.",
        "Modeling Simplifications: Linear payoff cannot fit non-linear sigmoid elasticity; regret flattens to non-zero slope (~Rs 0.09/round).",
        "Evaluation Caveats: Replay logging was non-random with 39% tier violations; 10 seeds sufficient for win counts but not asymptotic p-values.",
        "Deployment Barriers: Real stores require store-wide promotional spend budgets, competitor reaction modeling, and legal dynamic pricing audits."
    ]
    for b in v_pts:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

    # -------------------------------------------------------------
    # Backup Slide 22: Technical Future Scope
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_notes(slide, 22)
    add_header(slide, "Backup: Technical Future Scope", "BACKUP SLIDES")

    c1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = BLUE
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "FUTURE RESEARCH ROADMAP"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = BLUE

    f_pts = [
        "1. NeuralUCB [8]: Deep neural network feature extractors to directly model non-linear sigmoidal demand.",
        "2. Continuous Streaming Feedback: Real-time Kafka-based covariance updates upon live POS transaction completion.",
        "3. Real Identity Datasets: Empirical validation on longitudinal transaction records (e.g. UCI Online Retail dataset).",
        "4. Reward-Formulated Loyalty Policies: Integrating customer lifetime retention value directly into reward functions.",
        "5. Competitor-Reactive Pricing: Dynamic multi-agent game-theoretic pricing models reacting to competitor promotions."
    ]
    for b in f_pts:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = NAVY
        p.space_before = Pt(14)

    prs.save("slides/deck.pptx")
    print("slides/deck.pptx successfully created with 22 slides!")

if __name__ == "__main__":
    create_deck()
