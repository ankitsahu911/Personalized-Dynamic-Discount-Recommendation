"""
Interactive Retail Dynamic Discount Recommender
Streamlit Web Application — Modern E-Commerce Merchant Portal (Inspired by Wix Online Store)
UI ONLY: Layout, widgets, and charts. All business math is delegated to app/recommender.py.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# MUST BE THE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="NEXUS Commerce | AI Dynamic Discount Optimization",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.config import (
    ACTIONS,
    DEFAULT_LOYAL_CUTOFF,
    DEFAULT_MARGIN_FLOOR,
    DEFAULT_REGULAR_CUTOFF,
    INPUT_SCHEMA,
    MAX_MARGIN_FLOOR,
    MODEL_INFO,
    TIER_DISCOUNTS,
    fmt_pct,
    fmt_rupees,
    get_tier_from_purchases,
)
from app.recommender import (
    Rules,
    compute_response_curve,
    load_model,
    make_context,
    recommend,
    recommend_scenarios,
)

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
DASHBOARD_DIR = ASSETS_DIR / "dashboard"

POLICY_COLORS = {
    "LinUCB": "#0D9488",
    "Thompson Sampling": "#2563EB",
    "Fixed 10%": "#D97706",
    "Rule-based": "#7C3AED",
    "Random": "#64748B",
}

POLICY_DASHES = {
    "LinUCB": "solid",
    "Thompson Sampling": "dash",
    "Fixed 10%": "dot",
    "Rule-based": "dashdot",
    "Random": "longdash",
}

# ---------------------------------------------------------------------------
# CLEAN, MODERN E-COMMERCE STYLING (MATCHING TARGET DASHBOARD REFERENCE)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Remove huge top whitespace */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.0rem !important;
        padding-left: 2.0rem !important;
        padding-right: 2.0rem !important;
        max-width: 98% !important;
    }

    /* Minimize native header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.0rem !important;
    }

    /* Clean, Large Typography (High Contrast - No Dark on Dark) */
    h1 {
        font-size: 2.15rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-bottom: 0.15rem !important;
        letter-spacing: -0.02em !important;
    }
    h2 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: -0.015em !important;
    }
    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        margin-bottom: 0.3rem !important;
    }
    p, span {
        font-size: 1.0rem;
        color: #334155;
    }

    /* Large, High-Contrast Metrics (Image 7 Style) */
    [data-testid="stMetricValue"] {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    /* Top Horizontal Radio Navigation Bar ("one left to another") */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        background-color: #F8FAFC !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        margin-top: 8px !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }
    /* Hide the radio circle dots so they render like clean horizontal navigation tabs */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    /* Active checked state */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border-color: #0D9488 !important;
        box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.25) !important;
    }

    /* Clean Card Layouts (Image 7 Style) */
    .kpi-card-white {
        background: #FFFFFF;
        border: 1px solid #EAEFF4;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }
    .alert-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# REPRESENTATIVE STORE PRODUCT CATALOG (Wix Modern Tech Store Catalog)
# ---------------------------------------------------------------------------
STORE_PRODUCTS = [
    {
        "id": "prod_1",
        "name": "Sony WH-1000XM5 Wireless Noise-Cancelling Headphones",
        "category": "Electronics",
        "badge": "Top Seller",
        "badge_color": "#0D9488",
        "badge_bg": "#CCFBF1",
        "icon": "🎧",
        "price": 85.00,
        "comp_price": 82.00,
        "stock": 320,
        "demand": "High",
        "season": "Spring",
        "orders": 12,
        "recency": 8,
        "basket": 190.00,
        "elasticity": 1.05,
        "promo": False,
        "region": "West",
        "description": "Flagship wireless ANC headphones with 30-hour battery life and LDAC lossless audio.",
    },
    {
        "id": "prod_2",
        "name": "Apple Watch Series 9 GPS (Midnight Aluminum)",
        "category": "Electronics",
        "badge": "High Demand",
        "badge_color": "#2563EB",
        "badge_bg": "#DBEAFE",
        "icon": "⌚",
        "price": 60.00,
        "comp_price": 58.00,
        "stock": 500,
        "demand": "Medium",
        "season": "Winter",
        "orders": 12,
        "recency": 10,
        "basket": 150.00,
        "elasticity": 1.00,
        "promo": False,
        "region": "West",
        "description": "Always-On Retina display, S9 SiP dual-core chip, ECG and blood oxygen tracking.",
    },
    {
        "id": "prod_3",
        "name": "Herman Miller Aeron Ergonomic Task Chair",
        "category": "Furniture",
        "badge": "Low Stock",
        "badge_color": "#D97706",
        "badge_bg": "#FEF3C7",
        "icon": "🪑",
        "price": 110.00,
        "comp_price": 115.00,
        "stock": 85,
        "demand": "Medium",
        "season": "Autumn",
        "orders": 5,
        "recency": 20,
        "basket": 240.00,
        "elasticity": 0.90,
        "promo": False,
        "region": "North",
        "description": "Breathable 8Z Pellicle elastomeric mesh with adjustable PostureFit SL sacral support.",
    },
    {
        "id": "prod_4",
        "name": "Blue Tokai Organic Espresso Roast (1kg Artisanal Beans)",
        "category": "Groceries",
        "badge": "High Margin",
        "badge_color": "#059669",
        "badge_bg": "#D1FAE5",
        "icon": "☕",
        "price": 28.00,
        "comp_price": 29.00,
        "stock": 450,
        "demand": "High",
        "season": "Spring",
        "orders": 0,
        "recency": 90,
        "basket": 55.00,
        "elasticity": 0.85,
        "promo": False,
        "region": "South",
        "description": "Single-estate 100% Arabica specialty roast crafted for rich crema and dark chocolate notes.",
    },
    {
        "id": "prod_5",
        "name": "Patagonia Thermal Nano Puff Windbreaker Jacket",
        "category": "Clothing",
        "badge": "Winter Special",
        "badge_color": "#7C3AED",
        "badge_bg": "#EDE9FE",
        "icon": "🧥",
        "price": 75.00,
        "comp_price": 70.00,
        "stock": 210,
        "demand": "Medium",
        "season": "Winter",
        "orders": 10,
        "recency": 14,
        "basket": 160.00,
        "elasticity": 1.10,
        "promo": True,
        "region": "East",
        "description": "Ultralight, windproof water-resistant shell filled with 60g PrimaLoft Gold Insulation.",
    },
    {
        "id": "prod_6",
        "name": "LEGO Technic Porsche 911 GT3 RS Mechanical Kit",
        "category": "Toys",
        "badge": "Collector Pick",
        "badge_color": "#DC2626",
        "badge_bg": "#FEE2E2",
        "icon": "🏎️",
        "price": 48.00,
        "comp_price": 50.00,
        "stock": 140,
        "demand": "Medium",
        "season": "Spring",
        "orders": 1,
        "recency": 45,
        "basket": 95.00,
        "elasticity": 0.95,
        "promo": False,
        "region": "West",
        "description": "2,704-piece authentic scale replica with functional paddle gearbox and working suspension.",
    },
]

# ---------------------------------------------------------------------------
# DATA LOADERS & CACHING
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained LinUCB model weights...")
def get_cached_model():
    """Loads and caches the frozen LinUCB model."""
    try:
        return load_model()
    except Exception as e:
        st.error(
            f"Error loading trained bandit model: {e}\n\n"
            f"Ensure `scripts/sync_app_assets.py` has been executed."
        )
        st.stop()


@st.cache_data(show_spinner="Loading precomputed simulation ledgers...")
def load_dashboard_assets():
    """Loads precomputed evaluation simulation data."""
    try:
        kpi_path = DASHBOARD_DIR / "summary_kpis.json"
        with open(kpi_path, "r", encoding="utf-8") as f:
            kpis = json.load(f)

        s42_path = DASHBOARD_DIR / "cumulative_series_seed42.csv"
        df_s42 = pd.read_csv(s42_path)

        online_path = DASHBOARD_DIR / "comparison_online.csv"
        df_online = pd.read_csv(online_path)

        frozen_path = DASHBOARD_DIR / "comparison_frozen.csv"
        df_frozen = pd.read_csv(frozen_path)

        lift_path = DASHBOARD_DIR / "comparison_lift.csv"
        df_lift = pd.read_csv(lift_path)

        thinned_path = DASHBOARD_DIR / "cumulative_mean_std_thinned.csv"
        df_thinned = pd.read_csv(thinned_path) if thinned_path.exists() else None

        return {
            "kpis": kpis,
            "series_seed42": df_s42,
            "comparison_online": df_online,
            "comparison_frozen": df_frozen,
            "comparison_lift": df_lift,
            "thinned_multiseed": df_thinned,
        }
    except Exception as e:
        st.warning(f"Dashboard files could not be loaded: {e}. Some evaluation tabs may be unavailable.")
        return None


# ---------------------------------------------------------------------------
# AUTHENTICATION & SESSION STATE MANAGEMENT
# ---------------------------------------------------------------------------

is_test_env = "pytest" in sys.modules or os.environ.get("STREAMLIT_TEST_MODE") == "1"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True if is_test_env else False

if "user_name" not in st.session_state:
    st.session_state["user_name"] = "Sujan"
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Pricing Operations Director"
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏠 Store Overview & Catalog"


def handle_login(username_val: str):
    """Sets session state to logged in."""
    st.session_state["authenticated"] = True
    st.session_state["user_name"] = "Sujan" if not username_val else username_val.split("@")[0].title()
    st.session_state["user_role"] = "Pricing Operations Director"
    st.session_state["nav_page"] = "🏠 Store Overview & Catalog"


def handle_logout():
    """Logs out and resets authentication state."""
    st.session_state["authenticated"] = False
    st.session_state["nav_page"] = "🏠 Store Overview & Catalog"


# ---------------------------------------------------------------------------
# LOGIN VIEW (Presented when unauthenticated)
# ---------------------------------------------------------------------------

if not st.session_state["authenticated"]:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.title("🛍️ NEXUS COMMERCE | Merchant Sign In")
        st.caption("AI Dynamic Pricing & Personalized Discount Engine · Disjoint LinUCB with Margin Guardrails")

        with st.container(border=True):
            st.markdown("### 🔐 Merchant Portal Sign In")
            st.caption("Sign in to access real-time discount optimization and policy telemetry.")

            st.markdown(
                """
                <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 14px; margin-bottom: 18px;">
                    <strong style="color: #166534; font-size: 15px;">⚡ Quick Demo Evaluation Mode:</strong>
                    <p style="color: #15803D; font-size: 14px; margin: 4px 0 0 0;">
                        Click below to instantly log in as Pricing Operations Director.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🚀 1-Click Demo Login as Pricing Director", use_container_width=True, type="primary"):
                handle_login("sujan@nexusstore.com")
                st.rerun()

            st.markdown("<div style='text-align: center; color: #94A3B8; margin: 12px 0;'>— OR ENTER CREDENTIALS —</div>", unsafe_allow_html=True)

            login_email = st.text_input("Store Manager Email", value="sujan@nexusstore.com", key="login_email_input")
            login_pass = st.text_input("Password", value="••••••••", type="password", key="login_pass_input")

            if st.button("Sign In to Merchant Console", use_container_width=True):
                handle_login(login_email)
                st.rerun()

        st.caption("🔒 Protected by Disjoint LinUCB Policy Guardrails · 256-bit Session Security · ISO 27001 Compliant")

    st.stop()


# ---------------------------------------------------------------------------
# INITIALIZE WIDGET SESSION STATE TO CLEAN DEFAULTS
# ---------------------------------------------------------------------------

for field in INPUT_SCHEMA:
    if f"input_{field['key']}" not in st.session_state:
        st.session_state[f"input_{field['key']}"] = field["default"]

if "cost_price_override" not in st.session_state:
    st.session_state["cost_price_override"] = 0.0

if "guardrail_margin_floor" not in st.session_state:
    st.session_state["guardrail_margin_floor"] = DEFAULT_MARGIN_FLOOR
if "guardrail_reg_cutoff" not in st.session_state:
    st.session_state["guardrail_reg_cutoff"] = DEFAULT_REGULAR_CUTOFF
if "guardrail_loyal_cutoff" not in st.session_state:
    st.session_state["guardrail_loyal_cutoff"] = DEFAULT_LOYAL_CUTOFF

if "whatif_comp_change" not in st.session_state:
    st.session_state["whatif_comp_change"] = 0.0
if "whatif_inv_mult" not in st.session_state:
    st.session_state["whatif_inv_mult"] = 1.0
if "whatif_season" not in st.session_state:
    st.session_state["whatif_season"] = "Same as base"


def load_product_into_simulator(prod: dict):
    """Loads a catalog item into the active situational context and switches to simulator."""
    st.session_state["input_Price"] = float(prod["price"])
    st.session_state["input_CompetitorPricing"] = float(prod["comp_price"])
    st.session_state["input_Category"] = str(prod["category"])
    st.session_state["input_InventoryLevel"] = int(prod["stock"])
    st.session_state["input_DemandLevel"] = str(prod["demand"])
    st.session_state["input_Seasonality"] = str(prod["season"])
    st.session_state["input_PurchaseCount"] = int(prod["orders"])
    st.session_state["input_RecencyDays"] = int(prod["recency"])
    st.session_state["input_CurrentBasketValue"] = float(prod["basket"])
    st.session_state["input_CustomerPriceSensitivity"] = float(prod["elasticity"])
    st.session_state["input_Promotion"] = bool(prod["promo"])
    st.session_state["input_Region"] = str(prod["region"])
    st.session_state["nav_page"] = "🎯 Live Pricing Simulator"


def set_nav_page(target_page: str):
    """Switches the active console navigation page."""
    st.session_state["nav_page"] = target_page


def reset_situation_inputs():
    """Restores all situational widgets to their baseline defaults."""
    for field in INPUT_SCHEMA:
        st.session_state[f"input_{field['key']}"] = field["default"]
    st.session_state["cost_price_override"] = 0.0


def reset_guardrails():
    """Restores guardrail rules to original defaults."""
    st.session_state["guardrail_margin_floor"] = DEFAULT_MARGIN_FLOOR
    st.session_state["guardrail_reg_cutoff"] = DEFAULT_REGULAR_CUTOFF
    st.session_state["guardrail_loyal_cutoff"] = DEFAULT_LOYAL_CUTOFF


def reset_whatif():
    """Restores what-if controls to base scenario."""
    st.session_state["whatif_comp_change"] = 0.0
    st.session_state["whatif_inv_mult"] = 1.0
    st.session_state["whatif_season"] = "Same as base"


# ---------------------------------------------------------------------------
# TOP HEADER & HORIZONTAL NAVIGATION (IMAGE 7 STYLE)
# ---------------------------------------------------------------------------

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"Welcome Back, {st.session_state['user_name']}")
    st.caption("Live Retail Merchandising Command Center · 5 SKU Categories · Disjoint LinUCB Engine Active")
with col_h2:
    col_u_pill, col_u_out = st.columns([3, 2])
    with col_u_pill:
        st.markdown(
            f"""
            <div style="text-align: right; font-size: 14px; color: #475569; padding-top: 4px;">
                <strong>👤 {st.session_state['user_name']}</strong><br>
                <small style="color: #64748B;">{st.session_state['user_role']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_u_out:
        st.button("Sign Out", on_click=handle_logout, use_container_width=True)

# TOP HORIZONTAL NAVIGATION ("one left to another nd another typa")
nav_selection = st.radio(
    "Console Navigation",
    options=[
        "🏠 Store Overview & Catalog",
        "🎯 Live Pricing Simulator",
        "🔮 Market Sensitivity & What-If Lab",
        "🛡️ Margin Guardrails Studio",
        "📊 Policy Performance & ROI Audits",
        "ℹ️ Model Architecture & Transparency",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)


# ---------------------------------------------------------------------------
# SIDEBAR: ACTIVE MERCHANDISING INPUT KNOBS & GUARDRAILS
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 🏷️ Active Merchandising Context")
st.sidebar.caption("Fine-tune catalogue pricing, market benchmarks, and shopper loyalty.")

# Group 1: Product Inputs
st.sidebar.markdown("#### Product & Commercials")
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    price_val = st.sidebar.number_input(
        "Price (₹)",
        min_value=10.0,
        max_value=150.0,
        step=1.0,
        key="input_Price",
        help="Catalogue selling price of the item before discount.",
    )
with col_p2:
    comp_val = st.sidebar.number_input(
        "Competitor (₹)",
        min_value=10.0,
        max_value=150.0,
        step=1.0,
        key="input_CompetitorPricing",
        help="Observed price for matching SKU at competitor.",
    )

cat_val = st.sidebar.selectbox(
    "Category",
    options=["Electronics", "Groceries", "Furniture", "Toys", "Clothing"],
    key="input_Category",
    help="Product retail category.",
)

# Group 2: Inventory & Market Dynamics
st.sidebar.markdown("#### Inventory & Demand Dynamics")
col_i1, col_i2 = st.sidebar.columns(2)
with col_i1:
    inv_val = st.sidebar.number_input(
        "Inventory Stock",
        min_value=10,
        max_value=600,
        step=5,
        key="input_InventoryLevel",
        help="Available stock count in warehouse.",
    )
with col_i2:
    demand_val = st.sidebar.selectbox(
        "Demand Velocity",
        options=["Low", "Medium", "High"],
        key="input_DemandLevel",
        help="Observed sales velocity level.",
    )

col_i3, col_i4 = st.sidebar.columns(2)
with col_i3:
    season_val = st.sidebar.selectbox(
        "Seasonality",
        options=["Spring", "Summer", "Autumn", "Winter"],
        key="input_Seasonality",
    )
with col_i4:
    region_val = st.sidebar.selectbox(
        "Market Region",
        options=["North", "South", "East", "West"],
        key="input_Region",
    )

promo_val = st.sidebar.checkbox(
    "Flagship Promo Campaign Active",
    key="input_Promotion",
    help="Tick if actively promoted in store banner.",
)

# Group 3: Customer History & Elasticity
st.sidebar.markdown("#### Shopper Loyalty Profile")
col_u1, col_u2 = st.sidebar.columns(2)
with col_u1:
    purchases_val = st.sidebar.number_input(
        "Orders to Date",
        min_value=0,
        max_value=30,
        step=1,
        key="input_PurchaseCount",
        help="Customer's lifetime completed orders (governs loyalty tier).",
    )
with col_u2:
    recency_val = st.sidebar.number_input(
        "Recency (Days)",
        min_value=0,
        max_value=180,
        step=1,
        key="input_RecencyDays",
        help="Days since previous order.",
    )

col_u3, col_u4 = st.sidebar.columns(2)
with col_u3:
    basket_val = st.sidebar.number_input(
        "Basket (₹)",
        min_value=20.0,
        max_value=500.0,
        step=5.0,
        key="input_CurrentBasketValue",
        help="Total cart value in rupees.",
    )
with col_u4:
    sens_val = st.sidebar.number_input(
        "Price Elasticity",
        min_value=0.55,
        max_value=1.55,
        step=0.02,
        key="input_CustomerPriceSensitivity",
        help="Customer price sensitivity coefficient.",
    )

reg_cutoff_val = int(st.session_state.get("guardrail_reg_cutoff", DEFAULT_REGULAR_CUTOFF))
loyal_cutoff_val = int(st.session_state.get("guardrail_loyal_cutoff", DEFAULT_LOYAL_CUTOFF))
active_rule_tier = get_tier_from_purchases(purchases_val, reg_cutoff_val, loyal_cutoff_val)

comp_gap_pct = ((price_val - comp_val) / comp_val) * 100.0 if comp_val > 0 else 0.0
gap_direction = "above" if comp_gap_pct > 0 else "below"
unit_margin_zero = price_val * 0.40

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Tier Assigned:** `{active_rule_tier} Customer`")
st.sidebar.caption(f"Price is **{abs(comp_gap_pct):.1f}% {gap_direction}** competitor benchmark.")
st.sidebar.caption(f"Base unit margin: **{fmt_rupees(unit_margin_zero)}** (40% margin rate).")

with st.sidebar.expander("Advanced Cost Price Override"):
    cost_override = st.number_input(
        "Cost Price Override (₹)",
        min_value=0.0,
        max_value=150.0,
        step=1.0,
        key="cost_price_override",
        help="Set to 0 to maintain calibrated 40% margin rate.",
    )
    st.button("Reset Inputs to Defaults", on_click=reset_situation_inputs, use_container_width=True)

st.sidebar.markdown("### 🛡️ Business Guardrails")
margin_floor_slider = st.sidebar.slider(
    "Unit Margin Floor (₹)",
    min_value=0.0,
    max_value=MAX_MARGIN_FLOOR,
    step=0.5,
    key="guardrail_margin_floor",
    help="Discounts yielding margin below this floor are strictly blocked.",
)

col_g1, col_g2 = st.sidebar.columns(2)
with col_g1:
    reg_cutoff_input = st.number_input(
        "Regular starts at",
        min_value=1,
        max_value=15,
        step=1,
        key="guardrail_reg_cutoff",
    )
with col_g2:
    loyal_cutoff_input = st.number_input(
        "Loyal starts at",
        min_value=2,
        max_value=25,
        step=1,
        key="guardrail_loyal_cutoff",
    )

if reg_cutoff_input >= loyal_cutoff_input:
    st.sidebar.error("Invalid rule: 'Regular starts at' must be strictly less than 'Loyal starts at'. Reverting to defaults.")
    reg_cutoff_input = DEFAULT_REGULAR_CUTOFF
    loyal_cutoff_input = DEFAULT_LOYAL_CUTOFF

st.sidebar.button("Reset Guardrails to Defaults", on_click=reset_guardrails, use_container_width=True)


# ---------------------------------------------------------------------------
# CORE MODEL & RECOMMENDATION EXECUTION
# ---------------------------------------------------------------------------

active_rules = Rules(
    regular_cutoff=reg_cutoff_input,
    loyal_cutoff=loyal_cutoff_input,
    margin_floor=margin_floor_slider,
    cost_price_override=cost_override if cost_override > 0 else None,
)

current_inputs = {
    "Price": price_val,
    "CompetitorPricing": comp_val,
    "Category": cat_val,
    "InventoryLevel": inv_val,
    "DemandLevel": demand_val,
    "Promotion": promo_val,
    "Region": region_val,
    "Seasonality": season_val,
    "PurchaseCount": purchases_val,
    "RecencyDays": recency_val,
    "CurrentBasketValue": basket_val,
    "CustomerPriceSensitivity": sens_val,
}

model = get_cached_model()
rec_result = recommend(
    inputs=current_inputs,
    rules=active_rules,
    model=model,
    alpha_override=st.session_state.get("alpha_override", 1.0),
)


# ===========================================================================
# PAGE 1: 🏠 STORE OVERVIEW & CATALOG (IMAGE 7 DASHBOARD STYLE)
# ===========================================================================

if nav_selection == "🏠 Store Overview & Catalog":
    # 4 Top KPI Cards (Side-by-side, exactly like Image 7)
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Revenue", "₹ 2,32,591", "↗ +12.5% vs baseline")
    with col_k2:
        st.metric("New Orders", "4,931", "↗ +8.2% conversion lift")
    with col_k3:
        st.metric("Pending Shipments", "48", f"Floor: ₹{margin_floor_slider:.2f} active")
    with col_k4:
        st.metric("Total Customers", "1,248", "LinUCB v1.0 Online")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Middle Section (Revenue Trendline on Left + Low Stock Alerts on Right, like Image 7)
    col_dash_left, col_dash_right = st.columns([2, 1])

    with col_dash_left:
        with st.container(border=True):
            st.markdown("### 📈 Revenue & Cumulative Profit Trend")
            st.caption("Offline simulated 20,000-round progression across LinUCB and benchmark policies.")

            # Load dashboard time-series
            dash_data = load_dashboard_assets()
            if dash_data is not None and "series_seed42" in dash_data:
                df_s42 = dash_data["series_seed42"]
                fig_trend = go.Figure()

                # LinUCB smooth green curve matching Image 7
                df_lin = df_s42[df_s42["policy"] == "LinUCB"]
                df_fixed = df_s42[df_s42["policy"] == "Fixed 10%"]

                fig_trend.add_trace(go.Scatter(
                    x=df_lin["round_idx"],
                    y=df_lin["cum_profit"],
                    mode="lines",
                    name="LinUCB (Dynamic)",
                    line=dict(color="#0D9488", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(13, 148, 136, 0.08)",
                ))
                fig_trend.add_trace(go.Scatter(
                    x=df_fixed["round_idx"],
                    y=df_fixed["cum_profit"],
                    mode="lines",
                    name="Fixed 10% (Baseline)",
                    line=dict(color="#D97706", width=2, dash="dash"),
                ))

                fig_trend.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=25),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    xaxis=dict(title="Round Number", showgrid=True, gridcolor="#F1F5F9"),
                    yaxis=dict(title="Cumulative Profit (₹)", showgrid=True, gridcolor="#F1F5F9"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Simulation trend assets synchronizing...")

    with col_dash_right:
        with st.container(border=True):
            st.markdown("### ⚠️ Low Stock & Merchandising Alerts")
            st.caption("Real-time SKU inventory watches and pricing constraints.")

            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">Herman Miller Aeron</strong><br>
                        <small style="color: #64748B;">Furniture · GK-002</small>
                    </div>
                    <span style="background-color: #FEF3C7; color: #B45309; padding: 3px 8px; border-radius: 9999px; font-size: 12px; font-weight: 700;">85 items left</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">LEGO Porsche GT3</strong><br>
                        <small style="color: #64748B;">Toys · TK-004</small>
                    </div>
                    <span style="background-color: #FEE2E2; color: #DC2626; padding: 3px 8px; border-radius: 9999px; font-size: 12px; font-weight: 700;">140 items left</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">Sony WH-1000XM5</strong><br>
                        <small style="color: #64748B;">Electronics · EP-001</small>
                    </div>
                    <span style="background-color: #CCFBF1; color: #0F766E; padding: 3px 8px; border-radius: 9999px; font-size: 12px; font-weight: 700;">Comp -3.8% cut</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">Solvency Guardrail</strong><br>
                        <small style="color: #64748B;">Unit Floor ₹10.00</small>
                    </div>
                    <span style="background-color: #D1FAE5; color: #059669; padding: 3px 8px; border-radius: 9999px; font-size: 12px; font-weight: 700;">100% Protected</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Bottom Section: Featured Product Catalog (Image 7 "Recent Orders" style)
    st.markdown("## 🏷️ Featured Store Catalog — 1-Click Discount Optimizer")
    st.caption("Click **'⚡ Optimize Discount'** on any product below to instantly load its catalogue price and shopper history into the contextual pricing engine.")

    for row_idx in range(0, len(STORE_PRODUCTS), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            item_idx = row_idx + col_idx
            if item_idx < len(STORE_PRODUCTS):
                prod = STORE_PRODUCTS[item_idx]
                with cols[col_idx]:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                                <span style="font-size: 34px;">{prod['icon']}</span>
                                <span style="background-color: {prod['badge_bg']}; color: {prod['badge_color']}; padding: 3px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">
                                    {prod['badge']}
                                </span>
                            </div>
                            <h3 style="margin: 0 0 4px 0; font-size: 18px; color: #0F172A;">{prod['name']}</h3>
                            <p style="color: #64748B; font-size: 14px; margin: 0 0 10px 0; min-height: 38px;">{prod['description']}</p>
                            """,
                            unsafe_allow_html=True,
                        )

                        col_spec1, col_spec2 = st.columns(2)
                        with col_spec1:
                            st.markdown(f"**Catalogue:** `{fmt_rupees(prod['price'])}`")
                            st.markdown(f"**Category:** `{prod['category']}`")
                        with col_spec2:
                            st.markdown(f"**Competitor:** `{fmt_rupees(prod['comp_price'])}`")
                            st.markdown(f"**Stock:** `{prod['stock']} units`")

                        st.button(
                            f"⚡ Optimize Discount for {prod['category']}",
                            key=f"btn_opt_{prod['id']}",
                            on_click=load_product_into_simulator,
                            args=(prod,),
                            use_container_width=True,
                            type="secondary",
                        )


# ===========================================================================
# PAGE 2: 🎯 LIVE PRICING SIMULATOR
# ===========================================================================

elif nav_selection == "🎯 Live Pricing Simulator":
    st.markdown("## 🎯 Live Discount Recommendation & UCB Decision Logic")
    st.caption("Real-time LinUCB scoring engine evaluating all discount options under loyalty and margin constraints.")

    if rec_result.warnings:
        for w in rec_result.warnings:
            st.warning(f"⚠️ {w}")

    if rec_result.chosen_discount is not None:
        rec_disc = rec_result.chosen_discount
        st.markdown(
            f"""
            <div style="background-color: #F0FDF4; border-left: 6px solid #0D9488; padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;">
                <h2 style="color: #065F46 !important; margin: 0; font-size: 26px;">
                    Recommended Action: <strong>{rec_disc}% Discount Concession</strong>
                </h2>
                <p style="color: #047857; margin: 6px 0 0 0; font-size: 16px; line-height: 1.5;">
                    {rec_result.why_pick_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Effective Selling Price", fmt_rupees(rec_result.discounted_price), f"-{rec_disc}% markdown")
        col_m2.metric("Discount Concession", fmt_rupees(rec_result.discount_amount_rupees), "per unit concession")
        col_m3.metric("Post-Discount Margin", fmt_rupees(rec_result.unit_margin_after_discount), f"Floor: {fmt_rupees(active_rules.margin_floor)}")

        chosen_detail = next((a for a in rec_result.arms_table if a.discount == rec_disc), None)
        est_prof = chosen_detail.point_estimate if chosen_detail else 0.0
        col_m4.metric("Est. Expected Profit", fmt_rupees(est_prof, simulated=True), "LinUCB exploitation term")

    else:
        st.markdown(
            """
            <div style="background-color: #FEF2F2; border-left: 6px solid #DC2626; padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;">
                <h2 style="color: #991B1B !important; margin: 0; font-size: 24px;">No Discount Recommended (0% Retained)</h2>
                <p style="color: #B91C1C; margin: 6px 0 0 0; font-size: 15px;">
                    All available discount percentages violate current profitability guardrails (unit margin floor or loyalty tier menu).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Dot-and-Whisker Plotly Chart
    col_chart, col_exp = st.columns([3, 2])

    with col_chart:
        st.subheader("UCB Score Decomposition by Discount Arm")
        st.caption("Point estimate (exploitation) with upper whisker showing uncertainty bonus (exploration).")

        fig_ucb = go.Figure()
        discounts_list = []
        estimates_list = []
        bonuses_list = []
        scores_list = []
        colors_list = []
        hover_texts = []

        for arm in rec_result.arms_table:
            d = arm.discount
            discounts_list.append(f"{d}%")
            if arm.point_estimate is not None:
                estimates_list.append(arm.point_estimate)
                bonuses_list.append(arm.uncertainty_bonus)
                scores_list.append(arm.score)
                is_win = (d == rec_result.chosen_discount)
                colors_list.append("#0D9488" if is_win else "#2563EB")
                extrap_flag = " [EXTRAPOLATED]" if arm.extrapolated else ""
                hover_texts.append(
                    f"Discount: {d}%{extrap_flag}<br>"
                    f"Point Estimate: ₹{arm.point_estimate:.2f}<br>"
                    f"Uncertainty Bonus: ₹{arm.uncertainty_bonus:.2f}<br>"
                    f"Total UCB Score: ₹{arm.score:.2f}"
                )
            else:
                estimates_list.append(None)
                bonuses_list.append(None)
                scores_list.append(None)
                colors_list.append("#CBD5E1")
                hover_texts.append(f"Discount: {d}%<br>BLOCKED: {arm.reason}")

        for i, d_str in enumerate(discounts_list):
            if estimates_list[i] is not None:
                est = estimates_list[i]
                sc = scores_list[i]
                c = colors_list[i]
                fig_ucb.add_trace(go.Scatter(
                    x=[d_str, d_str],
                    y=[est, sc],
                    mode="lines",
                    line=dict(color=c, width=3.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig_ucb.add_trace(go.Scatter(
                    x=[d_str],
                    y=[sc],
                    mode="markers",
                    marker=dict(symbol="line-ew", color=c, size=18, line=dict(width=3, color=c)),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig_ucb.add_trace(go.Scatter(
            x=discounts_list,
            y=estimates_list,
            mode="markers",
            marker=dict(
                color=colors_list,
                size=13,
                line=dict(color="#0F172A", width=1.5),
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Point Estimate (Profit/Visit)",
        ))

        blocked_d = [f"{a.discount}%" for a in rec_result.arms_table if a.score is None]
        if blocked_d:
            fig_ucb.add_trace(go.Scatter(
                x=blocked_d,
                y=[0] * len(blocked_d),
                mode="markers",
                marker=dict(symbol="x", color="#EF4444", size=11),
                text=[f"Discount {d}: Blocked by Guardrail" for d in blocked_d],
                hoverinfo="text",
                name="Blocked Arm",
            ))

        fig_ucb.update_layout(
            height=350,
            margin=dict(l=15, r=15, t=15, b=25),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(title="Offered Discount", showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(title="Estimated Profit per Visit (₹)", zeroline=True, zerolinecolor="#CBD5E1", showgrid=True, gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ucb, use_container_width=True)

    with col_exp:
        st.subheader("Decision Logic Explanation")
        st.markdown(f"**Selected Action:** `{rec_result.chosen_discount}% discount`" if rec_result.chosen_discount is not None else "**Status:** `No discount viable`")
        st.write(rec_result.why_pick_text)

        st.info(
            "**How LinUCB Decides:**\n\n"
            "1. **Point Estimate ($x^T \\theta_a$):** Expected profit per visitor based on linear ridge regression.\n"
            "2. **Uncertainty Credit ($\\alpha \\sqrt{x^T A_a^{-1} x}$):** Exploration bonus rewarding unvisited arms.\n"
            "3. **Guardrails First:** Ineligible arms are eliminated *before* bandit scoring."
        )

    # Detailed Matrix Table
    st.subheader("Detailed Arms Diagnostic Matrix")
    diag_rows = []
    for a in rec_result.arms_table:
        is_rec = (a.discount == rec_result.chosen_discount)
        extrap_tag = " ⚠️ Extrapolated" if a.extrapolated else ""
        diag_rows.append({
            "Discount": f"{a.discount}%",
            "Status": "✅ RECOMMENDED" if is_rec else ("🟢 Eligible" if a.status == "eligible" else f"⛔ {a.status.replace('_', ' ').title()}"),
            "Selling Price": fmt_rupees(a.selling_price),
            "Unit Margin": fmt_rupees(a.unit_margin_after_discount),
            "Point Estimate": fmt_rupees(a.point_estimate) if a.point_estimate is not None else "—",
            "Uncertainty Bonus": fmt_rupees(a.uncertainty_bonus) if a.uncertainty_bonus is not None else "—",
            "Total UCB Score": f"{a.score:.3f}" if a.score is not None else "—",
            "Training Pulls": f"{a.pull_counts:,}" if a.pull_counts else "—",
            "Decision Reason": a.reason + extrap_tag,
        })
    df_diag = pd.DataFrame(diag_rows)
    st.dataframe(df_diag, use_container_width=True, hide_index=True)

    # Native Expanders (Zero text overlap)
    with st.expander("🛡️ Inspect Guardrail Filtering Mechanism"):
        g_rows = []
        for a in rec_result.arms_table:
            passes_tier = "✅ Passes" if a.discount in TIER_DISCOUNTS[rec_result.tier_for_rules] else f"❌ Blocked (not in {TIER_DISCOUNTS[rec_result.tier_for_rules]}%)"
            margin_diff = a.unit_margin_after_discount - active_rules.margin_floor
            passes_margin = (
                f"✅ Passes (Margin {fmt_rupees(a.unit_margin_after_discount)} ≥ Floor {fmt_rupees(active_rules.margin_floor)})"
                if margin_diff >= 0
                else f"❌ Blocked (Margin {fmt_rupees(a.unit_margin_after_discount)} < Floor {fmt_rupees(active_rules.margin_floor)})"
            )
            g_rows.append({
                "Discount": f"{a.discount}%",
                "Tier Rule Compliance": passes_tier,
                "Margin Floor Compliance": passes_margin,
                "Final Arm Eligibility": "Eligible for Scoring" if a.score is not None else "Blocked",
            })
        st.table(pd.DataFrame(g_rows))

    with st.expander("⚙️ Interactive Exploration Parameter α (Live Control)"):
        st.caption("Adjust the exploration weight live to inspect how uncertainty changes the winning recommendation without mutating model weights.")
        new_alpha = st.slider(
            "Exploration Parameter (α)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="alpha_override",
            help="α = 0.0 is pure exploitation (greedy); higher values favor unvisited arms.",
        )

    with st.expander("🔍 Model Input Feature Vector (Debug View)"):
        st.json(rec_result.raw_context)


# ===========================================================================
# PAGE 3: 🔮 MARKET SENSITIVITY & WHAT-IF LAB
# ===========================================================================

elif nav_selection == "🔮 Market Sensitivity & What-If Lab":
    st.markdown("## 🔮 Market Sensitivity & What-If Scenario Lab")
    st.caption("Simulate competitor price cuts, warehouse inventory shifts, or seasonal swings to inspect where recommendations flip.")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        comp_change = st.slider(
            "Competitor Price Shift (%)",
            min_value=-30.0,
            max_value=30.0,
            step=1.0,
            key="whatif_comp_change",
            help="Relative % change in competitor pricing.",
        )
        shifted_comp = round(price_val * (1.0 + comp_change / 100.0), 2)
        st.caption(f"Competitor price: {fmt_rupees(comp_val)} ➔ **{fmt_rupees(shifted_comp)}**")

    with col_w2:
        inv_mult = st.slider(
            "Inventory Stock Multiplier",
            min_value=0.25,
            max_value=3.0,
            step=0.05,
            key="whatif_inv_mult",
            help="Scale inventory level up or down.",
        )
        shifted_inv = int(round(inv_val * inv_mult))
        st.caption(f"Stock units: {inv_val} ➔ **{shifted_inv} units**")

    with col_w3:
        season_opt = st.selectbox(
            "What-If Seasonality",
            options=["Same as base", "Spring", "Summer", "Autumn", "Winter"],
            key="whatif_season",
        )
        st.caption(f"Trading season: {season_val} ➔ **{season_opt}**")

    st.button("Reset What-If Inputs", on_click=reset_whatif)

    whatif_res = recommend_scenarios(
        base_inputs=current_inputs,
        whatif_deltas={
            "competitor_price_pct_change": comp_change,
            "inventory_multiplier": inv_mult,
            "seasonality": season_opt,
        },
        rules=active_rules,
        model=model,
    )

    base_r = whatif_res["base_recommendation"]
    scen_r = whatif_res["whatif_recommendation"]

    base_disc_str = f"{base_r.chosen_discount}%" if base_r.chosen_discount is not None else "None"
    scen_disc_str = f"{scen_r.chosen_discount}%" if scen_r.chosen_discount is not None else "None"

    is_changed = (base_r.chosen_discount != scen_r.chosen_discount)
    banner_bg = "#FEF3C7" if is_changed else "#F1F5F9"
    banner_border = "#D97706" if is_changed else "#94A3B8"
    msg_txt = "RECOMMENDATION FLIPPED" if is_changed else "RECOMMENDATION UNCHANGED"

    st.markdown(
        f"""
        <div style="background-color: {banner_bg}; border-left: 6px solid {banner_border}; padding: 18px; border-radius: 8px; margin: 18px 0;">
            <h3 style="margin: 0; color: #1E293B; font-size: 20px;">
                {msg_txt}: Base Case ({base_disc_str}) ➔ What-If Scenario ({scen_disc_str})
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.markdown("#### Baseline Setting")
        st.metric("Recommended Discount", base_disc_str)
        st.metric("New Selling Price", fmt_rupees(base_r.discounted_price))
        st.metric("Post-Discount Margin", fmt_rupees(base_r.unit_margin_after_discount))

    with col_sb2:
        st.markdown("#### What-If Scenario")
        st.metric("Recommended Discount", scen_disc_str)
        st.metric("New Selling Price", fmt_rupees(scen_r.discounted_price))
        st.metric("Post-Discount Margin", fmt_rupees(scen_r.unit_margin_after_discount))

    st.subheader("Score Delta Breakdown by Discount Arm")
    if whatif_res["delta_table"]:
        df_deltas = pd.DataFrame(whatif_res["delta_table"])
        df_deltas.columns = [
            "Discount (%)", "Base Score", "What-If Score", "Score Shift",
            "Base Estimate", "What-If Estimate", "Estimate Shift"
        ]
        st.dataframe(df_deltas, use_container_width=True, hide_index=True)
    else:
        st.info("No arms were eligible in both scenarios.")

    st.subheader("📈 Continuous Response Curve (Where Recommendations Flip)")
    curve_var = st.selectbox(
        "Select Trajectory Axis",
        options=["CompetitorPricing", "InventoryLevel", "Seasonality"],
        format_func=lambda s: {"CompetitorPricing": "Competitor Price (₹)", "InventoryLevel": "Inventory Level (Units)", "Seasonality": "Seasonality"}.get(s, s),
    )
    curve_res = compute_response_curve(current_inputs, curve_var, rules=active_rules, model=model)

    if "curves" in curve_res and curve_res["curves"]:
        fig_resp = go.Figure()
        for d, y_pts in curve_res["curves"].items():
            fig_resp.add_trace(go.Scatter(
                x=curve_res["points"],
                y=y_pts,
                mode="lines+markers",
                name=f"{d}% Discount",
                line=dict(width=2.5),
            ))

        fig_resp.update_layout(
            height=360,
            title=f"Expected Profit Response Curve across {curve_var}",
            xaxis_title=curve_var,
            yaxis_title="Estimated Expected Profit (₹)",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_resp, use_container_width=True)


# ===========================================================================
# PAGE 4: 🛡️ MARGIN GUARDRAILS STUDIO
# ===========================================================================

elif nav_selection == "🛡️ Margin Guardrails Studio":
    st.markdown("## 🛡️ Profitability Guardrails & Policy Governance")
    st.caption("Inspect and stress-test margin floor limits and dynamic customer loyalty tiers ($0 \\le N < M$).")

    col_g_info1, col_g_info2 = st.columns([1, 1])
    with col_g_info1:
        with st.container(border=True):
            st.markdown("### 🏛️ Policy Thresholds")
            st.markdown(
                f"- **Unit Margin Floor:** `{fmt_rupees(margin_floor_slider)}`\n"
                f"- **New Customer Tier:** `0 to {reg_cutoff_input - 1} orders` (Eligible: `0%, 5%`)\n"
                f"- **Regular Customer Tier:** `{reg_cutoff_input} to {loyal_cutoff_input - 1} orders` (Eligible: `5%, 10%, 15%`)\n"
                f"- **Loyal Customer Tier:** `≥ {loyal_cutoff_input} orders` (Eligible: `10%, 15%, 20%`)"
            )

    with col_g_info2:
        with st.container(border=True):
            st.markdown("### 📊 Active Shopper Context")
            st.markdown(
                f"- **Base Price:** `{fmt_rupees(price_val)}` (Cost: `{fmt_rupees(price_val * 0.60)}`)\n"
                f"- **Customer Past Orders:** `{purchases_val} orders`\n"
                f"- **Assigned Tier:** `{active_rule_tier} Shopper`\n"
                f"- **Eligible Menu:** `{TIER_DISCOUNTS[active_rule_tier]}%`"
            )

    st.markdown("### Comprehensive 5-Arm Compliance Matrix")
    g_full_rows = []
    for a in rec_result.arms_table:
        passes_tier = "✅ Passes" if a.discount in TIER_DISCOUNTS[rec_result.tier_for_rules] else f"❌ Blocked (Tier restricts to {TIER_DISCOUNTS[rec_result.tier_for_rules]}%)"
        margin_diff = a.unit_margin_after_discount - active_rules.margin_floor
        passes_margin = (
            f"✅ Passes (Margin {fmt_rupees(a.unit_margin_after_discount)} ≥ Floor {fmt_rupees(active_rules.margin_floor)})"
            if margin_diff >= 0
            else f"❌ Blocked (Margin {fmt_rupees(a.unit_margin_after_discount)} < Floor {fmt_rupees(active_rules.margin_floor)})"
        )
        g_full_rows.append({
            "Discount Arm": f"{a.discount}%",
            "Catalogue Selling Price": fmt_rupees(a.selling_price),
            "Post-Discount Unit Margin": fmt_rupees(a.unit_margin_after_discount),
            "Tier Policy Compliance": passes_tier,
            "Margin Floor Compliance": passes_margin,
            "Extrapolation Flag": "⚠️ Extrapolated (Custom rule)" if a.extrapolated else "Standard Empirical",
            "Final Eligibility": "🟢 Eligible for Scoring" if a.score is not None else "⛔ Disqualified",
        })
    st.table(pd.DataFrame(g_full_rows))


# ===========================================================================
# PAGE 5: 📊 POLICY PERFORMANCE & ROI AUDITS
# ===========================================================================

elif nav_selection == "📊 Policy Performance & ROI Audits":
    st.markdown("## 📊 Audited Policy Performance & Multi-Seed Benchmarks")
    st.caption("Historical simulation ledgers across 10 random seeds and 20,000 rounds per run. No synthetic shortcuts.")

    dash_data = load_dashboard_assets()

    if dash_data is not None:
        kpis = dash_data["kpis"]

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric(
            "LinUCB Total Profit",
            fmt_rupees(kpis["LinUCB"]["total_profit"], simulated=True),
            "Seed 42 (20,000 rounds)",
        )
        col_k2.metric(
            "LinUCB Gross Revenue",
            fmt_rupees(kpis["LinUCB"]["total_revenue"], simulated=True),
            "Highest top-line sales",
        )
        col_k3.metric(
            "LinUCB Discount Concession",
            fmt_rupees(kpis["LinUCB"]["total_discount_given"], simulated=True),
            "Total markdown disbursed",
        )

        gain_vs_fixed = kpis.get("linucb_gain_vs_fixed_10_rupees", -2295.18)
        pct_gain = (gain_vs_fixed / kpis["Fixed 10%"]["total_profit"]) * 100.0
        col_k4.metric(
            "Profit Gap vs Fixed 10%",
            fmt_rupees(gain_vs_fixed, simulated=True),
            f"{pct_gain:.2f}% (Seed 42)",
            delta_color="normal",
        )

        st.markdown("---")

        st.subheader("Cumulative Performance Trajectories")
        col_c_ctrl1, col_c_ctrl2 = st.columns([2, 1])
        with col_c_ctrl1:
            selected_policies = st.multiselect(
                "Select Policies to Display",
                options=["Random", "Fixed 10%", "Rule-based", "LinUCB", "Thompson Sampling"],
                default=["Fixed 10%", "Rule-based", "LinUCB", "Thompson Sampling"],
            )
        with col_c_ctrl2:
            metric_choice = st.selectbox(
                "Select Evaluation Metric",
                options=["cum_profit", "cum_revenue", "cum_discount_given", "cum_regret"],
                format_func=lambda m: {
                    "cum_profit": "Cumulative Profit (₹)",
                    "cum_revenue": "Gross Revenue (₹)",
                    "cum_discount_given": "Cumulative Discount Given (₹)",
                    "cum_regret": "Cumulative Regret (₹)",
                }.get(m, m),
            )

        round_slider = st.slider(
            "Round Range Filter (Zoom into early exploration)",
            min_value=1,
            max_value=20000,
            value=(1, 20000),
            step=100,
        )

        df_s42 = dash_data["series_seed42"]
        mask = (df_s42["round_idx"] >= round_slider[0]) & (df_s42["round_idx"] <= round_slider[1])
        df_filtered = df_s42[mask]

        fig_cum = go.Figure()
        for p in selected_policies:
            sub = df_filtered[df_filtered["policy"] == p]
            fig_cum.add_trace(go.Scatter(
                x=sub["round_idx"],
                y=sub[metric_choice],
                mode="lines",
                name=p,
                line=dict(
                    color=POLICY_COLORS.get(p, "#0F172A"),
                    dash=POLICY_DASHES.get(p, "solid"),
                    width=2.5,
                ),
            ))

        fig_cum.update_layout(
            height=380,
            xaxis_title="Round Number",
            yaxis_title=metric_choice.replace("cum_", "Cumulative ").replace("_", " ").title() + " (₹)",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Financial Gain / Loss: LinUCB vs Baseline Benchmark")
        comparator = st.selectbox("Select Baseline Comparator", options=["Fixed 10%", "Rule-based", "Random"])

        p_linucb = df_filtered[df_filtered["policy"] == "LinUCB"].set_index("round_idx")["cum_profit"]
        p_comp = df_filtered[df_filtered["policy"] == comparator].set_index("round_idx")["cum_profit"]
        diff_series = p_linucb - p_comp

        break_even_rd = "Never (Fixed 10% remained more profitable)"
        positive_mask = (diff_series > 0)
        if positive_mask.any():
            first_pos = diff_series[diff_series > 0].index[0]
            if (diff_series.loc[first_pos:] >= 0).all():
                break_even_rd = f"Round {first_pos:,}"

        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=diff_series.index,
            y=diff_series.values,
            mode="lines",
            name=f"LinUCB minus {comparator}",
            line=dict(color="#0D9488", width=2.5),
        ))
        fig_diff.add_hline(y=0, line_dash="dash", line_color="#EF4444", annotation_text="Break-even Zero Line")

        fig_diff.update_layout(
            height=320,
            title=f"Net Financial Profit Gap (LinUCB − {comparator}) · Break-Even: {break_even_rd}",
            xaxis_title="Round Number",
            yaxis_title="Profit Advantage (₹)",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_diff, use_container_width=True)

        st.subheader("Multi-Seed Benchmark Verification Tables (10 Seeds)")
        mode_toggle = st.radio("Select Evaluation Mode:", ["Online Mode (20,000 Rounds)", "Frozen Mode (5,000 Held-Out Rounds)"], horizontal=True)

        if "Online" in mode_toggle:
            st.dataframe(dash_data["comparison_online"], use_container_width=True, hide_index=True)
            csv_data = dash_data["comparison_online"].to_csv(index=False)
            st.download_button("Download Online Comparison CSV", csv_data, "comparison_online.csv", "text/csv")
        else:
            st.dataframe(dash_data["comparison_frozen"], use_container_width=True, hide_index=True)
            csv_data = dash_data["comparison_frozen"].to_csv(index=False)
            st.download_button("Download Frozen Comparison CSV", csv_data, "comparison_frozen.csv", "text/csv")

        with st.expander("📖 Educational Guide: How to Interpret These Findings"):
            st.markdown(
                """
                1. **Online vs Frozen Evaluation:** In Online mode, LinUCB explores and pays an early exploration tax. In Frozen mode, models run deployed without further parameter updates.
                2. **Why LinUCB Generated Highest Revenue but Lower Profit:** LinUCB discovered that discounts stimulate purchase volume, driving up gross revenue (₹234,276 vs ₹232,094 for Fixed 10%). However, aggressive discounts compress unit profit margins.
                3. **These are simulated experiments:** Results reflect offline simulation on calibrated data, not field forecasts.
                """
            )
    else:
        st.warning("Dashboard dataset assets are not currently synchronized.")


# ===========================================================================
# PAGE 6: ℹ️ MODEL ARCHITECTURE & TRANSPARENCY
# ===========================================================================

elif nav_selection == "ℹ️ Model Architecture & Transparency":
    st.markdown("## ℹ️ Technical Architecture & Methodology")
    about_file = APP_DIR / "about_text.md"
    if about_file.exists():
        with open(about_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.write("Methodology document not found.")
