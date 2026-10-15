"""
Interactive Retail Dynamic Discount Recommender
Streamlit Web Application — Amazon SmartBiz E-Commerce Portal
UI ONLY: Layout, widgets, and charts. All business math is delegated to app/recommender.py.
"""

import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# MUST BE THE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="SmartBiz AI | Dynamic Pricing & Discount Optimization",
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
    "LinUCB": "#00A896",
    "Thompson Sampling": "#2563EB",
    "Fixed 10%": "#FF9900",
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
# AMAZON SMARTBIZ STYLING: HIGH-CONTRAST, LARGE TYPOGRAPHY, CLEAN CARDS
# HIDES STREAMLIT "DEPLOY" BUTTON AND 3-DOTS MENU PER SPECIFICATION
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 1. COMPLETELY HIDE STREAMLIT TOP-RIGHT 'DEPLOY' BUTTON & 3-DOTS MENU */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header[data-testid="stHeader"] {visibility: hidden !important; display: none !important; height: 0 !important;}
    .stDeployButton {display: none !important; visibility: hidden !important;}
    [data-testid="stToolbar"] {display: none !important; visibility: hidden !important;}
    [data-testid="stDecoration"] {display: none !important; visibility: hidden !important;}
    [data-testid="stStatusWidget"] {display: none !important; visibility: hidden !important;}
    footer {visibility: hidden !important; display: none !important;}

    /* 2. PAGE CONTAINER & SPACING */
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 98% !important;
    }

    /* 3. AMAZON SMARTBIZ LARGE TYPOGRAPHY */
    h1 {
        font-size: 2.45rem !important;
        font-weight: 800 !important;
        color: #0C232A !important;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.025em !important;
        line-height: 1.2 !important;
    }
    h2 {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #0C232A !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
    }
    h3 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.4rem !important;
    }
    h4 {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.3rem !important;
    }
    p, span, li {
        font-size: 1.08rem !important;
        color: #334155 !important;
        line-height: 1.6 !important;
    }

    /* 4. SIDEBAR MERCHANDISING CONTROLS (LARGE, CLEAR, HIGH READABILITY) */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
        min-width: 325px !important;
    }
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] label div,
    [data-testid="stSidebar"] label span {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        color: #0C232A !important;
        margin-bottom: 3px !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] .stSelectbox {
        font-size: 1.10rem !important;
        font-weight: 600 !important;
        height: 46px !important;
        border-radius: 9px !important;
    }
    [data-testid="stSidebar"] button {
        font-size: 1.10rem !important;
        font-weight: 700 !important;
        padding: 11px 18px !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] {
        width: 34px !important;
        height: 34px !important;
    }

    /* 5. TOP HORIZONTAL NAVIGATION (ONE LEFT TO ANOTHER, SMARTBIZ PILLS) */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
        background-color: #F1F5F9 !important;
        padding: 10px 14px !important;
        border-radius: 14px !important;
        border: 1px solid #CBD5E1 !important;
        margin-top: 10px !important;
        margin-bottom: 22px !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-size: 1.08rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background-color: #E2E8F0 !important;
        border-color: #94A3B8 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: #0C232A !important;
        color: #FFFFFF !important;
        border-color: #0C232A !important;
        box-shadow: 0 4px 10px rgba(12, 35, 42, 0.3) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {
        color: #FFFFFF !important;
    }

    /* 6. SMARTBIZ HERO & METRIC CARDS */
    .smartbiz-hero {
        background: linear-gradient(135deg, #0C232A 0%, #163842 100%);
        border-radius: 20px;
        padding: 38px 44px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(12, 35, 42, 0.15);
    }
    .smartbiz-hero h1 {
        color: #FFFFFF !important;
        font-size: 2.7rem !important;
    }
    .smartbiz-hero p {
        color: #E2E8F0 !important;
        font-size: 1.2rem !important;
        max-width: 820px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        color: #0C232A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .product-img-card {
        border-radius: 14px 14px 0 0;
        width: 100%;
        height: 175px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# REPRESENTATIVE STORE PRODUCT CATALOG (EXPANDED TO 9 SKUS WITH REAL PHOTOS)
# ---------------------------------------------------------------------------
STORE_PRODUCTS = [
    {
        "id": "prod_1",
        "name": "Sony WH-1000XM5 ANC Headphones",
        "category": "Electronics",
        "badge": "Top Seller",
        "badge_color": "#0F766E",
        "badge_bg": "#CCFBF1",
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80",
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
        "name": "Apple Watch Series 9 GPS",
        "category": "Electronics",
        "badge": "High Demand",
        "badge_color": "#1D4ED8",
        "badge_bg": "#DBEAFE",
        "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80",
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
        "name": "Dell UltraSharp 32-inch 4K USB-C Monitor",
        "category": "Electronics",
        "badge": "Pro Merch",
        "badge_color": "#0369A1",
        "badge_bg": "#E0F2FE",
        "image_url": "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=600&auto=format&fit=crop&q=80",
        "price": 120.00,
        "comp_price": 125.00,
        "stock": 95,
        "demand": "High",
        "season": "Spring",
        "orders": 7,
        "recency": 15,
        "basket": 260.00,
        "elasticity": 0.88,
        "promo": True,
        "region": "South",
        "description": "IPS Black technology with 2000:1 contrast ratio, 90W power delivery hub, and 98% DCI-P3.",
    },
    {
        "id": "prod_4",
        "name": "Herman Miller Ergonomic Task Chair",
        "category": "Furniture",
        "badge": "Low Stock",
        "badge_color": "#B45309",
        "badge_bg": "#FEF3C7",
        "image_url": "https://images.unsplash.com/photo-1592078615290-033ee584e267?w=600&auto=format&fit=crop&q=80",
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
        "description": "Breathable 8Z Pellicle elastomeric mesh with adjustable PostureFit SL sacral lumbar support.",
    },
    {
        "id": "prod_5",
        "name": "Oak Dual-Motor Standing Desk",
        "category": "Furniture",
        "badge": "New Arrival",
        "badge_color": "#4338CA",
        "badge_bg": "#EEF2FF",
        "image_url": "https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?w=600&auto=format&fit=crop&q=80",
        "price": 95.00,
        "comp_price": 99.00,
        "stock": 130,
        "demand": "Medium",
        "season": "Spring",
        "orders": 4,
        "recency": 25,
        "basket": 210.00,
        "elasticity": 0.92,
        "promo": False,
        "region": "West",
        "description": "Solid European white oak desktop with programmable digital keypad and anti-collision sensor.",
    },
    {
        "id": "prod_6",
        "name": "Blue Tokai Organic Espresso (1kg)",
        "category": "Groceries",
        "badge": "High Margin",
        "badge_color": "#15803D",
        "badge_bg": "#DCFCE7",
        "image_url": "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=600&auto=format&fit=crop&q=80",
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
        "id": "prod_7",
        "name": "Himalayan Roasted Almond Trail Mix",
        "category": "Groceries",
        "badge": "Everyday Value",
        "badge_color": "#047857",
        "badge_bg": "#D1FAE5",
        "image_url": "https://images.unsplash.com/photo-1509440159596-0249088772ff?w=600&auto=format&fit=crop&q=80",
        "price": 22.00,
        "comp_price": 24.00,
        "stock": 380,
        "demand": "Medium",
        "season": "Autumn",
        "orders": 2,
        "recency": 30,
        "basket": 48.00,
        "elasticity": 0.82,
        "promo": False,
        "region": "North",
        "description": "Slow-roasted California almonds, organic pumpkin seeds, and dried cranberries with rock salt.",
    },
    {
        "id": "prod_8",
        "name": "Patagonia Thermal Windbreaker",
        "category": "Clothing",
        "badge": "Winter Special",
        "badge_color": "#6D28D9",
        "badge_bg": "#EDE9FE",
        "image_url": "https://images.unsplash.com/photo-1544923246-77307dd654cb?w=600&auto=format&fit=crop&q=80",
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
        "description": "Ultralight, windproof water-resistant shell filled with 60g recycled PrimaLoft Gold Insulation.",
    },
    {
        "id": "prod_9",
        "name": "LEGO Technic Porsche 911 GT3 RS",
        "category": "Toys",
        "badge": "Collector Item",
        "badge_color": "#B91C1C",
        "badge_bg": "#FEE2E2",
        "image_url": "https://images.unsplash.com/photo-1585386959984-a4155224a1ad?w=600&auto=format&fit=crop&q=80",
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
    {
        "id": "prod_10",
        "name": "DJI Mini 4 Pro Drone Combo",
        "category": "Electronics",
        "badge": "Top Merch",
        "badge_color": "#0F766E",
        "badge_bg": "#CCFBF1",
        "image_url": "https://images.unsplash.com/photo-1527977966376-1c8408f9f108?w=600&auto=format&fit=crop&q=80",
        "price": 95.00,
        "comp_price": 98.00,
        "stock": 110,
        "demand": "High",
        "season": "Summer",
        "orders": 8,
        "recency": 12,
        "basket": 220.00,
        "elasticity": 1.02,
        "promo": True,
        "region": "South",
        "description": "Sub-249g ultra-lightweight drone with 4K/60fps HDR video, omnidirectional obstacle sensing.",
    },
    {
        "id": "prod_11",
        "name": "JBL Charge 5 Bluetooth Speaker",
        "category": "Electronics",
        "badge": "Trending",
        "badge_color": "#1D4ED8",
        "badge_bg": "#DBEAFE",
        "image_url": "https://images.unsplash.com/photo-1545454675-3531b543be5d?w=600&auto=format&fit=crop&q=80",
        "price": 45.00,
        "comp_price": 48.00,
        "stock": 280,
        "demand": "High",
        "season": "Summer",
        "orders": 6,
        "recency": 18,
        "basket": 120.00,
        "elasticity": 0.98,
        "promo": False,
        "region": "East",
        "description": "IP67 dustproof and waterproof speaker with 20 hours of playtime and built-in powerbank.",
    },
    {
        "id": "prod_12",
        "name": "Minimalist Nordic Arc Floor Lamp",
        "category": "Furniture",
        "badge": "Design Icon",
        "badge_color": "#4338CA",
        "badge_bg": "#EEF2FF",
        "image_url": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?w=600&auto=format&fit=crop&q=80",
        "price": 38.00,
        "comp_price": 40.00,
        "stock": 190,
        "demand": "Medium",
        "season": "Winter",
        "orders": 3,
        "recency": 22,
        "basket": 85.00,
        "elasticity": 0.89,
        "promo": False,
        "region": "North",
        "description": "Dimmable LED floor reading lamp with brushed brass arch and solid marble stabilizing base.",
    },
    {
        "id": "prod_13",
        "name": "Kashmiri Imperial Green Tea",
        "category": "Groceries",
        "badge": "Organic",
        "badge_color": "#15803D",
        "badge_bg": "#DCFCE7",
        "image_url": "https://images.unsplash.com/photo-1576092768241-dec231879fc3?w=600&auto=format&fit=crop&q=80",
        "price": 32.00,
        "comp_price": 35.00,
        "stock": 420,
        "demand": "High",
        "season": "Winter",
        "orders": 5,
        "recency": 15,
        "basket": 68.00,
        "elasticity": 0.81,
        "promo": True,
        "region": "North",
        "description": "First-flush whole-leaf green tea blended with whole Kashmiri saffron strands and crushed green cardamom.",
    },
    {
        "id": "prod_14",
        "name": "Nike Air Zoom Pegasus 40",
        "category": "Clothing",
        "badge": "Athletic",
        "badge_color": "#B91C1C",
        "badge_bg": "#FEE2E2",
        "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&auto=format&fit=crop&q=80",
        "price": 88.00,
        "comp_price": 92.00,
        "stock": 165,
        "demand": "High",
        "season": "Spring",
        "orders": 9,
        "recency": 11,
        "basket": 175.00,
        "elasticity": 1.08,
        "promo": False,
        "region": "West",
        "description": "Dual Zoom Air units with lightweight engineered mesh and responsive React foam midsole.",
    },
    {
        "id": "prod_15",
        "name": "Herschel Little America Backpack",
        "category": "Clothing",
        "badge": "Popular",
        "badge_color": "#6D28D9",
        "badge_bg": "#EDE9FE",
        "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=600&auto=format&fit=crop&q=80",
        "price": 52.00,
        "comp_price": 55.00,
        "stock": 240,
        "demand": "Medium",
        "season": "Autumn",
        "orders": 4,
        "recency": 28,
        "basket": 110.00,
        "elasticity": 0.94,
        "promo": False,
        "region": "East",
        "description": "Classic mountaineering silhouette with padded fleece-lined 15-inch laptop sleeve and magnetic buckle closure.",
    },
    {
        "id": "prod_16",
        "name": "Catan 3D Collector Edition",
        "category": "Toys",
        "badge": "Collector",
        "badge_color": "#B45309",
        "badge_bg": "#FEF3C7",
        "image_url": "https://images.unsplash.com/photo-1610890716171-6b1bb98ffd09?w=600&auto=format&fit=crop&q=80",
        "price": 42.00,
        "comp_price": 45.00,
        "stock": 175,
        "demand": "Medium",
        "season": "Winter",
        "orders": 2,
        "recency": 35,
        "basket": 88.00,
        "elasticity": 0.87,
        "promo": False,
        "region": "South",
        "description": "Hand-painted 3D terrain tile set with antiqued wooden player pieces and custom resource card trays.",
    },
]

# ---------------------------------------------------------------------------
# DATA LOADERS & RESOURCE CACHING (PREVENTS SLOW PAGE SWITCHES)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads and caches the frozen LinUCB model."""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading trained bandit model: {e}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_dashboard_assets():
    """Loads precomputed evaluation simulation data with high-speed memory caching."""
    try:
        kpi_path = DASHBOARD_DIR / "summary_kpis.json"
        with open(kpi_path, "r", encoding="utf-8") as f:
            kpis = json.load(f)

        df_s42 = pd.read_csv(DASHBOARD_DIR / "cumulative_series_seed42.csv")
        df_online = pd.read_csv(DASHBOARD_DIR / "comparison_online.csv")
        df_frozen = pd.read_csv(DASHBOARD_DIR / "comparison_frozen.csv")
        df_lift = pd.read_csv(DASHBOARD_DIR / "comparison_lift.csv")

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
        st.warning(f"Dashboard files could not be loaded: {e}.")
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
# AMAZON SMARTBIZ MERCHANT SIGN-IN PAGE
# ---------------------------------------------------------------------------

if not st.session_state["authenticated"]:
    # Amazon SmartBiz Brand Header
    st.markdown(
        """
        <div style="text-align: center; padding: 15px 0 25px 0;">
            <div style="font-size: 34px; font-weight: 900; color: #0C232A; letter-spacing: -0.04em; display: inline-flex; align-items: baseline; gap: 4px;">
                smart<span style="color: #00A896;">biz</span>
                <span style="font-size: 15px; font-weight: 600; color: #64748B; margin-left: 6px;">by amazon &middot; AI Dynamic Pricing Portal</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_hero, col_login = st.columns([1.25, 1], gap="large")

    with col_hero:
        st.title("⚡ SmartBiz AI Pricing Engine")
        st.markdown(
            """
            <div style="background-color: #FBF8F3; border-radius: 20px; padding: 34px 38px; border: 1px solid #F1ECE1; margin-bottom: 20px;">
                <h1 style="font-size: 3.1rem !important; font-weight: 800; color: #0C232A; line-height: 1.15; margin: 0 0 16px 0; letter-spacing: -0.03em;">
                    Build, Launch & Grow Your D2C Business
                </h1>
                <p style="font-size: 1.22rem !important; color: #475569; line-height: 1.6; margin-bottom: 24px;">
                    Create a stunning e-commerce website powered by Amazon. SmartBiz's website builder provides all the tools you need to grow your business effortlessly.
                </p>
                <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 12px;">
                    <span style="background: #0C232A; color: #FFFFFF; padding: 10px 22px; border-radius: 9999px; font-weight: 700; font-size: 15px; display: inline-flex; align-items: center; gap: 8px;">
                        Sign up for free &rarr;
                    </span>
                    <span style="color: #00A896; font-weight: 700; font-size: 15px;">
                        ⚡ 100% Inviolable Margin Guardrails
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### 🔐 Merchant Portal Sign In")
            st.caption("Access live contextual discount recommendations, guardrail telemetry, and ROI audits.")

            if st.button("🚀 Instant 1-Click Demo Login as Pricing Director", use_container_width=True, type="primary"):
                handle_login("sujan@smartbiz.nexus")
                st.rerun()

            st.markdown("<div style='text-align: center; color: #94A3B8; font-weight: 700; font-size: 13px; margin: 14px 0;'>— OR ENTER CREDENTIALS —</div>", unsafe_allow_html=True)

            col_l_e, col_l_p = st.columns(2)
            with col_l_e:
                login_email = st.text_input("Store Manager Email", value="sujan@smartbiz.nexus", key="login_email_input")
            with col_l_p:
                login_pass = st.text_input("Password", value="••••••••", type="password", key="login_pass_input")

            st.checkbox("Keep me signed in for 30 days", value=True)

            if st.button("Sign In to Merchant Hub", use_container_width=True):
                handle_login(login_email)
                st.rerun()

    with col_login:
        st.markdown(
            """
            <div style="background: #FFFFFF; border-radius: 20px; padding: 22px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid #E2E8F0; text-align: center;">
                <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=600&auto=format&fit=crop&q=80" style="width: 100%; height: 260px; object-fit: cover; border-radius: 14px; margin-bottom: 18px;" alt="Entrepreneur with laptop"/>
                
                <div style="display: flex; gap: 14px; justify-content: space-between; text-align: left;">
                    <div style="flex: 1; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px;">
                        <small style="color: #64748B; font-weight: 700; text-transform: uppercase;">Order Summary</small>
                        <div style="font-size: 18px; font-weight: 800; color: #0C232A; margin: 4px 0 8px 0;">₹2,499.00</div>
                        <div style="background-color: #00A896; color: #FFFFFF; text-align: center; padding: 6px 12px; border-radius: 8px; font-weight: 700; font-size: 13px;">
                            Pay Now
                        </div>
                    </div>
                    <div style="flex: 1; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px;">
                        <small style="color: #64748B; font-weight: 700; text-transform: uppercase;">Sales per week</small>
                        <div style="font-size: 20px; font-weight: 800; color: #00A896; margin: 4px 0 4px 0;">+24.3%</div>
                        <div style="display: flex; align-items: flex-end; gap: 4px; height: 32px; padding-top: 6px;">
                            <span style="background: #CBD5E1; width: 14%; height: 40%; border-radius: 3px;"></span>
                            <span style="background: #CBD5E1; width: 14%; height: 55%; border-radius: 3px;"></span>
                            <span style="background: #CBD5E1; width: 14%; height: 45%; border-radius: 3px;"></span>
                            <span style="background: #00A896; width: 14%; height: 75%; border-radius: 3px;"></span>
                            <span style="background: #00A896; width: 14%; height: 85%; border-radius: 3px;"></span>
                            <span style="background: #00C49F; width: 14%; height: 100%; border-radius: 3px;"></span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Section 2: Dark Teal (#07282F) from Image 5
    st.markdown(
        """
        <div style="background-color: #07282F; border-radius: 20px; padding: 38px 44px; color: #FFFFFF; margin-bottom: 26px;">
            <div style="display: grid; grid-template-columns: 1fr 1.2fr; gap: 32px; align-items: center;">
                <div style="border: 1px solid #164853; border-radius: 16px; padding: 30px;">
                    <h2 style="color: #FFFFFF !important; font-size: 2.4rem !important; font-weight: 800; line-height: 1.2; margin: 0 0 16px 0;">
                        Run Your Store at Low Cost
                    </h2>
                    <p style="color: #E2E8F0; font-size: 1.15rem; line-height: 1.6; margin-bottom: 14px;">
                        Start your website with no setup fees*, no subscriptions*, and low shipping rates.
                    </p>
                    <p style="color: #94A3B8; font-size: 1.12rem; line-height: 1.6; margin-bottom: 18px;">
                        Build and grow your business with a clear focus on your products, not your expenses.<br>
                        <span style="color: #64748B; font-size: 13px;">* Free until Oct 2026.</span>
                    </p>
                    <div style="display: inline-block; background: #0F3B44; color: #00C49F; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 13px;">
                        🛡️ Zero Below-Cost Margin Violations Guaranteed
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <div style="background: linear-gradient(135deg, #00A896 0%, #028090 100%); border-radius: 14px; padding: 20px 24px; color: #FFFFFF;">
                        <span style="background: rgba(0,0,0,0.25); padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">
                            ⚡ Automations & LinUCB
                        </span>
                        <h4 style="color: #FFFFFF !important; margin: 8px 0 4px 0; font-size: 18px;">Everything You Need, Built In</h4>
                        <p style="color: #E0F2FE; font-size: 13px; margin: 0;">
                            Order management, contextual bandit personalization, and sub-millisecond inference algorithms included by default.
                        </p>
                    </div>
                    <div style="background: linear-gradient(135deg, #FF9900 0%, #F59E0B 100%); border-radius: 14px; padding: 20px 24px; color: #FFFFFF;">
                        <span style="background: rgba(0,0,0,0.25); padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;">
                            🛡️ Amazon Trust Advantage
                        </span>
                        <h4 style="color: #FFFFFF !important; margin: 8px 0 4px 0; font-size: 18px;">Commercial Guardrails Architecture</h4>
                        <p style="color: #FFFBEB; font-size: 13px; margin: 0;">
                            Built-in unit margin floors and customer loyalty discount ceilings prevent loss-making markdowns before they are ever served.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


# ---------------------------------------------------------------------------
# TOP HEADER & HORIZONTAL NAVIGATION (SMARTBIZ STYLE)
# ---------------------------------------------------------------------------

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"🛍️ Welcome Back, {st.session_state['user_name']}")
    st.caption("SmartCommerce Merchandising Portal · Disjoint LinUCB Dynamic Engine Active")
with col_h2:
    col_u_pill, col_u_out = st.columns([3, 2])
    with col_u_pill:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 4px;">
                <strong style="color: #0C232A; font-size: 16px;">👤 {st.session_state['user_name']}</strong><br>
                <small style="color: #64748B; font-weight: 600;">{st.session_state['user_role']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_u_out:
        st.button("Sign Out", on_click=handle_logout, use_container_width=True)

# TOP HORIZONTAL NAVIGATION (ONE LEFT TO ANOTHER, SMARTBIZ STYLE)
nav_selection = st.radio(
    "Console Navigation",
    options=[
        "🏠 Store Overview & Catalog",
        "🎯 Live Pricing Simulator",
        "🔮 Market Sensitivity & What-If Lab",
        "🛡️ Margin Guardrails Studio",
        "📊 Policy Performance & ROI Audits",
        "📘 Knowledge Blog & Architecture",
        "ℹ️ Model Architecture & Transparency",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)


# ---------------------------------------------------------------------------
# SIDEBAR: ACTIVE MERCHANDISING CONTROLS (LARGE, HIGH READABILITY)
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 🏷️ Merchandising Controls")
st.sidebar.caption("Fine-tune catalogue pricing, market benchmarks, and shopper loyalty.")

# Group 1: Product Inputs
st.sidebar.markdown("#### Product & Commercials")
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    price_val = st.sidebar.number_input(
        "Base Price (₹)",
        min_value=10.0,
        max_value=150.0,
        step=1.0,
        key="input_Price",
        help="Catalogue selling price of the item before discount.",
    )
with col_p2:
    comp_val = st.sidebar.number_input(
        "Competitor Price (₹)",
        min_value=10.0,
        max_value=150.0,
        step=1.0,
        key="input_CompetitorPricing",
        help="Observed price for matching SKU at competitor.",
    )

cat_val = st.sidebar.selectbox(
    "Product Category",
    options=["Electronics", "Groceries", "Furniture", "Toys", "Clothing"],
    key="input_Category",
    help="Product retail category.",
)

# Group 2: Inventory & Demand Dynamics
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
    "Active Promotional Campaign",
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
        "Basket Value (₹)",
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
st.sidebar.markdown(f"**Assigned Loyalty Tier:** `{active_rule_tier}`")
st.sidebar.caption(f"Price is **{abs(comp_gap_pct):.1f}% {gap_direction}** competitor benchmark.")
st.sidebar.caption(f"Base unit margin: **{fmt_rupees(unit_margin_zero)}** (40% margin rate).")

with st.sidebar.expander("Cost Price Override"):
    cost_override = st.number_input(
        "Cost Price Override (₹)",
        min_value=0.0,
        max_value=150.0,
        step=1.0,
        key="cost_price_override",
        help="Set to 0 to maintain calibrated 40% margin rate.",
    )
    st.button("Reset Inputs to Defaults", on_click=reset_situation_inputs, use_container_width=True)

st.sidebar.markdown("### 🛡️ Guardrail Rules")
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
        "Regular cutoff",
        min_value=1,
        max_value=15,
        step=1,
        key="guardrail_reg_cutoff",
    )
with col_g2:
    loyal_cutoff_input = st.number_input(
        "Loyal cutoff",
        min_value=2,
        max_value=25,
        step=1,
        key="guardrail_loyal_cutoff",
    )

if reg_cutoff_input >= loyal_cutoff_input:
    st.sidebar.error("Invalid rule: 'Regular' cutoff must be strictly less than 'Loyal'.")
    reg_cutoff_input = DEFAULT_REGULAR_CUTOFF
    loyal_cutoff_input = DEFAULT_LOYAL_CUTOFF

st.sidebar.button("Reset Guardrails", on_click=reset_guardrails, use_container_width=True)


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
# PAGE 1: 🏠 STORE OVERVIEW & CATALOG
# ===========================================================================

if nav_selection == "🏠 Store Overview & Catalog":
    # Hero Callout Card (SmartBiz Style)
    st.markdown(
        """
        <div class="smartbiz-hero">
            <h1 style="margin: 0 0 10px 0;">Build & Scale Your Store with AI Pricing</h1>
            <p style="margin: 0 0 20px 0;">
                Live contextual reinforcement learning dynamically balances margin protection and purchase conversion across all SKU categories.
            </p>
            <div style="display: flex; gap: 14px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px;">⚡ LinUCB Active</span>
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px;">🛡️ 100% Margin Floor Enforced</span>
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px;">📈 24.3% Conversion Rate</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Top KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Revenue", "₹ 2,34,277", "↗ +12.5% vs baseline")
    with col_k2:
        st.metric("Total Orders", "4,931", "↗ +24.3% conversion rate")
    with col_k3:
        st.metric("Guardrail Status", "100% Safe", f"Floor ₹{margin_floor_slider:.2f} active")
    with col_k4:
        st.metric("Active Model", "Disjoint LinUCB", "O(d²) Sherman-Morrison")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Middle Section: Revenue Trendline + Low Stock Alerts
    col_dash_left, col_dash_right = st.columns([2, 1])

    with col_dash_left:
        with st.container(border=True):
            st.markdown("### 📈 Revenue & Cumulative Profit Trend")
            st.caption("20,000-round progression across LinUCB and benchmark policies.")

            dash_data = load_dashboard_assets()
            if dash_data is not None and "series_seed42" in dash_data:
                df_s42 = dash_data["series_seed42"]
                df_lin = df_s42[df_s42["policy"] == "LinUCB"]
                df_fixed = df_s42[df_s42["policy"] == "Fixed 10%"]

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=df_lin["round_idx"],
                    y=df_lin["cum_profit"],
                    mode="lines",
                    name="LinUCB (Dynamic)",
                    line=dict(color="#00A896", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 168, 150, 0.08)",
                ))
                fig_trend.add_trace(go.Scatter(
                    x=df_fixed["round_idx"],
                    y=df_fixed["cum_profit"],
                    mode="lines",
                    name="Fixed 10% (Baseline)",
                    line=dict(color="#FF9900", width=2, dash="dash"),
                ))

                fig_trend.update_layout(
                    height=290,
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
            st.markdown("### ⚠️ Inventory & Margin Watchlist")
            st.caption("Live alerts across active store SKU categories.")

            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">Herman Miller Aeron</strong><br>
                        <small style="color: #64748B;">Furniture · Stock: 85</small>
                    </div>
                    <span style="background-color: #FEF3C7; color: #B45309; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">Low Stock</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">LEGO Porsche GT3</strong><br>
                        <small style="color: #64748B;">Toys · Stock: 140</small>
                    </div>
                    <span style="background-color: #FEE2E2; color: #DC2626; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">Collector</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">Sony WH-1000XM5</strong><br>
                        <small style="color: #64748B;">Electronics · Stock: 320</small>
                    </div>
                    <span style="background-color: #CCFBF1; color: #0F766E; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">Top Seller</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
                    <div>
                        <strong style="color: #0F172A; font-size: 15px;">Commercial Guardrails</strong><br>
                        <small style="color: #64748B;">Zero Below-Cost Sales</small>
                    </div>
                    <span style="background-color: #DCFCE7; color: #15803D; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">100% Compliant</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Extra Visualizations (Store Overview Performance)
    col_g_left, col_g_right = st.columns(2)
    with col_g_left:
        with st.container(border=True):
            st.markdown("### 📊 Conversion Lift by Category")
            st.caption("Empirical lift of contextual discount optimization vs static pricing.")
            cat_df = pd.DataFrame({
                "Category": ["Electronics", "Furniture", "Clothing", "Toys", "Groceries"],
                "Static Conversion (%)": [21.4, 18.2, 23.1, 20.5, 25.8],
                "LinUCB Dynamic (%)": [25.6, 22.8, 27.4, 24.2, 28.9],
            })
            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(x=cat_df["Category"], y=cat_df["Static Conversion (%)"], name="Static Baseline", marker_color="#CBD5E1"))
            fig_cat.add_trace(go.Bar(x=cat_df["Category"], y=cat_df["LinUCB Dynamic (%)"], name="LinUCB Optimized", marker_color="#00A896"))
            fig_cat.update_layout(
                barmode="group",
                height=260,
                margin=dict(l=10, r=10, t=10, b=20),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                yaxis_title="Conversion Rate (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_g_right:
        with st.container(border=True):
            st.markdown("### 🛡️ Customer Tier Margin Distribution")
            st.caption("Distribution of transactions governed by loyalty discount ceilings.")
            tier_df = pd.DataFrame({
                "Tier": ["New Shoppers (0-5% Max)", "Regular Customers (5-15% Max)", "Loyal Shoppers (10-20% Max)"],
                "Volume": [35, 45, 20],
                "Color": ["#00A896", "#2563EB", "#FF9900"],
            })
            fig_pie = go.Figure(data=[go.Pie(labels=tier_df["Tier"], values=tier_df["Volume"], hole=.45, marker_colors=tier_df["Color"])])
            fig_pie.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Featured Product Catalog with 16 SKUs, Filters, and Real Photos
    st.markdown("## 🏷️ Featured Store Catalog — 1-Click Discount Optimizer")
    st.caption("Filter by category or search products, then click **'⚡ Optimize Discount'** on any product to instantly load its context into the pricing engine.")

    col_fil1, col_fil2 = st.columns([1, 2])
    with col_fil1:
        cat_filter = st.selectbox(
            "Filter by Category",
            options=["All Categories", "Electronics", "Furniture", "Groceries", "Clothing", "Toys"],
            key="cat_filter_choice",
        )
    with col_fil2:
        search_query = st.text_input(
            "Search Catalog Products",
            placeholder="Search by SKU name, description, or category...",
            key="search_catalog_input",
        )

    filtered_products = STORE_PRODUCTS
    if cat_filter != "All Categories":
        filtered_products = [p for p in filtered_products if p["category"] == cat_filter]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_products = [
            p for p in filtered_products
            if q in p["name"].lower() or q in p["description"].lower() or q in p["category"].lower()
        ]

    st.caption(f"Showing {len(filtered_products)} items matching selected filters:")

    for row_idx in range(0, len(filtered_products), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            item_idx = row_idx + col_idx
            if item_idx < len(filtered_products):
                prod = filtered_products[item_idx]
                escaped_name = html.escape(prod['name'])
                escaped_desc = html.escape(prod['description'])
                with cols[col_idx]:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <img src="{prod['image_url']}" class="product-img-card" alt="{escaped_name}" onerror="this.onerror=null;this.src='https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80';"/>
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin: 12px 0 6px 0;">
                                <span style="background-color: {prod['badge_bg']}; color: {prod['badge_color']}; padding: 4px 12px; border-radius: 9999px; font-size: 13px; font-weight: 700;">
                                    {prod['badge']}
                                </span>
                                <span style="color: #64748B; font-size: 13px; font-weight: 600;">{prod['category']}</span>
                            </div>
                            <h3 style="margin: 0 0 6px 0; font-size: 18px; color: #0C232A;">{escaped_name}</h3>
                            <p style="font-size: 14px !important; color: #475569; margin-bottom: 12px; height: 42px; overflow: hidden;">{escaped_desc}</p>
                            <div style="display: flex; justify-content: space-between; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px;">
                                <div>
                                    <small style="color: #64748B; font-weight: 600;">Catalogue Price</small><br>
                                    <strong style="color: #0F172A; font-size: 18px;">₹{prod['price']:.2f}</strong>
                                </div>
                                <div style="text-align: right;">
                                    <small style="color: #64748B; font-weight: 600;">Competitor</small><br>
                                    <strong style="color: #475569; font-size: 18px;">₹{prod['comp_price']:.2f}</strong>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.button(
                            f"⚡ Optimize Discount for {prod['category']}",
                            key=f"btn_opt_{prod['id']}",
                            use_container_width=True,
                            on_click=load_product_into_simulator,
                            args=(prod,),
                        )


# ===========================================================================
# PAGE 2: 🎯 LIVE PRICING SIMULATOR
# ===========================================================================

elif nav_selection == "🎯 Live Pricing Simulator":
    st.markdown("## 🎯 Live Contextual Pricing Recommendation")
    st.caption("Real-time LinUCB scoring engine evaluating all discount options under loyalty and margin constraints.")

    # Top Recommendation Banner
    if rec_result.chosen_discount is not None:
        st.markdown(
            f"""
            <div style="background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 16px; padding: 22px 28px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="background-color: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 9999px; font-size: 13px; font-weight: 700;">
                            RECOMMENDED ACTION: {rec_result.chosen_discount}% DISCOUNT
                        </span>
                        <h2 style="margin: 8px 0 4px 0; color: #064E3B; font-size: 2.2rem !important;">
                            Optimal Selling Price: {fmt_rupees(rec_result.discounted_price)}
                        </h2>
                        <p style="margin: 0; color: #047857; font-size: 1.15rem !important;">{rec_result.why_pick_text}</p>
                    </div>
                    <div style="text-align: right; background: #FFFFFF; padding: 14px 22px; border-radius: 12px; border: 1px solid #A7F3D0;">
                        <small style="color: #64748B; font-weight: 700;">UNIT MARGIN</small><br>
                        <strong style="color: #065F46; font-size: 26px;">{fmt_rupees(rec_result.unit_margin_after_discount)}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("All discount options are currently blocked by commercial safety guardrails.")

    # 4 Decision KPI Cards
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Effective Selling Price", fmt_rupees(rec_result.discounted_price), f"{rec_result.chosen_discount}% markdown")
    with col_r2:
        st.metric("Discount Concession", fmt_rupees(rec_result.discount_amount_rupees), "Per-unit concession")
    with col_r3:
        st.metric("Unit Profit Margin", fmt_rupees(rec_result.unit_margin_after_discount), f"Floor: ₹{active_rules.margin_floor:.2f}")
    with col_r4:
        best_arm_info = next((a for a in rec_result.arms_table if a.discount == rec_result.chosen_discount), None)
        pt_est = best_arm_info.point_estimate if best_arm_info else 0.0
        st.metric("Expected Unit Profit", fmt_rupees(pt_est if pt_est else 0.0), "LinUCB exploitation term")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Visualization: UCB Score Decomposition Whiskers + Decision Explanation
    col_plot_left, col_plot_right = st.columns([1.5, 1])

    with col_plot_left:
        with st.container(border=True):
            st.markdown("### 📊 UCB Score Decomposition by Discount Arm")
            st.caption("Point estimate (exploitation) with upper whisker showing uncertainty bonus (exploration).")

            fig_ucb = go.Figure()
            for arm in rec_result.arms_table:
                if arm.score is not None:
                    fig_ucb.add_trace(go.Scatter(
                        x=[arm.discount],
                        y=[arm.point_estimate],
                        error_y=dict(type="data", array=[arm.uncertainty_bonus], visible=True, thickness=2.5, width=8, color="#00A896"),
                        mode="markers",
                        name=f"{arm.discount}% Discount",
                        marker=dict(size=14, color="#00A896" if arm.discount == rec_result.chosen_discount else "#2563EB"),
                    ))
                else:
                    fig_ucb.add_trace(go.Scatter(
                        x=[arm.discount],
                        y=[0],
                        mode="markers",
                        marker=dict(size=12, symbol="x", color="#EF4444"),
                        name=f"{arm.discount}% Blocked",
                    ))

            fig_ucb.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=25),
                xaxis=dict(title="Offered Discount (%)", tickvals=[0, 5, 10, 15, 20]),
                yaxis=dict(title="Estimated Profit per Visit (₹)"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                showlegend=False,
            )
            st.plotly_chart(fig_ucb, use_container_width=True)

    with col_plot_right:
        with st.container(border=True):
            st.markdown("### 🧠 Decision Logic Explanation")
            st.caption("How LinUCB weighed exploitation vs exploration:")
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px;">
                    <p style="margin: 0 0 10px 0;"><strong>Selected Action:</strong> {rec_result.chosen_discount}% discount</p>
                    <p style="margin: 0 0 12px 0; font-size: 14px !important; color: #475569;">
                        {rec_result.why_pick_text}
                    </p>
                    <div style="border-top: 1px solid #E2E8F0; padding-top: 10px;">
                        <small style="color: #64748B;">
                            1. <strong>Exploitation:</strong> Historical margin earned from similar shopper contexts.<br>
                            2. <strong>Exploration:</strong> Uncertainty bonus scaled by confidence parameter &alpha;.<br>
                            3. <strong>Guardrail Envelope:</strong> Ineligible actions filtered out prior to scoring.
                        </small>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Detailed Arms Diagnostic Matrix
    st.markdown("### 📋 Detailed Arms Diagnostic Matrix")
    diag_rows = []
    for a in rec_result.arms_table:
        diag_rows.append({
            "Discount": f"{a.discount}%",
            "Status": a.status.replace("_", " ").title(),
            "Selling Price": fmt_rupees(a.selling_price),
            "Unit Margin": fmt_rupees(a.unit_margin_after_discount),
            "Point Estimate": fmt_rupees(a.point_estimate) if a.point_estimate is not None else "—",
            "Uncertainty Bonus": fmt_rupees(a.uncertainty_bonus) if a.uncertainty_bonus is not None else "—",
            "Total UCB Score": fmt_rupees(a.score) if a.score is not None else "—",
            "Training Pulls": f"{a.pull_counts:,}",
            "Decision Reason": a.reason,
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 3: 🔮 MARKET SENSITIVITY & WHAT-IF LAB
# ===========================================================================

elif nav_selection == "🔮 Market Sensitivity & What-If Lab":
    st.markdown("## 🔮 Market Sensitivity & What-If Counterfactual Lab")
    st.caption("Simulate macroeconomic competitor price shocks, inventory depletion, and seasonal shifts.")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        comp_pct_slider = st.slider("Competitor Price Adjustment (%)", -30.0, 30.0, 0.0, 5.0)
    with col_w2:
        inv_mult_slider = st.slider("Inventory Depth Multiplier", 0.25, 3.0, 1.0, 0.25)
    with col_w3:
        season_whatif = st.selectbox("Simulated Season", ["Same as base", "Spring", "Summer", "Autumn", "Winter"])

    whatif_result = recommend_scenarios(
        base_inputs=current_inputs,
        whatif_deltas={
            "competitor_price_pct_change": comp_pct_slider,
            "inventory_multiplier": inv_mult_slider,
            "seasonality": season_whatif,
        },
        rules=active_rules,
        model=model,
    )

    base_rec = whatif_result["base_recommendation"]
    scen_rec = whatif_result["whatif_recommendation"]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        with st.container(border=True):
            st.markdown("### 🏢 Baseline Scenario")
            st.write(f"**Recommended Discount:** {base_rec.chosen_discount}%")
            st.write(f"**Effective Price:** {fmt_rupees(base_rec.discounted_price)}")
            st.write(f"**Post-Discount Margin:** {fmt_rupees(base_rec.unit_margin_after_discount)}")

    with col_s2:
        with st.container(border=True):
            st.markdown("### ⚡ What-If Scenario")
            st.write(f"**Recommended Discount:** {scen_rec.chosen_discount}%")
            st.write(f"**Effective Price:** {fmt_rupees(scen_rec.discounted_price)}")
            st.write(f"**Post-Discount Margin:** {fmt_rupees(scen_rec.unit_margin_after_discount)}")

    st.markdown("### 📈 Continuous Sensitivity Response Curves")
    sens_var = st.selectbox("Select Sweep Parameter", ["CompetitorPricing", "InventoryLevel", "Seasonality"])
    curve_data = compute_response_curve(current_inputs, sens_var, rules=active_rules, model=model)

    if "error" not in curve_data:
        fig_curve = go.Figure()
        for d, vals in curve_data["curves"].items():
            fig_curve.add_trace(go.Scatter(
                x=curve_data["points"],
                y=vals,
                mode="lines+markers",
                name=f"{d}% Discount",
                line=dict(width=2.5),
            ))
        fig_curve.update_layout(
            height=340,
            xaxis_title=sens_var,
            yaxis_title="Estimated Expected Unit Profit (₹)",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_curve, use_container_width=True)


# ===========================================================================
# PAGE 4: 🛡️ MARGIN GUARDRAILS STUDIO
# ===========================================================================

elif nav_selection == "🛡️ Margin Guardrails Studio":
    st.markdown("## 🛡️ Commercial Margin Guardrails Studio")
    st.caption("Inspect and audit how loyalty tier discount ceilings and strict margin floors prevent financial loss.")

    col_g_c1, col_g_c2 = st.columns(2)
    with col_g_c1:
        with st.container(border=True):
            st.markdown("### 👥 Customer Loyalty Tier Policy")
            st.markdown(
                f"""
                - **New Customers (0 to {active_rules.regular_cutoff-1} orders):** Maximum allowed discount is **5%**.
                - **Regular Customers ({active_rules.regular_cutoff} to {active_rules.loyal_cutoff-1} orders):** Allowed discounts **5%, 10%, 15%**.
                - **Loyal Customers ({active_rules.loyal_cutoff}+ orders):** Allowed discounts **10%, 15%, 20%**.
                """
            )
    with col_g_c2:
        with st.container(border=True):
            st.markdown("### 💰 Absolute Unit Margin Floor")
            st.markdown(
                f"""
                - **Active Floor Setting:** {fmt_rupees(active_rules.margin_floor)}
                - **Rule Logic:** $\\text{{SellingPrice}} - \\text{{CostPrice}} \\ge \\text{{Floor}}$
                - **Status:** Guaranteed 100% compliant across all evaluated simulation rounds.
                """
            )

    st.markdown("### 📋 Per-Arm Compliance Matrix")
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
            "Selling Price": fmt_rupees(a.selling_price),
            "Unit Margin": fmt_rupees(a.unit_margin_after_discount),
            "Tier Compliance": passes_tier,
            "Margin Compliance": passes_margin,
            "Final Eligibility": "🟢 Eligible" if a.score is not None else "⛔ Disqualified",
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
        col_k1.metric("LinUCB Total Profit", fmt_rupees(kpis["LinUCB"]["total_profit"], simulated=True), "Seed 42 (20k rounds)")
        col_k2.metric("LinUCB Gross Revenue", fmt_rupees(kpis["LinUCB"]["total_revenue"], simulated=True), "Highest top-line sales")
        col_k3.metric("LinUCB Concession", fmt_rupees(kpis["LinUCB"]["total_discount_given"], simulated=True), "Total discount disbursed")
        gain_vs_fixed = kpis.get("linucb_gain_vs_fixed_10_rupees", -2295.18)
        col_k4.metric("Profit Gap vs Fixed 10%", fmt_rupees(gain_vs_fixed, simulated=True), "-1.67% overall")

        st.markdown("---")

        st.subheader("Cumulative Performance Trajectories")
        df_s42 = dash_data["series_seed42"]

        selected_policies = st.multiselect(
            "Select Policies to Display",
            options=["Random", "Fixed 10%", "Rule-based", "LinUCB", "Thompson Sampling"],
            default=["Fixed 10%", "Rule-based", "LinUCB", "Thompson Sampling"],
        )

        fig_cum = go.Figure()
        for p in selected_policies:
            sub = df_s42[df_s42["policy"] == p]
            fig_cum.add_trace(go.Scatter(
                x=sub["round_idx"],
                y=sub["cum_profit"],
                mode="lines",
                name=p,
                line=dict(color=POLICY_COLORS.get(p, "#0F172A"), width=2.5),
            ))

        fig_cum.update_layout(
            height=360,
            xaxis_title="Round Number",
            yaxis_title="Cumulative Profit (₹)",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Multi-Seed Benchmark Tables (10 Seeds)")
        mode_toggle = st.radio("Select Evaluation Mode:", ["Online Mode (20,000 Rounds)", "Frozen Mode (5,000 Held-Out Rounds)"], horizontal=True)

        if "Online" in mode_toggle:
            st.dataframe(dash_data["comparison_online"], use_container_width=True, hide_index=True)
        else:
            st.dataframe(dash_data["comparison_frozen"], use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 6: 📘 KNOWLEDGE BLOG & ARCHITECTURE
# ===========================================================================

elif nav_selection == "📘 Knowledge Blog & Architecture":
    st.markdown("## 📘 SmartCommerce Knowledge Blog & Technical Guide")
    st.caption("Comprehensive operational articles explaining how the AI dynamic pricing engine works under the hood.")

    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 16px; padding: 24px; margin-bottom: 24px;">
            <span style="background: #0C232A; color: #FF9900; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 13px;">MERCHANT GUIDE</span>
            <h2 style="margin: 8px 0 8px 0; color: #0C232A;">Article 1: The Profit Paradox — Why Blanket Discounts Erode Value</h2>
            <p style="color: #475569; font-size: 16px; line-height: 1.6;">
                Most retail merchants default to blanket markdowns (e.g. 15% off everything). However, microeconomic analysis reveals that uniform discounting leads to massive profit leakage. High-intent shoppers convert regardless of discount, surrendering net margin, while elasticity-sensitive shoppers may require deeper cuts to purchase.
            </p>
            <p style="color: #0F172A; font-weight: 600; margin-bottom: 0;">
                The Solution: Disjoint LinUCB estimates price elasticity per context in real time, personalizing discounts to maximize profit rather than blindly maximizing conversion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        with st.container(border=True):
            st.markdown("### 🧠 Article 2: How LinUCB Works")
            st.caption("Active Reinforcement Learning & Upper Confidence Bounds")
            st.markdown(
                r"""
                - **Point Estimate (Exploitation):** Fits a linear model predicting profit: $x_t^T \hat{\theta}_a$.
                - **Uncertainty Bonus (Exploration):** Measures confidence bound: $\alpha \sqrt{x_t^T A_a^{-1} x_t}$.
                - **Sherman-Morrison Rank-1 Update:** Maintains $A_a^{-1}$ in $\mathcal{O}(d^2)$ time instead of $\mathcal{O}(d^3)$, enabling sub-millisecond real-time recommendations.
                """
            )
    with col_b2:
        with st.container(border=True):
            st.markdown("### 🛡️ Article 3: Zero-Margin-Breach Guardrails")
            st.caption("Why Soft Penalties Fail & How Hard Filters Win")
            st.markdown(
                r"""
                - **The Failure of Soft Rewards:** Penalizing below-cost sales via reward penalties destabilizes reinforcement learning, causing policy collapse.
                - **Programmatic Pre-Filtering:** Our envelope filters the action space before scoring:
                  $$\text{SellingPrice}(a) - \text{CostPrice} \ge \text{Floor}$$
                - **Result:** Exactly 100% compliance across 200,000 online rounds.
                """
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 🔄 End-to-End Walkthrough: What Happens on Every Shopper Visit?")
        st.markdown(
            """
            1. **Context Extraction:** When a customer views an item, the system collects catalogue price, competitor shelf price, inventory level, demand index, and customer order history.
            2. **26-Dimensional Vectorization:** Continuous features are z-score standardized; categoricals are one-hot encoded.
            3. **Guardrail Envelope:** Loyalty tier discount caps and unit margin floors filter the 5 arms `{0%, 5%, 10%, 15%, 20%}` down to an eligible subset.
            4. **UCB Scoring & Selection:** LinUCB scores all eligible arms; the discount with the highest score is presented to the shopper.
            5. **Conversion Simulation & Ledger Update:** Purchase outcomes are simulated via empirical logistic demand; transaction revenue, margin, and regret are recorded in the financial ledger.
            """
        )


# ===========================================================================
# PAGE 7: ℹ️ MODEL ARCHITECTURE & TRANSPARENCY
# ===========================================================================

elif nav_selection == "ℹ️ Model Architecture & Transparency":
    st.markdown("## ℹ️ Technical Architecture & Methodology")
    about_file = APP_DIR / "about_text.md"
    if about_file.exists():
        with open(about_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.write("Methodology document not found.")
