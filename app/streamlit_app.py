"""
Interactive Retail Dynamic Discount Recommender
Streamlit Web Application — Dynamic Discount Studio
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
    page_title="Dynamic Discount Studio | Pricing Optimization",
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
# APPLICATION STYLES
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, .stDeployButton, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }
    .block-container {
        max-width: 1440px !important;
        padding: 0.55rem 2rem 3rem !important;
    }
    h1, h2, h3 {
        color: #102A43 !important;
        letter-spacing: 0 !important;
    }
    h1 { font-size: 2rem !important; font-weight: 750 !important; }
    h2 { font-size: 1.5rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.15rem !important; font-weight: 700 !important; }
    [data-testid="stAppViewContainer"] {
        background: #F7FAFC;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        min-width: 340px !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #1E3A4C !important;
        font-size: 0.92rem !important;
        font-weight: 650 !important;
        line-height: 1.35 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] {
        margin-bottom: 0.6rem;
    }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
        min-height: 2.55rem !important;
        font-size: 0.95rem !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%; border-radius: 6px; font-weight: 650;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.45rem !important;
        margin: 0.5rem 0 1.5rem !important;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 0.45rem 0.75rem !important;
        font-size: 0.9rem !important;
        font-weight: 650 !important;
        color: #0F172A !important;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: #0F766E !important;
        color: #FFFFFF !important;
        border-color: #0F766E !important;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {
        color: #FFFFFF !important;
    }
    .portal-hero {
        background: #103B4C;
        border-radius: 8px;
        padding: 2rem;
        color: #FFFFFF;
        margin-bottom: 1.25rem;
    }
    .portal-hero h1, .portal-hero h2, .portal-hero h3, .portal-hero p, .portal-hero span {
        color: #FFFFFF !important;
    }
    .brand-lockup { display: flex; align-items: center; gap: 0.7rem; margin: 0.2rem 0 1.1rem; }
    .brand-mark {
        display: inline-flex; align-items: center; justify-content: center; width: 2.15rem; height: 2.15rem;
        border-radius: 6px; background: #0F766E; color: #FFFFFF; font-size: 0.7rem; font-weight: 800;
        letter-spacing: 0.06em;
    }
    .brand-name { color: #102A43; font-size: 1rem; font-weight: 750; }
    .page-kicker { color: #0F766E; font-size: 0.76rem; font-weight: 750; letter-spacing: 0.07em; text-transform: uppercase; }
    .page-heading { color: #102A43; font-size: 1.85rem; font-weight: 750; margin: 0.15rem 0 0.35rem; }
    .page-description { color: #66788A; font-size: 1rem; margin: 0 0 1.25rem; }
    .insight-card { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 8px; padding: 1.05rem; }
    .insight-card--teal { border-top: 3px solid #0F766E; }
    .insight-card--blue { border-top: 3px solid #2563EB; }
    .insight-card--amber { border-top: 3px solid #D97706; }
    .table-note { color: #66788A; font-size: 0.9rem; margin: 0.25rem 0 0.75rem; }
    .login-shell { max-width: 460px; margin: 9vh auto 0; }
    .login-brand { color: #0F766E; font-weight: 800; font-size: 1.1rem; }
    .login-subtitle { color: #52616B; font-size: 1rem; margin: 0.4rem 0 1.5rem; }
    [data-testid="stForm"] { border: 1px solid #DCE3E8; border-radius: 8px; padding: 1.5rem; background: #FFFFFF; }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; }
    [data-testid="stMetricLabel"] p { color: #52616B !important; font-size: 0.82rem !important; }
    [data-testid="stMetricValue"] { color: #102A43 !important; font-size: 1.55rem !important; }
    .product-img-card {
        border-radius: 6px; width: 100%; height: 168px; object-fit: cover; display: block;
    }
    .stButton > button[kind="primary"] {
        background: #0F766E !important; color: #FFFFFF !important; border-color: #0F766E !important;
    }
    .stButton > button:not([kind="primary"]) { border-color: #9FB3C8 !important; color: #102A43 !important; }
    .stButton > button:focus-visible { outline: 3px solid #7DD3C7 !important; outline-offset: 2px !important; }
    [data-testid="stDataFrame"] { border: 1px solid #D9E2EC; border-radius: 8px; overflow: hidden; }
    @media (max-width: 800px) {
        .block-container { padding: 1rem !important; }
        .portal-hero { padding: 1.25rem; }
        [data-testid="stSidebar"] { min-width: 290px !important; }
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
    st.session_state["nav_page"] = "Store Overview & Catalog"

LEGACY_NAV_PAGES = {
    "🏠 Store Overview & Catalog": "Store Overview & Catalog",
    "🎯 Live Pricing Simulator": "Live Pricing Simulator",
    "🔮 Market Sensitivity & What-If Lab": "Market Sensitivity & What-If Lab",
    "🛡️ Margin Guardrails Studio": "Margin Guardrails Studio",
    "📊 Policy Performance & ROI Audits": "Policy Performance & ROI Audits",
    "📘 Knowledge Blog & Architecture": "Knowledge Blog & Architecture",
    "ℹ️ Model Architecture & Transparency": "Model Architecture & Transparency",
}
st.session_state["nav_page"] = LEGACY_NAV_PAGES.get(
    st.session_state["nav_page"], st.session_state["nav_page"]
)


def handle_login(username_val: str):
    """Sets session state to logged in."""
    st.session_state["authenticated"] = True
    st.session_state["user_name"] = "Sujan" if not username_val else username_val.split("@")[0].title()
    st.session_state["user_role"] = "Pricing Operations Director"
    st.session_state["nav_page"] = "Store Overview & Catalog"


def handle_logout():
    """Logs out and resets authentication state."""
    st.session_state["authenticated"] = False
    st.session_state["nav_page"] = "Store Overview & Catalog"


def render_page_heading(kicker: str, title: str, description: str) -> None:
    """Render a consistent, compact page header with the application mark."""
    st.markdown(
        f"""
        <div class="brand-lockup">
            <span class="brand-mark">DD</span>
            <span class="brand-name">Dynamic Discount Studio</span>
        </div>
        <div class="page-kicker">{html.escape(kicker)}</div>
        <div class="page-heading">{html.escape(title)}</div>
        <p class="page-description">{html.escape(description)}</p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# AMAZON SMARTBIZ MERCHANT SIGN-IN PAGE
# ---------------------------------------------------------------------------

if not st.session_state["authenticated"]:
    _, login_column, _ = st.columns([1, 1.35, 1])
    with login_column:
        st.markdown('<div class="login-brand">DYNAMIC DISCOUNT STUDIO</div>', unsafe_allow_html=True)
        st.title("Merchant sign in")
        st.markdown(
            '<p class="login-subtitle">AI-powered dynamic pricing and merchandising for safer, more profitable discount decisions.</p>',
            unsafe_allow_html=True,
        )
        with st.form("merchant_login", clear_on_submit=False):
            login_email = st.text_input("Email address", placeholder="you@store.com", key="login_email_input")
            st.text_input("Password", type="password", key="login_pass_input")
            st.checkbox("Keep me signed in", value=True, key="keep_signed_in")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                if login_email.strip():
                    handle_login(login_email)
                    st.rerun()
                st.error("Enter an email address to continue.")
        if st.button("Use pricing director demo", use_container_width=True):
            handle_login("sujan@smartbiz.nexus")
            st.rerun()

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
    st.session_state["nav_page"] = "Live Pricing Simulator"


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
    st.title(f"Welcome back, {st.session_state['user_name']}")
    st.caption("Dynamic Discount Studio · Disjoint LinUCB pricing engine")
with col_h2:
    col_u_pill, col_u_out = st.columns([3, 2])
    with col_u_pill:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 4px;">
                <strong style="color: #0C232A; font-size: 16px;">{st.session_state['user_name']}</strong><br>
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
        "Store Overview & Catalog",
        "Live Pricing Simulator",
        "Market Sensitivity & What-If Lab",
        "Margin Guardrails Studio",
        "Policy Performance & ROI Audits",
        "Knowledge Blog & Architecture",
        "Model Architecture & Transparency",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)


# ---------------------------------------------------------------------------
# SIDEBAR: ACTIVE MERCHANDISING CONTROLS (LARGE, HIGH READABILITY)
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Merchandising controls")
st.sidebar.caption("Configure pricing, inventory, demand, and shopper settings.")

# Group 1: Product Inputs
st.sidebar.markdown("### Product & commercials")
price_val = st.sidebar.number_input(
    "Base Price (₹)",
    min_value=10.0,
    max_value=150.0,
    step=1.0,
    key="input_Price",
    help="Catalogue selling price of the item before discount.",
)
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
st.sidebar.markdown("### Inventory & demand")
inv_val = st.sidebar.number_input(
    "Inventory Stock",
    min_value=10,
    max_value=600,
    step=5,
    key="input_InventoryLevel",
    help="Available stock count in warehouse.",
)
demand_val = st.sidebar.selectbox(
    "Demand Velocity",
    options=["Low", "Medium", "High"],
    key="input_DemandLevel",
    help="Observed sales velocity level.",
)
season_val = st.sidebar.selectbox(
    "Seasonality",
    options=["Spring", "Summer", "Autumn", "Winter"],
    key="input_Seasonality",
)
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
st.sidebar.markdown("### Shopper loyalty")
purchases_val = st.sidebar.number_input(
    "Orders to Date",
    min_value=0,
    max_value=30,
    step=1,
    key="input_PurchaseCount",
    help="Customer's lifetime completed orders (governs loyalty tier).",
)
recency_val = st.sidebar.number_input(
    "Recency (Days)",
    min_value=0,
    max_value=180,
    step=1,
    key="input_RecencyDays",
    help="Days since previous order.",
)
basket_val = st.sidebar.number_input(
    "Basket Value (₹)",
    min_value=20.0,
    max_value=500.0,
    step=5.0,
    key="input_CurrentBasketValue",
    help="Total cart value in rupees.",
)
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

st.sidebar.markdown("### Guardrail rules")
margin_floor_slider = st.sidebar.slider(
    "Unit Margin Floor (₹)",
    min_value=0.0,
    max_value=MAX_MARGIN_FLOOR,
    step=0.5,
    key="guardrail_margin_floor",
    help="Discounts yielding margin below this floor are strictly blocked.",
)

reg_cutoff_input = st.sidebar.number_input(
    "Regular tier cutoff",
    min_value=1,
    max_value=15,
    step=1,
    key="guardrail_reg_cutoff",
)
loyal_cutoff_input = st.sidebar.number_input(
    "Loyal tier cutoff",
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

if nav_selection == "Store Overview & Catalog":
    # Hero callout
    st.markdown(
        """
        <div class="portal-hero">
            <h1 style="margin: 0 0 10px 0;">Build & Scale Your Store with AI Pricing</h1>
            <p style="margin: 0 0 20px 0;">
                Live contextual reinforcement learning dynamically balances margin protection and purchase conversion across all SKU categories.
            </p>
            <div style="display: flex; gap: 14px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px;">LinUCB active</span>
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px;">Margin floor enforced</span>
                <span style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 9999px; font-weight: 700; font-size: 14px;">24.3% conversion rate</span>
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
    col_dash_left, col_dash_right = st.columns([1.65, 1])

    with col_dash_left:
        with st.container(border=True):
            st.markdown("### Revenue & cumulative profit trend")
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
                    height=340,
                    margin=dict(l=44, r=22, t=28, b=42),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    xaxis=dict(title="Round Number", showgrid=True, gridcolor="#F1F5F9"),
                    yaxis=dict(title="Cumulative Profit (₹)", showgrid=True, gridcolor="#F1F5F9"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                    font=dict(color="#334E68"),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Simulation trend assets synchronizing...")

    with col_dash_right:
        with st.container(border=True):
            st.markdown("### Inventory & margin watchlist")
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
            st.markdown("### Conversion lift by category")
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
                height=285,
                margin=dict(l=44, r=20, t=20, b=35),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                yaxis_title="Conversion Rate (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_g_right:
        with st.container(border=True):
            st.markdown("### Customer tier margin distribution")
            st.caption("Distribution of transactions governed by loyalty discount ceilings.")
            tier_df = pd.DataFrame({
                "Tier": ["New Shoppers (0-5% Max)", "Regular Customers (5-15% Max)", "Loyal Shoppers (10-20% Max)"],
                "Volume": [35, 45, 20],
                "Color": ["#00A896", "#2563EB", "#FF9900"],
            })
            fig_pie = go.Figure(data=[go.Pie(labels=tier_df["Tier"], values=tier_df["Volume"], hole=.45, marker_colors=tier_df["Color"])])
            fig_pie.update_layout(
                height=285,
                margin=dict(l=20, r=20, t=20, b=45),
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Featured Product Catalog with 16 SKUs, Filters, and Real Photos
    st.markdown("## 🏷️ Featured Store Catalog — 1-Click Discount Optimizer")
    st.caption("Filter by category or search products, then choose Analyze discount to load an item into the pricing engine.")

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
                            f"Analyze {prod['category']} discount",
                            key=f"btn_opt_{prod['id']}",
                            use_container_width=True,
                            on_click=load_product_into_simulator,
                            args=(prod,),
                        )


# ===========================================================================
# PAGE 2: 🎯 LIVE PRICING SIMULATOR
# ===========================================================================

elif nav_selection == "Live Pricing Simulator":
    render_page_heading(
        "Live decision", "Contextual pricing recommendation",
        "Real-time LinUCB scoring across eligible discounts, constrained by loyalty and margin rules.",
    )

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
            st.markdown("### UCB score by discount arm")
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
            st.markdown("### Decision logic")
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
    st.markdown("### Detailed arm diagnostics")
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

elif nav_selection == "Market Sensitivity & What-If Lab":
    render_page_heading(
        "Scenario planning", "Market sensitivity lab",
        "Compare competitor price shocks, inventory changes, and seasonal shifts before committing a discount.",
    )

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
            st.markdown("### What-if scenario")
            st.write(f"**Recommended Discount:** {scen_rec.chosen_discount}%")
            st.write(f"**Effective Price:** {fmt_rupees(scen_rec.discounted_price)}")
            st.write(f"**Post-Discount Margin:** {fmt_rupees(scen_rec.unit_margin_after_discount)}")

    st.markdown("### Sensitivity response curves")
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

elif nav_selection == "Margin Guardrails Studio":
    render_page_heading(
        "Safety controls", "Commercial margin guardrails",
        "Inspect how loyalty discount menus and the margin floor filter unsafe offers before the model makes a recommendation.",
    )

    eligible_count = len(rec_result.eligible_discounts)
    blocked_count = len(ACTIONS) - eligible_count
    margin_headroom = max(0.0, rec_result.unit_margin_after_discount - active_rules.margin_floor)
    guardrail_kpis = st.columns(4)
    guardrail_kpis[0].metric("Current shopper tier", rec_result.tier_for_rules)
    guardrail_kpis[1].metric("Eligible discounts", f"{eligible_count} of {len(ACTIONS)}")
    guardrail_kpis[2].metric("Blocked discounts", str(blocked_count))
    guardrail_kpis[3].metric("Recommended margin headroom", fmt_rupees(margin_headroom))

    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)

    col_g_c1, col_g_c2 = st.columns(2)
    with col_g_c1:
        st.markdown('<div class="insight-card insight-card--blue">', unsafe_allow_html=True)
        st.markdown("### Loyalty discount menu")
        st.caption("The shopper can only be offered discounts assigned to their current tier.")
        st.markdown(
            f"""
            - **New customers (0 to {active_rules.regular_cutoff - 1} orders):** 0% or 5%
            - **Regular customers ({active_rules.regular_cutoff} to {active_rules.loyal_cutoff - 1} orders):** 5%, 10%, or 15%
            - **Loyal customers ({active_rules.loyal_cutoff}+ orders):** 10%, 15%, or 20%
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_g_c2:
        st.markdown('<div class="insight-card insight-card--amber">', unsafe_allow_html=True)
        st.markdown("### Margin protection rule")
        st.caption("An offer is removed if it drops the item's post-discount unit margin below the chosen floor.")
        st.markdown(
            f"""
            - **Active margin floor:** {fmt_rupees(active_rules.margin_floor)}
            - **Cost basis used:** {fmt_rupees(rec_result.raw_context['CostPrice'])}
            - **Recommended post-discount margin:** {fmt_rupees(rec_result.unit_margin_after_discount)}
            - **Rule:** selling price minus cost price must remain at or above the floor.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Per-arm compliance matrix")
    st.markdown('<p class="table-note">Teal rows are available for scoring. Amber rows are blocked by the loyalty menu; red rows would fail the margin floor.</p>', unsafe_allow_html=True)
    g_full_rows = []
    for a in rec_result.arms_table:
        tier_ok = a.discount in TIER_DISCOUNTS[rec_result.tier_for_rules]
        passes_tier = "Allowed" if tier_ok else "Blocked by tier"
        margin_diff = a.unit_margin_after_discount - active_rules.margin_floor
        margin_ok = margin_diff >= 0
        passes_margin = "Passes" if margin_ok else "Blocked by floor"
        g_full_rows.append({
            "Discount Arm": f"{a.discount}%",
            "Selling Price": fmt_rupees(a.selling_price),
            "Unit Margin": fmt_rupees(a.unit_margin_after_discount),
            "Tier Compliance": passes_tier,
            "Margin Compliance": passes_margin,
            "Final Eligibility": "Eligible" if a.score is not None else "Disqualified",
        })
    guardrail_df = pd.DataFrame(g_full_rows)

    def style_guardrail_table(row: pd.Series) -> list[str]:
        if row["Final Eligibility"] == "Eligible":
            color = "background-color: #E6FFFA; color: #0F4C45; font-weight: 600;"
        elif row["Margin Compliance"] == "Blocked by floor":
            color = "background-color: #FEE2E2; color: #991B1B;"
        else:
            color = "background-color: #FFF7E6; color: #9A5B00;"
        return [color] * len(row)

    st.dataframe(
        guardrail_df.style.apply(style_guardrail_table, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Discount Arm": st.column_config.TextColumn("Discount", width="small"),
            "Selling Price": st.column_config.TextColumn("Selling price", width="small"),
            "Unit Margin": st.column_config.TextColumn("Unit margin", width="small"),
            "Tier Compliance": st.column_config.TextColumn("Tier rule", width="medium"),
            "Margin Compliance": st.column_config.TextColumn("Margin floor", width="medium"),
            "Final Eligibility": st.column_config.TextColumn("Decision", width="small"),
        },
    )

    with st.expander("How the guardrails protect this recommendation"):
        st.markdown(
            f"""
            The current shopper is in the **{rec_result.tier_for_rules}** tier, so the loyalty policy narrows the action menu first.
            The margin floor then checks the remaining actions using the active cost basis. Only after both checks does LinUCB score each safe option.
            This prevents below-floor markdowns even when a larger discount would otherwise receive a high model score.
            """
        )


# ===========================================================================
# PAGE 5: 📊 POLICY PERFORMANCE & ROI AUDITS
# ===========================================================================

elif nav_selection == "Policy Performance & ROI Audits":
    render_page_heading(
        "Evaluation", "Policy performance and ROI audits",
        "Compare policies on the same pre-generated customer streams across ten random seeds.",
    )

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
            height=430,
            margin=dict(l=58, r=28, t=58, b=56),
            xaxis_title="Round Number",
            yaxis_title="Cumulative Profit (₹)",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#334E68"),
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Multi-Seed Benchmark Tables (10 Seeds)")
        mode_toggle = st.radio("Select Evaluation Mode:", ["Online Mode (20,000 Rounds)", "Frozen Mode (5,000 Held-Out Rounds)"], horizontal=True)

        if "Online" in mode_toggle:
            benchmark_df = dash_data["comparison_online"]
        else:
            benchmark_df = dash_data["comparison_frozen"]

        def style_benchmark_table(row: pd.Series) -> list[str]:
            policy = str(row.iloc[0])
            if policy == "LinUCB":
                color = "background-color: #E6FFFA; color: #0F4C45; font-weight: 650;"
            elif policy == "Fixed 10%":
                color = "background-color: #FFF7E6; color: #8A4B00;"
            elif policy == "Thompson Sampling":
                color = "background-color: #EFF6FF; color: #1E429F;"
            else:
                color = "background-color: #FFFFFF; color: #243B53;"
            return [color] * len(row)

        st.dataframe(
            benchmark_df.style.apply(style_benchmark_table, axis=1),
            use_container_width=True,
            hide_index=True,
            height=270,
        )


# ===========================================================================
# PAGE 6: 📘 KNOWLEDGE BLOG & ARCHITECTURE
# ===========================================================================

elif nav_selection == "Knowledge Blog & Architecture":
    render_page_heading(
        "Project guide", "How the pricing engine works",
        "Open a topic below during your demo to explain the system at the right level of detail.",
    )

    with st.expander("Why a uniform discount can reduce profit", expanded=True):
        st.markdown(
            """
            A fixed markdown gives away margin to high-intent shoppers who may have purchased anyway, while still failing to help price-sensitive shoppers in the moments where an incentive matters.
            The application instead evaluates the product, competitor, inventory, demand, promotion, region, season, and shopper context before selecting an eligible discount.
            """
        )

    with st.expander("How LinUCB chooses a discount"):
        st.markdown(
            r"""
            - **Point estimate:** predicts expected unit profit for a discount in a similar context: $x_t^T \hat{\theta}_a$.
            - **Uncertainty bonus:** gives less-tested actions a fair chance: $\alpha \sqrt{x_t^T A_a^{-1} x_t}$.
            - **Final decision:** the eligible discount with the highest combined score is selected.

            The model is a contextual multi-armed bandit: it learns from outcomes over individual pricing decisions rather than planning a multi-step sequence.
            """
        )

    with st.expander("How commercial guardrails work"):
        st.markdown(
            r"""
            Before LinUCB scores anything, the guardrail layer filters the five actions `{0%, 5%, 10%, 15%, 20%}`.
            Loyalty tiers define which discounts may be offered, and the unit-margin rule checks:

            $$\text{Selling price} - \text{Cost price} \geq \text{Margin floor}$$

            This makes guardrail compliance deterministic rather than hoping the learning model discovers it through a reward penalty.
            """
        )

    with st.expander("End-to-end shopper visit walkthrough"):
        st.markdown(
            """
            1. Product and shopper context is collected from the merchandising controls.
            2. Numeric inputs are standardized and categorical inputs are encoded into the model feature vector.
            3. Tier and margin guardrails remove unsafe discount options.
            4. LinUCB scores each remaining discount using expected profit plus an uncertainty bonus.
            5. The app displays the selected price, margin, diagnostics, and scenario analysis.
            """
        )


# ===========================================================================
# PAGE 7: ℹ️ MODEL ARCHITECTURE & TRANSPARENCY
# ===========================================================================

elif nav_selection == "Model Architecture & Transparency":
    render_page_heading(
        "Transparency", "Technical architecture and methodology",
        "Reference notes on the data pipeline, guardrails, and trained recommendation model.",
    )
    about_file = APP_DIR / "about_text.md"
    if about_file.exists():
        with open(about_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.write("Methodology document not found.")
