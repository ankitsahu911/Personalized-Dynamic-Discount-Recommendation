"""
Application Configuration and Input Schema Loader
Single source of truth for UI parameters, defaults, and financial formatting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "app_config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing app configuration file at {CONFIG_PATH}. Run scripts/build_app_config.py first.")

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    APP_CONFIG = json.load(_f)

# Global constants
ACTIONS: List[int] = APP_CONFIG["constants"]["ACTIONS"]
DEFAULT_REGULAR_CUTOFF: int = int(APP_CONFIG["constants"]["DEFAULT_REGULAR_CUTOFF"])
DEFAULT_LOYAL_CUTOFF: int = int(APP_CONFIG["constants"]["DEFAULT_LOYAL_CUTOFF"])
ASSUMED_MARGIN_RATE: float = float(APP_CONFIG["constants"]["ASSUMED_MARGIN_RATE"])
DEFAULT_MARGIN_FLOOR: float = float(APP_CONFIG["constants"]["DEFAULT_MARGIN_FLOOR"])
MAX_MARGIN_FLOOR: float = float(APP_CONFIG["constants"]["MAX_MARGIN_FLOOR"])
TIER_DISCOUNTS: Dict[str, List[int]] = APP_CONFIG["constants"]["TIER_DISCOUNTS"]

MODEL_INFO: Dict[str, Any] = APP_CONFIG.get("metadata", {})
NUMERIC_INPUTS: Dict[str, Any] = APP_CONFIG.get("numeric_inputs", {})
CATEGORICAL_INPUTS: Dict[str, Any] = APP_CONFIG.get("categorical_inputs", {})

# Standard schema for iterating and generating form widgets
INPUT_SCHEMA: List[Dict[str, Any]] = [
    {
        "key": "Price",
        "group": "Product",
        "type": "float",
        "min": NUMERIC_INPUTS["Price"]["min"],
        "max": NUMERIC_INPUTS["Price"]["max"],
        "default": NUMERIC_INPUTS["Price"]["default"],
        "step": NUMERIC_INPUTS["Price"]["step"],
        "unit": "Rs",
        "label": NUMERIC_INPUTS["Price"]["label"],
        "help": NUMERIC_INPUTS["Price"]["help"],
        "p01": NUMERIC_INPUTS["Price"]["p01"],
        "p99": NUMERIC_INPUTS["Price"]["p99"],
    },
    {
        "key": "Category",
        "group": "Product",
        "type": "select",
        "options": CATEGORICAL_INPUTS["Category"]["options"],
        "default": CATEGORICAL_INPUTS["Category"]["default"],
        "label": CATEGORICAL_INPUTS["Category"]["label"],
        "help": CATEGORICAL_INPUTS["Category"]["help"],
    },
    {
        "key": "CompetitorPricing",
        "group": "Product",
        "type": "float",
        "min": NUMERIC_INPUTS["CompetitorPricing"]["min"],
        "max": NUMERIC_INPUTS["CompetitorPricing"]["max"],
        "default": NUMERIC_INPUTS["CompetitorPricing"]["default"],
        "step": NUMERIC_INPUTS["CompetitorPricing"]["step"],
        "unit": "Rs",
        "label": NUMERIC_INPUTS["CompetitorPricing"]["label"],
        "help": NUMERIC_INPUTS["CompetitorPricing"]["help"],
        "p01": NUMERIC_INPUTS["CompetitorPricing"]["p01"],
        "p99": NUMERIC_INPUTS["CompetitorPricing"]["p99"],
    },
    {
        "key": "InventoryLevel",
        "group": "Product",
        "type": "int",
        "min": NUMERIC_INPUTS["InventoryLevel"]["min"],
        "max": NUMERIC_INPUTS["InventoryLevel"]["max"],
        "default": NUMERIC_INPUTS["InventoryLevel"]["default"],
        "step": NUMERIC_INPUTS["InventoryLevel"]["step"],
        "unit": "units",
        "label": NUMERIC_INPUTS["InventoryLevel"]["label"],
        "help": NUMERIC_INPUTS["InventoryLevel"]["help"],
        "p01": NUMERIC_INPUTS["InventoryLevel"]["p01"],
        "p99": NUMERIC_INPUTS["InventoryLevel"]["p99"],
    },
    {
        "key": "DemandLevel",
        "group": "Product",
        "type": "select_slider",
        "options": ["Low", "Medium", "High"],
        "default": "Medium",
        "label": "Demand Velocity",
        "help": "Product sales velocity relative to department peers. Low (0.2), Medium (0.5), High (0.8).",
    },
    {
        "key": "Promotion",
        "group": "Product",
        "type": "toggle",
        "default": False,
        "label": "Promotional Event Active",
        "help": "Toggle whether this product is featured in an active store-wide promotion banner.",
    },
    {
        "key": "Region",
        "group": "Context",
        "type": "select",
        "options": CATEGORICAL_INPUTS["Region"]["options"],
        "default": CATEGORICAL_INPUTS["Region"]["default"],
        "label": CATEGORICAL_INPUTS["Region"]["label"],
        "help": CATEGORICAL_INPUTS["Region"]["help"],
    },
    {
        "key": "Seasonality",
        "group": "Context",
        "type": "select",
        "options": CATEGORICAL_INPUTS["Seasonality"]["options"],
        "default": CATEGORICAL_INPUTS["Seasonality"]["default"],
        "label": CATEGORICAL_INPUTS["Seasonality"]["label"],
        "help": CATEGORICAL_INPUTS["Seasonality"]["help"],
    },
    {
        "key": "PurchaseCount",
        "group": "Customer",
        "type": "int",
        "min": NUMERIC_INPUTS["PurchaseCount"]["min"],
        "max": NUMERIC_INPUTS["PurchaseCount"]["max"],
        "default": NUMERIC_INPUTS["PurchaseCount"]["default"],
        "step": NUMERIC_INPUTS["PurchaseCount"]["step"],
        "unit": "orders",
        "label": NUMERIC_INPUTS["PurchaseCount"]["label"],
        "help": NUMERIC_INPUTS["PurchaseCount"]["help"],
        "p01": NUMERIC_INPUTS["PurchaseCount"]["p01"],
        "p99": NUMERIC_INPUTS["PurchaseCount"]["p99"],
    },
    {
        "key": "RecencyDays",
        "group": "Customer",
        "type": "int",
        "min": NUMERIC_INPUTS["RecencyDays"]["min"],
        "max": NUMERIC_INPUTS["RecencyDays"]["max"],
        "default": NUMERIC_INPUTS["RecencyDays"]["default"],
        "step": NUMERIC_INPUTS["RecencyDays"]["step"],
        "unit": "days",
        "label": NUMERIC_INPUTS["RecencyDays"]["label"],
        "help": NUMERIC_INPUTS["RecencyDays"]["help"],
        "p01": NUMERIC_INPUTS["RecencyDays"]["p01"],
        "p99": NUMERIC_INPUTS["RecencyDays"]["p99"],
    },
    {
        "key": "CurrentBasketValue",
        "group": "Customer",
        "type": "float",
        "min": NUMERIC_INPUTS["CurrentBasketValue"]["min"],
        "max": NUMERIC_INPUTS["CurrentBasketValue"]["max"],
        "default": NUMERIC_INPUTS["CurrentBasketValue"]["default"],
        "step": NUMERIC_INPUTS["CurrentBasketValue"]["step"],
        "unit": "Rs",
        "label": NUMERIC_INPUTS["CurrentBasketValue"]["label"],
        "help": NUMERIC_INPUTS["CurrentBasketValue"]["help"],
        "p01": NUMERIC_INPUTS["CurrentBasketValue"]["p01"],
        "p99": NUMERIC_INPUTS["CurrentBasketValue"]["p99"],
    },
    {
        "key": "CustomerPriceSensitivity",
        "group": "Customer",
        "type": "float",
        "min": NUMERIC_INPUTS["CustomerPriceSensitivity"]["min"],
        "max": NUMERIC_INPUTS["CustomerPriceSensitivity"]["max"],
        "default": NUMERIC_INPUTS["CustomerPriceSensitivity"]["default"],
        "step": NUMERIC_INPUTS["CustomerPriceSensitivity"]["step"],
        "unit": "elasticity",
        "label": NUMERIC_INPUTS["CustomerPriceSensitivity"]["label"],
        "help": NUMERIC_INPUTS["CustomerPriceSensitivity"]["help"],
        "p01": NUMERIC_INPUTS["CustomerPriceSensitivity"]["p01"],
        "p99": NUMERIC_INPUTS["CustomerPriceSensitivity"]["p99"],
    },
]

# Formatting helpers
USE_INDIAN_GROUPING: bool = False


def fmt_rupees(amount: float, decimals: int = 2, simulated: bool = False) -> str:
    """Format rupee values with currency symbol, decimals, and thousands separators."""
    if amount is None:
        return "N/A"
    if USE_INDIAN_GROUPING:
        # Indian grouping (12,34,567.00)
        s = f"{amount:.{decimals}f}"
        parts = s.split(".")
        int_part = parts[0]
        dec_part = ("." + parts[1]) if len(parts) > 1 else ""
        if len(int_part) <= 3:
            formatted_int = int_part
        else:
            last3 = int_part[-3:]
            remaining = int_part[:-3]
            groups = []
            while len(remaining) > 2:
                groups.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.insert(0, remaining)
            formatted_int = ",".join(groups) + "," + last3
        formatted = f"₹{formatted_int}{dec_part}"
    else:
        formatted = f"₹{amount:,.{decimals}f}"
    return f"{formatted} (simulated)" if simulated else formatted


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format percentages."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def get_tier_from_purchases(purchases: int, reg_cutoff: int = DEFAULT_REGULAR_CUTOFF, loyal_cutoff: int = DEFAULT_LOYAL_CUTOFF) -> str:
    """Deterministic tier classification from lifetime purchase count."""
    if purchases < reg_cutoff:
        return "New"
    elif purchases < loyal_cutoff:
        return "Regular"
    else:
        return "Loyal"
