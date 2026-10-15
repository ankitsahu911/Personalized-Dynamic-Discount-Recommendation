"""
Developer script to build app/app_config.json from historical dataset and saved model metadata.
Run once during development.
"""

import os
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "historical_73k.pkl")
META_PATH = os.path.join(PROJECT_ROOT, "models", "linucb_v1_meta.json")
CONFIG_OUT = os.path.join(PROJECT_ROOT, "app", "app_config.json")


def build_config():
    print(f"Reading model metadata from {META_PATH}...")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Reading historical data from {DATA_PATH}...")
    if os.path.exists(DATA_PATH):
        df = pd.read_pickle(DATA_PATH)
    else:
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    # Build numeric limits & quantiles
    numeric_config = {
        "Price": {
            "min": 10.0,
            "max": 150.0,
            "default": round(float(df["Price"].median()), 2),
            "p01": round(float(df["Price"].quantile(0.01)), 2),
            "p99": round(float(df["Price"].quantile(0.99)), 2),
            "step": 1.0,
            "unit": "Rs",
            "label": "Retail Price",
            "help": "Base catalogue selling price of the item before discount.",
        },
        "CompetitorPricing": {
            "min": 10.0,
            "max": 150.0,
            "default": round(float(df["Competitor Pricing"].median()), 2),
            "p01": round(float(df["Competitor Pricing"].quantile(0.01)), 2),
            "p99": round(float(df["Competitor Pricing"].quantile(0.99)), 2),
            "step": 1.0,
            "unit": "Rs",
            "label": "Competitor Price",
            "help": "Observed market benchmark price from primary local competitor.",
        },
        "InventoryLevel": {
            "min": 10,
            "max": 600,
            "default": int(df["Inventory Level"].median()),
            "p01": int(df["Inventory Level"].quantile(0.01)),
            "p99": int(df["Inventory Level"].quantile(0.99)),
            "step": 5,
            "unit": "units",
            "label": "Current Stock Level",
            "help": "On-hand warehouse/shelf inventory units available.",
        },
        "PurchaseCount": {
            "min": 0,
            "max": 30,
            "default": int(df["PurchaseCount"].median()),
            "p01": 0,
            "p99": 15,
            "step": 1,
            "unit": "purchases",
            "label": "Lifetime Purchases",
            "help": "Total completed orders by this customer to date (determines Loyalty Tier).",
        },
        "RecencyDays": {
            "min": 0,
            "max": 180,
            "default": 14,
            "p01": 1,
            "p99": 120,
            "step": 1,
            "unit": "days",
            "label": "Recency (Days Since Last Order)",
            "help": "Number of days elapsed since the customer's previous transaction.",
        },
        "CurrentBasketValue": {
            "min": 20.0,
            "max": 500.0,
            "default": 150.0,
            "p01": 50.0,
            "p99": 450.0,
            "step": 5.0,
            "unit": "Rs",
            "label": "Current Basket Value",
            "help": "Total gross rupee value of items currently in shopper's cart.",
        },
        "CustomerPriceSensitivity": {
            "min": 0.55,
            "max": 1.55,
            "default": 1.00,
            "p01": 0.55,
            "p99": 1.50,
            "step": 0.01,
            "unit": "index",
            "label": "Customer Price Sensitivity",
            "help": "Estimated elasticity coefficient (1.0 = average; >1.0 = discount-seeking).",
        },
    }

    # Categorical options taken strictly from saved metadata
    cats = meta["preprocessing"]["categorical_values"]["Category"]
    regions = meta["preprocessing"]["categorical_values"]["Region"]
    seasons = meta["preprocessing"]["categorical_values"]["Seasonality"]
    tiers = meta["preprocessing"]["categorical_values"]["LoyaltyTier"]

    categorical_config = {
        "Category": {
            "options": cats,
            "default": "Electronics",
            "label": "Product Category",
            "help": "Merchandise department.",
        },
        "Region": {
            "options": regions,
            "default": "West",
            "label": "Store Region",
            "help": "Geographic retail district.",
        },
        "Seasonality": {
            "options": seasons,
            "default": "Spring",
            "label": "Retail Season",
            "help": "Calendar trading season.",
        },
        "DemandLevel": {
            "options": ["Low", "Medium", "High"],
            "default": "Medium",
            "label": "Demand Velocity",
            "help": "Product sales velocity relative to department peers (Low / Medium / High).",
        },
    }

    # Business rule constants & guardrails
    app_config = {
        "metadata": {
            "model_version": "linucb_v1",
            "model_type": meta.get("model_type", "DisjointLinUCB"),
            "training_rounds": meta.get("n_rounds", 20000),
            "seed": meta.get("main_seed", 42),
            "alpha": meta.get("alpha", 1.0),
            "context_dim": meta.get("context_dim", 26),
            "saved_at": meta.get("saved_at", ""),
        },
        "numeric_inputs": numeric_config,
        "categorical_inputs": categorical_config,
        "constants": {
            "ACTIONS": [0, 5, 10, 15, 20],
            "DEFAULT_REGULAR_CUTOFF": 3,
            "DEFAULT_LOYAL_CUTOFF": 10,
            "ASSUMED_MARGIN_RATE": 0.40,
            "DEFAULT_MARGIN_FLOOR": 0.0,
            "MAX_MARGIN_FLOOR": 50.0,
            "TIER_DISCOUNTS": {
                "New": [0, 5],
                "Regular": [5, 10, 15],
                "Loyal": [10, 15, 20],
            },
        },
    }

    os.makedirs(os.path.dirname(CONFIG_OUT), exist_ok=True)
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(app_config, f, indent=2)

    print(f"Successfully generated app configuration: {CONFIG_OUT}")
    return app_config


if __name__ == "__main__":
    build_config()
