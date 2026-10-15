"""
data_loader.py — Real Dataset Lookup for the Simulator
Retail Discount-Recommendation Bandit Project

Loads the real 73,100-row dataset once at import time and builds a fast
in-memory lookup table keyed by (Category, LoyaltyTier, DemandLevel, Promotion).

Each bucket holds a list of real rows. When the simulator needs a purchase
probability and profit for a given context + discount, it:
  1. Looks up the matching bucket
  2. Picks one row at random from it (random sampling, not nearest-neighbour)
  3. Returns that row's real Prob_at_X and Profit_at_X values

Why random sampling instead of exact lookup?
    Our live_generator produces synthetic contexts (not real row identifiers),
    so there is no exact match. Randomly sampling from the matching bucket
    preserves the real statistical distribution of outcomes for that
    (Category, Tier, Demand, Promotion) combination, which is exactly what
    we want: realistic variance in outcomes, not a deterministic lookup.

This file is imported by simulator.py. It reads the Excel file once and
caches the lookup table -- subsequent calls do NOT re-read the file.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CONFIG - update this path if the dataset moves
# ---------------------------------------------------------------------------
DATASET_PATH = r"C:\Users\anknk\Downloads\retail_store_inventory_with_simulator.xlsx"
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lookup_table.pkl")

DISCOUNT_LEVELS = [0, 5, 10, 15, 20]

# Column name -> index mapping (0-indexed), confirmed from dataset inspection
_COL = {
    "Category":                  3,
    "LoyaltyTier":               20,
    "Promotion":                 11,
    "DemandLevel":               27,
    "CustomerPriceSensitivity":  22,
    "CostPrice":                 31,
    "Prob_at_0":                 42,
    "Profit_at_0":               43,
    "Prob_at_5":                 45,
    "Profit_at_5":               46,
    "Prob_at_10":                48,
    "Profit_at_10":              49,
    "Prob_at_15":                51,
    "Profit_at_15":              52,
    "Prob_at_20":                54,
    "Profit_at_20":              55,
    "OracleBestDiscount":        58,
    "OracleBestProfit":          59,
}

# Each lookup entry is a dict: {discount: (prob, profit), ..., "oracle": (best_disc, best_profit)}
_LookupEntry = Dict

# The lookup table: (category, tier, demand_level, promotion) -> list of entries
_LookupTable = Dict[Tuple, List[_LookupEntry]]

# Module-level cache - loaded once, reused every call
_TABLE: Optional[_LookupTable] = None


def _demand_level_from_float(demand_index: float) -> str:
    """
    Convert live_generator's float DemandIndex (0-1) to the dataset's
    categorical DemandLevel (Low / Medium / High).

    Mapping chosen to match the dataset's near-uniform distribution:
      [0.00, 0.34) -> High  (high DemandIndex = high demand)
      [0.34, 0.67) -> Medium
      [0.67, 1.00] -> Low
    """
    if demand_index >= 0.67:
        return "Low"
    elif demand_index >= 0.34:
        return "Medium"
    else:
        return "High"


def _load_table() -> _LookupTable:
    """Read the Excel file (or cache) and build the lookup table. Called once."""
    import pickle

    if os.path.exists(CACHE_PATH):
        print(f"[data_loader] Loading cached lookup table from {CACHE_PATH}...")
        try:
            with open(CACHE_PATH, "rb") as f:
                table = pickle.load(f)
            total_rows = sum(len(v) for v in table.values())
            print(f"[data_loader] Loaded {total_rows:,} rows from cache ({len(table)} buckets).")
            return table
        except Exception as e:
            print(f"[data_loader] Failed to load cache ({e}), reloading from Excel...")

    import openpyxl

    print("[data_loader] Loading dataset from Excel... (once only)")
    wb = openpyxl.load_workbook(DATASET_PATH, read_only=True, data_only=True)
    ws = wb.active

    table: _LookupTable = defaultdict(list)
    row_iter = ws.iter_rows(values_only=True)
    next(row_iter)  # skip header

    for row in row_iter:
        try:
            category   = str(row[_COL["Category"]])
            tier       = str(row[_COL["LoyaltyTier"]])
            demand     = str(row[_COL["DemandLevel"]])
            promotion  = int(row[_COL["Promotion"]] or 0)

            entry = {
                "cps": float(row[_COL["CustomerPriceSensitivity"]] or 1.0),
            }
            for disc in DISCOUNT_LEVELS:
                prob_key   = f"Prob_at_{disc}"
                profit_key = f"Profit_at_{disc}"
                prob   = row[_COL[prob_key]]
                profit = row[_COL[profit_key]]
                if prob is not None and profit is not None:
                    entry[disc] = (float(prob), float(profit))

            oracle_disc   = row[_COL["OracleBestDiscount"]]
            oracle_profit = row[_COL["OracleBestProfit"]]
            if oracle_disc is not None:
                entry["oracle"] = (int(oracle_disc), float(oracle_profit or 0.0))

            key = (category, tier, demand, promotion)
            table[key].append(entry)

        except (TypeError, ValueError):
            continue  # skip malformed rows

    wb.close()
    res_table = dict(table)
    total_rows = sum(len(v) for v in res_table.values())
    print(f"[data_loader] Loaded {total_rows:,} rows into {len(res_table)} buckets.")

    # Save cache
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(res_table, f)
        print(f"[data_loader] Saved fast cache to {CACHE_PATH}")
    except Exception as e:
        print(f"[data_loader] Warning: could not write cache file: {e}")

    return res_table


def get_table() -> _LookupTable:
    """Return the lookup table, loading it from disk the first time."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _load_table()
    return _TABLE


def lookup(
    category: str,
    loyalty_tier: str,
    demand_index: float,
    promotion: int,
    discount: int,
) -> Tuple[float, float]:
    """
    Return (purchase_probability, profit) for the given context and discount
    by sampling a real row from the matching bucket.

    Falls back gracefully if no exact bucket match:
      1. Try exact (category, tier, demand, promo)
      2. Try (category, tier, demand) ignoring promotion
      3. Try (category, demand) ignoring tier and promotion
      4. Last resort: sample any row in the table

    Parameters
    ----------
    category     : str   e.g. "Electronics"
    loyalty_tier : str   "New" | "Regular" | "Loyal"
    demand_index : float 0-1 (from live_generator's DemandIndex field)
    promotion    : int   0 or 1
    discount     : int   one of [0, 5, 10, 15, 20]

    Returns
    -------
    (prob, profit) : Tuple[float, float]
    """
    table = get_table()
    demand_level = _demand_level_from_float(demand_index)

    # Normalise tier: "" or unknown → "New"
    if not loyalty_tier or loyalty_tier not in ("New", "Regular", "Loyal"):
        loyalty_tier = "New"

    # Try increasingly broad fallback keys
    fallback_keys = [
        (category, loyalty_tier, demand_level, promotion),
        (category, loyalty_tier, demand_level, 1 - promotion),  # flip promotion
        (category, "Regular",    demand_level, promotion),       # relax tier
        (category, "Regular",    demand_level, 1 - promotion),
        (category, "New",        demand_level, promotion),
    ]

    bucket = None
    for key in fallback_keys:
        if key in table and table[key]:
            bucket = table[key]
            break

    if bucket is None:
        # Last resort: any row with this category
        all_rows = [entry for k, v in table.items() if k[0] == category for entry in v]
        if all_rows:
            bucket = all_rows
        else:
            # Absolute last resort: uniform random from whole table
            bucket = [entry for v in table.values() for entry in v]

    entry = random.choice(bucket)

    if discount in entry:
        return entry[discount]

    # If discount somehow not in entry, return zeros
    return 0.0, 0.0


def get_context_probabilities(
    category: str,
    loyalty_tier: str,
    demand_index: float,
    promotion: int,
    seed: Optional[int] = None,
) -> Dict[int, float]:
    """
    Return a dict {discount: purchase_probability} for all discount levels
    [0, 5, 10, 15, 20] drawn from a single real row matching this context.
    If seed is provided, the choice is deterministic for reproducibility.
    """
    table = get_table()
    demand_level = _demand_level_from_float(demand_index)

    if not loyalty_tier or loyalty_tier not in ("New", "Regular", "Loyal"):
        loyalty_tier = "New"

    fallback_keys = [
        (category, loyalty_tier, demand_level, promotion),
        (category, loyalty_tier, demand_level, 1 - promotion),
        (category, "Regular",    demand_level, promotion),
        (category, "Regular",    demand_level, 1 - promotion),
        (category, "New",        demand_level, promotion),
    ]

    bucket = None
    for key in fallback_keys:
        if key in table and table[key]:
            bucket = table[key]
            break

    if bucket is None:
        all_rows = [entry for k, v in table.items() if k[0] == category for entry in v]
        bucket = all_rows if all_rows else [entry for v in table.values() for entry in v]

    if seed is not None:
        rng = random.Random(seed)
        entry = rng.choice(bucket)
    else:
        entry = random.choice(bucket)

    return {disc: entry[disc][0] for disc in DISCOUNT_LEVELS if disc in entry}


def oracle_lookup(
    category: str,
    loyalty_tier: str,
    demand_index: float,
    promotion: int,
) -> Tuple[int, float]:
    """
    Return the oracle-optimal (best_discount, best_profit) for this context.
    Used in Phase 7 evaluation to compute regret.

    Returns (oracle_discount, oracle_profit).
    """
    table = get_table()
    demand_level = _demand_level_from_float(demand_index)

    if not loyalty_tier or loyalty_tier not in ("New", "Regular", "Loyal"):
        loyalty_tier = "New"

    key = (category, loyalty_tier, demand_level, int(promotion))
    bucket = table.get(key)

    if not bucket:
        # Fallback: any row for category
        bucket = [e for k, v in table.items() if k[0] == category for e in v]
    if not bucket:
        bucket = [e for v in table.values() for e in v]

    entry = random.choice(bucket)
    oracle = entry.get("oracle")
    if oracle:
        return oracle
    return 5, 0.0   # safe default


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("data_loader.py self-test")
    print("=" * 55)

    table = get_table()
    print(f"\nTotal buckets: {len(table)}")
    print("Sample buckets (key -> row count):")
    for key, rows in list(table.items())[:5]:
        print(f"  {key}: {len(rows)} rows")

    print("\nSample lookup - Electronics, Regular, DemandIndex=0.8 (Low), Promotion=1:")
    for disc in [0, 5, 10, 15, 20]:
        prob, profit = lookup("Electronics", "Regular", 0.8, 1, disc)
        print(f"  discount={disc}%  prob={prob:.4f}  profit=Rs {profit:.4f}")

    print("\nOracle lookup for same context:")
    best_disc, best_profit = oracle_lookup("Electronics", "Regular", 0.8, 1)
    print(f"  OracleBestDiscount={best_disc}%  OracleBestProfit=Rs {best_profit:.4f}")
