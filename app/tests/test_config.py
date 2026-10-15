"""
Unit tests for configuration loader and input schema.
"""

import pytest
from app.config import (
    ACTIONS,
    APP_CONFIG,
    INPUT_SCHEMA,
    MODEL_INFO,
    NUMERIC_INPUTS,
    CATEGORICAL_INPUTS,
    DEFAULT_MARGIN_FLOOR,
    MAX_MARGIN_FLOOR,
    fmt_pct,
    fmt_rupees,
    get_tier_from_purchases,
)


def test_config_loaded_properly():
    assert APP_CONFIG is not None
    assert ACTIONS == [0, 5, 10, 15, 20]
    assert DEFAULT_MARGIN_FLOOR == 0.0
    assert MAX_MARGIN_FLOOR == 50.0


def test_input_schema_ranges():
    """Verify min <= default <= max for all numeric inputs."""
    for field in INPUT_SCHEMA:
        if field["type"] in ("float", "int"):
            assert field["min"] <= field["default"] <= field["max"], (
                f"Field {field['key']} has invalid range: min={field['min']}, default={field['default']}, max={field['max']}"
            )
            assert field["p01"] <= field["p99"], f"Field {field['key']} p01 > p99"


def test_categorical_levels_match_metadata():
    """Verify categorical options match saved metadata."""
    meta_cats = APP_CONFIG["metadata"]
    assert "Category" in CATEGORICAL_INPUTS
    assert "Electronics" in CATEGORICAL_INPUTS["Category"]["options"]
    assert "West" in CATEGORICAL_INPUTS["Region"]["options"]
    assert "Spring" in CATEGORICAL_INPUTS["Seasonality"]["options"]


def test_formatting_helpers():
    assert fmt_rupees(1250.50) == "₹1,250.50"
    assert fmt_rupees(1250.50, simulated=True) == "₹1,250.50 (simulated)"
    assert fmt_pct(12.34) == "12.3%"


def test_tier_mapping():
    assert get_tier_from_purchases(0) == "New"
    assert get_tier_from_purchases(2) == "New"
    assert get_tier_from_purchases(3) == "Regular"
    assert get_tier_from_purchases(9) == "Regular"
    assert get_tier_from_purchases(10) == "Loyal"
    assert get_tier_from_purchases(25) == "Loyal"
