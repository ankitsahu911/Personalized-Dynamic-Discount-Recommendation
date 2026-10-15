from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_SCRIPT = str((Path(__file__).resolve().parent.parent / "streamlit_app.py"))


def test_app_smoke_default_execution():
    """App runs on default inputs without any exceptions."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"App raised exceptions: {at.exception}"
    assert len(at.title) > 0


def test_app_smoke_widget_interaction():
    """Modifying widgets reruns cleanly with zero exceptions."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Modify category
    if at.sidebar.selectbox(key="input_Category"):
        at.sidebar.selectbox(key="input_Category").select("Furniture")
        at.run()
        assert len(at.exception) == 0

    # Modify price
    if at.sidebar.number_input(key="input_Price"):
        at.sidebar.number_input(key="input_Price").set_value(90.0)
        at.run()
        assert len(at.exception) == 0


def test_app_smoke_page_navigation():
    """Cycles through all 6 pages with zero exceptions."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    pages = [
        "🏠 Store Overview & Catalog",
        "🎯 Live Pricing Simulator",
        "🔮 Market Sensitivity & What-If Lab",
        "🛡️ Margin Guardrails Studio",
        "📊 Policy Performance & ROI Audits",
        "ℹ️ Model Architecture & Transparency",
    ]
    for page in pages:
        at.radio(key="nav_page").set_value(page)
        at.run()
        assert len(at.exception) == 0
        assert len(at.title) > 0


def test_app_smoke_product_catalog_click():
    """Clicking 'Optimize Discount' on a product card transitions to the simulator."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Find the button for product 1 and click it
    btn = at.button(key="btn_opt_prod_1")
    if btn:
        btn.click()
        at.run()
        assert len(at.exception) == 0
        assert at.radio(key="nav_page").value == "🎯 Live Pricing Simulator"
        assert at.sidebar.number_input(key="input_Price").value == 85.0
