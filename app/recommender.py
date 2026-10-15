"""
Pure Recommendation Logic Engine
Retail Dynamic Discount Recommendation Project
NO STREAMLIT IMPORTS ALLOWED IN THIS MODULE.
"""

import math
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import (
    ACTIONS,
    APP_CONFIG,
    ASSUMED_MARGIN_RATE,
    DEFAULT_LOYAL_CUTOFF,
    DEFAULT_MARGIN_FLOOR,
    DEFAULT_REGULAR_CUTOFF,
    TIER_DISCOUNTS,
    fmt_rupees,
    get_tier_from_purchases,
)

# Reference root for loading model
APP_DIR = Path(__file__).resolve().parent
MODEL_PREFIX = APP_DIR / "assets" / "models" / "linucb_v1"


@dataclass
class Rules:
    """Configurable business guardrails."""
    regular_cutoff: int = DEFAULT_REGULAR_CUTOFF
    loyal_cutoff: int = DEFAULT_LOYAL_CUTOFF
    margin_floor: float = DEFAULT_MARGIN_FLOOR
    cost_price_override: Optional[float] = None

    def get_tier_boundaries(self) -> Dict[str, Tuple[int, Optional[int]]]:
        return {
            "New": (0, self.regular_cutoff),
            "Regular": (self.regular_cutoff, self.loyal_cutoff),
            "Loyal": (self.loyal_cutoff, None),
        }


@dataclass
class ArmDetail:
    """Detailed diagnostic per discount arm."""
    discount: int
    status: str  # 'recommended', 'eligible', 'blocked_by_tier', 'blocked_by_margin', 'blocked_by_both'
    reason: str
    point_estimate: Optional[float]
    uncertainty_bonus: Optional[float]
    score: Optional[float]
    pull_counts: int
    extrapolated: bool
    unit_margin_after_discount: float
    selling_price: float


@dataclass
class Recommendation:
    """Complete recommendation result object."""
    chosen_discount: Optional[int]
    eligible_discounts: List[int]
    discounted_price: float
    discount_amount_rupees: float
    unit_margin_after_discount: float
    tier_for_rules: str
    tier_for_model: str
    arms_table: List[ArmDetail]
    why_pick_text: str
    warnings: List[str] = field(default_factory=list)
    raw_context: Dict[str, Any] = field(default_factory=dict)


# Customer snapshot for feature construction
class CustomerSnapshot:
    def __init__(self, CustomerID: str, PurchaseCount: int, RecencyDays: int, LoyaltyTier: str, CurrentBasketValue: float):
        self.CustomerID = CustomerID
        self.PurchaseCount = PurchaseCount
        self.RecencyDays = RecencyDays
        self.LoyaltyTier = LoyaltyTier
        self.CurrentBasketValue = CurrentBasketValue


def make_context(inputs: Dict[str, Any], rules: Optional[Rules] = None) -> Tuple[Dict[str, Any], CustomerSnapshot]:
    """
    Converts raw form inputs into the exact dictionary and Customer object
    expected by the feature builder and simulator.
    """
    price = float(inputs.get("Price", 50.0))
    if rules and rules.cost_price_override is not None and rules.cost_price_override > 0:
        cost_price = float(rules.cost_price_override)
    else:
        cost_price = round(price * (1.0 - ASSUMED_MARGIN_RATE), 2)

    demand_level = str(inputs.get("DemandLevel", "Medium"))
    if demand_level == "Low":
        demand_idx = 0.20
        d_code = 0
    elif demand_level == "High":
        demand_idx = 0.80
        d_code = 2
    else:
        demand_idx = 0.50
        d_code = 1

    promo_val = inputs.get("Promotion", False)
    promo_int = 1 if (promo_val is True or promo_val == 1 or promo_val == "1") else 0

    purchase_count = int(inputs.get("PurchaseCount", 3))
    recency = int(inputs.get("RecencyDays", 14))
    basket = float(inputs.get("CurrentBasketValue", 150.0))
    cps = float(inputs.get("CustomerPriceSensitivity", 1.00))

    # Decision D1: Model tier is ALWAYS derived from default boundaries (3 and 10)
    tier_for_model = get_tier_from_purchases(purchase_count, DEFAULT_REGULAR_CUTOFF, DEFAULT_LOYAL_CUTOFF)

    cust = CustomerSnapshot(
        CustomerID="APP_USER",
        PurchaseCount=purchase_count,
        RecencyDays=recency,
        LoyaltyTier=tier_for_model,
        CurrentBasketValue=basket,
    )

    ctx = {
        "Price": price,
        "CostPrice": cost_price,
        "Category": str(inputs.get("Category", "Electronics")),
        "Region": str(inputs.get("Region", "West")),
        "Seasonality": str(inputs.get("Seasonality", "Spring")),
        "InventoryLevel": int(inputs.get("InventoryLevel", 250)),
        "CompetitorPricing": float(inputs.get("CompetitorPricing", 50.0)),
        "Promotion": promo_int,
        "DemandIndex": demand_idx,
        "demand_index": d_code,
        "CustomerPriceSensitivity": cps,
        "LoyaltyTier": tier_for_model,
    }
    return ctx, cust


class StandaloneLinUCB:
    """
    Lightweight, immutable, self-contained LinUCB inference loader.
    Never updates weights; thread-safe for concurrent Streamlit sessions.
    """
    def __init__(self, arms: List[int], context_dim: int, alpha: float, A: np.ndarray, b: np.ndarray, pull_counts: np.ndarray, preprocessing: dict):
        self.arms = arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = A  # shape (n_arms, d, d)
        self.b = b  # shape (n_arms, d)
        self.pull_counts = pull_counts  # shape (n_arms,)
        self.preprocessing = preprocessing

        # Precompute thetas and A_inv
        self.A_inv = np.zeros_like(A)
        self.theta = np.zeros_like(b)
        for i in range(len(arms)):
            self.A_inv[i] = np.linalg.inv(A[i])
            self.theta[i] = self.A_inv[i] @ b[i]

    def build_feature_vector(self, raw_context: Dict[str, Any], customer: CustomerSnapshot) -> np.ndarray:
        """
        Builds the 26-dimensional numeric feature vector x using
        the exact scaler stats and one-hot levels stored in metadata.
        """
        parts = []
        num_stats = self.preprocessing["numeric_stats"]

        numeric_order = [
            "Price", "InventoryLevel", "CompetitorPricing", "DemandIndex",
            "CurrentBasketValue", "PurchaseCount", "RecencyDays", "CustomerPriceSensitivity"
        ]

        vals = {
            "Price": float(raw_context.get("Price", 0.0)),
            "InventoryLevel": float(raw_context.get("InventoryLevel", 0.0)),
            "CompetitorPricing": float(raw_context.get("CompetitorPricing", 0.0)),
            "DemandIndex": float(raw_context.get("DemandIndex", 0.5)),
            "CurrentBasketValue": float(getattr(customer, "CurrentBasketValue", 0.0)),
            "PurchaseCount": float(getattr(customer, "PurchaseCount", 0.0)),
            "RecencyDays": float(getattr(customer, "RecencyDays", 0.0)),
            "CustomerPriceSensitivity": float(raw_context.get("CustomerPriceSensitivity", 1.0025)),
        }

        for feat in numeric_order:
            mean = num_stats[feat]["mean"]
            std = num_stats[feat]["std"]
            z = (vals[feat] - mean) / std if std > 0 else 0.0
            parts.append(z)

        # Binary: Promotion
        parts.append(float(raw_context.get("Promotion", 0)))

        # One-hot categoricals
        cat_values = self.preprocessing["categorical_values"]
        row_cats = {
            "Category": raw_context.get("Category", ""),
            "Region": raw_context.get("Region", ""),
            "Seasonality": raw_context.get("Seasonality", ""),
            "LoyaltyTier": getattr(customer, "LoyaltyTier", "New") or "New",
        }

        for cat_feat in ["Category", "Region", "Seasonality", "LoyaltyTier"]:
            allowed = cat_values[cat_feat]
            curr_val = row_cats[cat_feat]
            parts.extend([1.0 if curr_val == v else 0.0 for v in allowed])

        # Bias term
        parts.append(1.0)

        x = np.array(parts, dtype=np.float64)
        assert x.shape == (self.context_dim,), f"Shape mismatch: {x.shape} vs ({self.context_dim},)"
        return x

    def score_breakdown(
        self,
        raw_context: Dict[str, Any],
        customer: CustomerSnapshot,
        eligible_arms: List[int],
        alpha: Optional[float] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Returns {arm: {'estimate': float, 'bonus': float, 'score': float}}
        Does not mutate the model.
        """
        eff_alpha = self.alpha if alpha is None else float(alpha)
        x = self.build_feature_vector(raw_context, customer)
        breakdown = {}

        for arm in eligible_arms:
            i = self.arms.index(arm)
            A_inv_i = self.A_inv[i]
            theta_i = self.theta[i]

            est = float(theta_i @ x)
            var_term = float(x @ A_inv_i @ x)
            # Guard against fp overflow or negative-definite drift in A_inv
            if not math.isfinite(var_term) or var_term < 0.0:
                var_term = 0.0
            bonus = eff_alpha * math.sqrt(var_term)
            score = est + bonus

            breakdown[arm] = {
                "estimate": est,
                "bonus": bonus,
                "score": score,
            }
        return breakdown

    def select_action(
        self,
        raw_context: Dict[str, Any],
        customer: CustomerSnapshot,
        eligible_arms: List[int],
        alpha: Optional[float] = None
    ) -> int:
        """Picks the eligible arm maximizing score; breaks ties deterministically."""
        scores = self.score_breakdown(raw_context, customer, eligible_arms, alpha=alpha)
        best_arm = eligible_arms[0]
        best_score = -math.inf
        for a in eligible_arms:
            if scores[a]["score"] > best_score:
                best_score = scores[a]["score"]
                best_arm = a
        return best_arm


def load_model(path_prefix: Optional[Path] = None) -> StandaloneLinUCB:
    """Load model from saved npz and metadata files."""
    prefix = str(path_prefix or MODEL_PREFIX)
    npz_path = f"{prefix}.npz"
    meta_path = f"{prefix}_meta.json"

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing model files at '{prefix}.npz' or '{prefix}_meta.json'. "
            f"Run scripts/sync_app_assets.py to copy assets into app/assets/."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(npz_path)
    arms = [int(a) for a in data["arms"]]
    A = data["A"]
    b = data["b"]
    pull_counts = data["pull_counts"] if "pull_counts" in data else np.zeros(len(arms), dtype=int)

    return StandaloneLinUCB(
        arms=arms,
        context_dim=int(meta["context_dim"]),
        alpha=float(meta["alpha"]),
        A=A,
        b=b,
        pull_counts=pull_counts,
        preprocessing=meta["preprocessing"],
    )


def evaluate_guardrails(
    price: float,
    cost_price: float,
    purchases: int,
    rules: Rules
) -> Tuple[List[int], str, str, Dict[int, Dict[str, Any]]]:
    """
    Evaluates tier and margin guardrails across all 5 possible discount arms.
    Returns: (eligible_arms, tier_for_rules, tier_for_model, per_arm_status)
    """
    tier_for_rules = get_tier_from_purchases(purchases, rules.regular_cutoff, rules.loyal_cutoff)
    tier_for_model = get_tier_from_purchases(purchases, DEFAULT_REGULAR_CUTOFF, DEFAULT_LOYAL_CUTOFF)

    tier_allowed = TIER_DISCOUNTS[tier_for_rules]
    default_tier_allowed = TIER_DISCOUNTS[tier_for_model]

    eligible_arms = []
    arm_status = {}

    for d in ACTIONS:
        selling_price = round(price * (1.0 - d / 100.0), 2)
        unit_margin = round(selling_price - cost_price, 2)

        passes_tier = d in tier_allowed
        passes_margin = unit_margin >= rules.margin_floor

        # Extrapolation check (Decision D3b): allowed under current custom rules but not default rules
        is_extrapolated = passes_tier and (d not in default_tier_allowed)

        if passes_tier and passes_margin:
            status = "eligible"
            reason = f"Complies with {tier_for_rules} tier menu and exceeds minimum unit margin floor."
            eligible_arms.append(d)
        elif not passes_tier and not passes_margin:
            status = "blocked_by_both"
            reason = f"Blocked by {tier_for_rules} tier rule AND post-discount margin ({fmt_rupees(unit_margin)}) is below floor ({fmt_rupees(rules.margin_floor)})."
        elif not passes_tier:
            status = "blocked_by_tier"
            reason = f"Blocked by loyalty policy: {tier_for_rules} tier allows only {tier_allowed}% discounts."
        else:
            status = "blocked_by_margin"
            reason = f"Blocked by margin floor: post-discount margin ({fmt_rupees(unit_margin)}) is below floor ({fmt_rupees(rules.margin_floor)})."

        arm_status[d] = {
            "status": status,
            "reason": reason,
            "passes_tier": passes_tier,
            "passes_margin": passes_margin,
            "extrapolated": is_extrapolated,
            "unit_margin": unit_margin,
            "selling_price": selling_price,
        }

    return eligible_arms, tier_for_rules, tier_for_model, arm_status


def recommend(
    inputs: Dict[str, Any],
    rules: Optional[Rules] = None,
    model: Optional[StandaloneLinUCB] = None,
    alpha_override: Optional[float] = None,
) -> Recommendation:
    """
    Main recommendation pipeline (Milestones 42 & 43).
    Pure function: no UI side-effects, no model mutation.
    """
    if rules is None:
        rules = Rules()
    if model is None:
        model = load_model()

    ctx, cust = make_context(inputs, rules=rules)
    price = ctx["Price"]
    cost_price = ctx["CostPrice"]
    purchases = cust.PurchaseCount

    # Evaluate guardrails
    eligible_arms, tier_for_rules, tier_for_model, arm_guard_status = evaluate_guardrails(
        price, cost_price, purchases, rules
    )

    warnings = []
    # Decision D3a: Soft warnings for inputs outside [p01, p99]
    num_inputs = APP_CONFIG.get("numeric_inputs", {})
    for k, v in inputs.items():
        if k in num_inputs and isinstance(v, (int, float)):
            p01 = num_inputs[k]["p01"]
            p99 = num_inputs[k]["p99"]
            if v < p01 or v > p99:
                warnings.append(
                    f"{num_inputs[k]['label']} ({v}) is outside standard training bounds "
                    f"[{p01}, {p99}] — model estimate may have wider variance."
                )

    # Empty eligible set handling (Decision D2)
    if not eligible_arms:
        arms_details = []
        for d in ACTIONS:
            info = arm_guard_status[d]
            i_idx = model.arms.index(d)
            pulls = int(model.pull_counts[i_idx]) if i_idx < len(model.pull_counts) else 0
            arms_details.append(ArmDetail(
                discount=d,
                status=info["status"],
                reason=info["reason"],
                point_estimate=None,
                uncertainty_bonus=None,
                score=None,
                pull_counts=pulls,
                extrapolated=info["extrapolated"],
                unit_margin_after_discount=info["unit_margin"],
                selling_price=info["selling_price"],
            ))

        return Recommendation(
            chosen_discount=None,
            eligible_discounts=[],
            discounted_price=price,
            discount_amount_rupees=0.0,
            unit_margin_after_discount=round(price - cost_price, 2),
            tier_for_rules=tier_for_rules,
            tier_for_model=tier_for_model,
            arms_table=arms_details,
            why_pick_text="No discount can be offered: all discount options are blocked by the business guardrails (loyalty tier limits or margin floor requirements).",
            warnings=warnings,
            raw_context=ctx,
        )

    # Score breakdown across eligible arms
    scores = model.score_breakdown(ctx, cust, eligible_arms, alpha=alpha_override)
    chosen_discount = model.select_action(ctx, cust, eligible_arms, alpha=alpha_override)

    discounted_price = round(price * (1.0 - chosen_discount / 100.0), 2)
    discount_amount = round(price * (chosen_discount / 100.0), 2)
    unit_margin = round(discounted_price - cost_price, 2)

    arms_details = []
    for d in ACTIONS:
        info = arm_guard_status[d]
        i_idx = model.arms.index(d)
        pulls = int(model.pull_counts[i_idx]) if i_idx < len(model.pull_counts) else 0

        if d in eligible_arms:
            sc_info = scores[d]
            st = "recommended" if d == chosen_discount else "eligible"
            pt_est = round(sc_info["estimate"], 3)
            bon = round(sc_info["bonus"], 3)
            tot_sc = round(sc_info["score"], 3)
            rsn = "Selected: highest total acquisition score (point estimate + uncertainty bonus)." if d == chosen_discount else "Eligible but scored lower than the chosen discount."
        else:
            st = info["status"]
            pt_est = None
            bon = None
            tot_sc = None
            rsn = info["reason"]

        arms_details.append(ArmDetail(
            discount=d,
            status=st,
            reason=rsn,
            point_estimate=pt_est,
            uncertainty_bonus=bon,
            score=tot_sc,
            pull_counts=pulls,
            extrapolated=info["extrapolated"],
            unit_margin_after_discount=info["unit_margin"],
            selling_price=info["selling_price"],
        ))

    # Explanation narrative (Milestone 43 / Step 2d)
    best_score_data = scores[chosen_discount]
    other_eligible = [a for a in eligible_arms if a != chosen_discount]
    if other_eligible:
        runner_up = max(other_eligible, key=lambda a: scores[a]["score"])
        runner_score = scores[runner_up]["score"]
        score_gap = best_score_data["score"] - runner_score
        driver = (
            "point estimate (expected unit profitability)"
            if best_score_data["estimate"] >= scores[runner_up]["estimate"]
            else "exploration bonus (uncertainty credit)"
        )
        why_text = (
            f"The model recommended {chosen_discount}% with a total score of {best_score_data['score']:.2f}. "
            f"It beat runner-up {runner_up}% (score {runner_score:.2f}) by a margin of {score_gap:.2f}. "
            f"The pick is primarily driven by the {driver}."
        )
    else:
        why_text = f"Only {chosen_discount}% discount survived the margin floor and loyalty tier guardrails."

    return Recommendation(
        chosen_discount=chosen_discount,
        eligible_discounts=eligible_arms,
        discounted_price=discounted_price,
        discount_amount_rupees=discount_amount,
        unit_margin_after_discount=unit_margin,
        tier_for_rules=tier_for_rules,
        tier_for_model=tier_for_model,
        arms_table=arms_details,
        why_pick_text=why_text,
        warnings=warnings,
        raw_context=ctx,
    )


def recommend_scenarios(
    base_inputs: Dict[str, Any],
    whatif_deltas: Dict[str, Any],
    rules: Optional[Rules] = None,
    model: Optional[StandaloneLinUCB] = None,
) -> Dict[str, Any]:
    """
    Milestone 44: Compares baseline inputs against relative what-if modifications.
    whatif_deltas:
        - competitor_price_pct_change: float (e.g. -15.0 for 15% price cut)
        - inventory_multiplier: float (e.g. 0.50 for half inventory)
        - seasonality: str ('Same as base' or explicit season)
    """
    if rules is None:
        rules = Rules()
    if model is None:
        model = load_model()

    base_rec = recommend(base_inputs, rules=rules, model=model)

    # Compute relative what-if values
    whatif_inputs = dict(base_inputs)
    base_comp = float(base_inputs.get("CompetitorPricing", 50.0))
    pct_chg = float(whatif_deltas.get("competitor_price_pct_change", 0.0))
    whatif_comp = round(base_comp * (1.0 + pct_chg / 100.0), 2)
    whatif_inputs["CompetitorPricing"] = max(10.0, min(150.0, whatif_comp))

    base_inv = int(base_inputs.get("InventoryLevel", 250))
    inv_mult = float(whatif_deltas.get("inventory_multiplier", 1.0))
    whatif_inv = int(round(base_inv * inv_mult))
    whatif_inputs["InventoryLevel"] = max(10, min(600, whatif_inv))

    whatif_season = whatif_deltas.get("seasonality", "Same as base")
    if whatif_season and whatif_season != "Same as base":
        whatif_inputs["Seasonality"] = str(whatif_season)

    whatif_rec = recommend(whatif_inputs, rules=rules, model=model)

    # Compute delta table across arms eligible in both
    delta_table = []
    base_dict = {a.discount: a for a in base_rec.arms_table}
    whatif_dict = {a.discount: a for a in whatif_rec.arms_table}

    for d in ACTIONS:
        b_a = base_dict[d]
        w_a = whatif_dict[d]
        if b_a.score is not None and w_a.score is not None:
            delta_table.append({
                "discount": d,
                "base_score": b_a.score,
                "whatif_score": w_a.score,
                "score_change": round(w_a.score - b_a.score, 3),
                "base_estimate": b_a.point_estimate,
                "whatif_estimate": w_a.point_estimate,
                "estimate_change": round(w_a.point_estimate - b_a.point_estimate, 3),
            })

    return {
        "base_recommendation": base_rec,
        "whatif_recommendation": whatif_rec,
        "whatif_inputs": whatif_inputs,
        "delta_table": delta_table,
    }


def compute_response_curve(
    base_inputs: Dict[str, Any],
    variable: str,  # 'CompetitorPricing', 'InventoryLevel', 'Seasonality'
    rules: Optional[Rules] = None,
    model: Optional[StandaloneLinUCB] = None,
) -> Dict[str, Any]:
    """
    Computes score response trajectories across a parameter range (Milestone 44 Step 3).
    """
    if rules is None:
        rules = Rules()
    if model is None:
        model = load_model()

    base_ctx, base_cust = make_context(base_inputs, rules=rules)
    eligible_arms, _, _, _ = evaluate_guardrails(
        base_ctx["Price"], base_ctx["CostPrice"], base_cust.PurchaseCount, rules
    )

    if not eligible_arms:
        return {"error": "All arms blocked by guardrails."}

    curve_data = {"variable": variable, "points": [], "curves": {d: [] for d in eligible_arms}, "winners": []}

    if variable == "CompetitorPricing":
        base_val = float(base_inputs.get("CompetitorPricing", 50.0))
        pts = np.linspace(max(10.0, base_val * 0.5), min(150.0, base_val * 1.5), 35)
        for p in pts:
            temp_in = dict(base_inputs, CompetitorPricing=round(p, 2))
            c_ctx, c_cust = make_context(temp_in, rules=rules)
            scores = model.score_breakdown(c_ctx, c_cust, eligible_arms)
            winner = max(eligible_arms, key=lambda a: scores[a]["score"])
            curve_data["points"].append(round(p, 2))
            curve_data["winners"].append(winner)
            for d in eligible_arms:
                curve_data["curves"][d].append(round(scores[d]["estimate"], 3))

    elif variable == "InventoryLevel":
        base_val = int(base_inputs.get("InventoryLevel", 250))
        pts = np.linspace(max(10, int(base_val * 0.25)), min(600, int(base_val * 2.5)), 35)
        for p in pts:
            temp_in = dict(base_inputs, InventoryLevel=int(p))
            c_ctx, c_cust = make_context(temp_in, rules=rules)
            scores = model.score_breakdown(c_ctx, c_cust, eligible_arms)
            winner = max(eligible_arms, key=lambda a: scores[a]["score"])
            curve_data["points"].append(int(p))
            curve_data["winners"].append(winner)
            for d in eligible_arms:
                curve_data["curves"][d].append(round(scores[d]["estimate"], 3))

    elif variable == "Seasonality":
        seasons = ["Spring", "Summer", "Autumn", "Winter"]
        for s in seasons:
            temp_in = dict(base_inputs, Seasonality=s)
            c_ctx, c_cust = make_context(temp_in, rules=rules)
            scores = model.score_breakdown(c_ctx, c_cust, eligible_arms)
            winner = max(eligible_arms, key=lambda a: scores[a]["score"])
            curve_data["points"].append(s)
            curve_data["winners"].append(winner)
            for d in eligible_arms:
                curve_data["curves"][d].append(round(scores[d]["estimate"], 3))

    return curve_data
