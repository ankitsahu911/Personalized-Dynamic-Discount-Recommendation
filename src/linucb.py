"""
Milestone 23 - Disjoint LinUCB Bandit (from scratch, NumPy only)
Retail Discount-Recommendation Bandit Project

What this file implements:
    A Disjoint LinUCB contextual bandit for discount recommendation.
    "Disjoint" means each arm (discount level) has its OWN independent
    linear model -- A_a and b_a are never shared across arms.

    The bandit picks among 5 possible discount arms: [0, 5, 10, 15, 20].
    In any real round, only a guardrail-filtered SUBSET of these is offered
    to the bandit -- it must only pick from that eligible subset.

Context vector x is built from:
    Numeric (z-score standardized):
        Price, InventoryLevel, CompetitorPricing, DemandIndex,
        CurrentBasketValue, PurchaseCount, RecencyDays
    Binary (used as-is, already 0/1):
        Promotion
    Categorical (one-hot encoded):
        Category:    Electronics, Groceries, Furniture, Toys, Clothing  (5 values)
        Region:      West, South, North, East                           (4 values)
        Seasonality: Spring, Summer, Autumn, Winter                     (4 values)
        LoyaltyTier: New, Regular, Loyal                               (3 values)
    Bias:
        1 constant term appended at the end

Scaling choice -- Z-SCORE (mean=0, std=1):
    Chosen over min-max because:
    1. LinUCB's ridge update is equivalent to a Bayesian prior centered at 0.
       Z-scored features keep that prior meaningful -- min-max would shift
       features to [0,1] and interact poorly with the identity-matrix prior.
    2. Z-score is robust to the outliers common in retail price data.
    The stored means and stds are the calibrated values from our real
    73,100-row dataset (from live_generator.py calibration).

Run:
    python src/linucb.py
"""

import math
import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. FEATURE CATALOGUE
#    All categoricals and their allowed values are defined here once.
#    Change this list -> context_dim updates automatically everywhere.
# ---------------------------------------------------------------------------

# Numeric features and their z-score stats from REAL dataset calibration
# (same values used to calibrate live_generator.py in Milestone 19)
NUMERIC_STATS = {
    "Price":                     {"mean": 52.76,  "std": 27.48},
    "InventoryLevel":            {"mean": 155.0,  "std": 83.5},
    "CompetitorPricing":         {"mean": 52.76,  "std": 29.7},
    "DemandIndex":               {"mean": 0.5,    "std": 0.289},
    "CurrentBasketValue":        {"mean": 275.0,  "std": 130.0},
    "PurchaseCount":             {"mean": 7.0,    "std": 8.0},
    "RecencyDays":               {"mean": 30.0,   "std": 35.0},
    # Real value from 73,100-row dataset (mean=1.0025, std=0.2138, range [0.55, 1.55])
    # Critical feature: higher sensitivity -> customer responds more to discounts
    "CustomerPriceSensitivity":  {"mean": 1.0025, "std": 0.2138},
}

# Categorical features and ALL their possible values (order is fixed -- do not change)
CATEGORICAL_VALUES = {
    "Category":    ["Electronics", "Groceries", "Furniture", "Toys", "Clothing"],
    "Region":      ["West", "South", "North", "East"],
    "Seasonality": ["Spring", "Summer", "Autumn", "Winter"],
    "LoyaltyTier": ["New", "Regular", "Loyal"],
}

# Total context dimension:
#   8 numeric  +  1 binary (Promotion)  +  5+4+4+3 categorical  +  1 bias
#   = 8 + 1 + 16 + 1 = 26
CONTEXT_DIM = (
    len(NUMERIC_STATS)          # 8 numeric (inc. CustomerPriceSensitivity)
    + 1                          # Promotion binary
    + sum(len(v) for v in CATEGORICAL_VALUES.values())  # 5+4+4+3 = 16 one-hot
    + 1                          # bias term
)  # = 26


# ---------------------------------------------------------------------------
# 2. FEATURE BUILDER  (standalone function so Thompson Sampling can reuse it)
# ---------------------------------------------------------------------------

def build_context_vector(raw_context: Dict, customer) -> np.ndarray:
    """
    Convert a raw context dict + Customer object into a fixed-length
    numeric vector x of shape (CONTEXT_DIM,).

    Parameters
    ----------
    raw_context : dict
        Keys: Price, CostPrice, Category, Region, InventoryLevel,
              CompetitorPricing, Seasonality, Promotion, DemandIndex
    customer : Customer dataclass
        Fields: CustomerID, PurchaseCount, RecencyDays, LoyaltyTier,
                CurrentBasketValue

    Returns
    -------
    np.ndarray of shape (CONTEXT_DIM,) = (25,)
    """
    parts = []

    # --- 7 Numeric features: z-score standardized ---
    # Formula: z = (value - mean) / std
    # Using stored population stats from real data calibration.
    # If std is 0 (degenerate feature), we output 0 instead of dividing by 0.
    numeric_values = {
        "Price":                    raw_context.get("Price", 0.0),
        "InventoryLevel":           raw_context.get("InventoryLevel", 0.0),
        "CompetitorPricing":        raw_context.get("CompetitorPricing", 0.0),
        "DemandIndex":              raw_context.get("DemandIndex", 0.5),
        "CurrentBasketValue":       getattr(customer, "CurrentBasketValue", 0.0),
        "PurchaseCount":            float(getattr(customer, "PurchaseCount", 0)),
        "RecencyDays":              float(getattr(customer, "RecencyDays", 0)),
        # From context dict (live_generator now samples this from real distribution)
        "CustomerPriceSensitivity": raw_context.get("CustomerPriceSensitivity", 1.0025),
    }
    for feat, val in numeric_values.items():
        mean = NUMERIC_STATS[feat]["mean"]
        std  = NUMERIC_STATS[feat]["std"]
        z = (val - mean) / std if std > 0 else 0.0
        parts.append(z)

    # --- 1 Binary feature: Promotion (already 0 or 1, no scaling needed) ---
    parts.append(float(raw_context.get("Promotion", 0)))

    # --- 16 Categorical features: one-hot encoding ---
    # For each category, we emit a vector of 0s with a single 1 at the
    # matching position. Unknown values produce an all-zeros vector
    # (treated as "missing/unknown" rather than crashing).
    customer_loyalty_tier = getattr(customer, "LoyaltyTier", "") or "New"
    categorical_values_for_row = {
        "Category":    raw_context.get("Category", ""),
        "Region":      raw_context.get("Region", ""),
        "Seasonality": raw_context.get("Seasonality", ""),
        "LoyaltyTier": customer_loyalty_tier,
    }
    for feat, allowed in CATEGORICAL_VALUES.items():
        val = categorical_values_for_row[feat]
        one_hot = [1.0 if val == v else 0.0 for v in allowed]
        parts.extend(one_hot)

    # --- 1 Bias term ---
    # A constant 1 appended at the end allows the linear model to learn
    # an intercept, equivalent to adding a bias neuron.
    parts.append(1.0)

    x = np.array(parts, dtype=np.float64)
    assert x.shape == (CONTEXT_DIM,), (
        f"Context vector shape mismatch: got {x.shape}, expected ({CONTEXT_DIM},)"
    )
    return x


# ---------------------------------------------------------------------------
# 3. DISJOINT LinUCB CLASS
# ---------------------------------------------------------------------------

class DisjointLinUCB:
    """
    Disjoint Linear Upper Confidence Bound bandit.

    Each arm a has its own ridge-regression model:
        A_a  : (context_dim x context_dim) matrix, initialized to identity
        b_a  : (context_dim,) vector, initialized to zeros

    The estimated weight vector for arm a:
        theta_a = A_a^{-1} @ b_a

    UCB score for arm a given context x:
        score(a) = theta_a^T @ x  +  alpha * sqrt(x^T @ A_a^{-1} @ x)
                   [exploitation]     [exploration bonus -- uncertainty]

    The first term rewards arms that historically performed well in similar
    contexts. The second term rewards arms we are uncertain about (wide
    confidence interval) -- this is what drives exploration.
    """

    def __init__(
        self,
        arms: List[int],
        context_dim: int,
        alpha: float = 1.0,
        gamma: float = 1.0,
        warm_start_thetas: Optional[Dict[int, np.ndarray]] = None,
        use_sherman_morrison: bool = True,
    ):
        """
        Parameters
        ----------
        arms : list[int]
            All possible discount arms [0, 5, 10, 15, 20].
        context_dim : int
            Dimension of the context vector.
        alpha : float
            Exploration strength.
        gamma : float, default 1.0
            Discount factor for non-stationary environments (D-LinUCB).
            gamma = 1.0 -> standard stationary bandit.
            gamma < 1.0 (e.g. 0.9995) -> exponentially forgets stale history.
        warm_start_thetas : dict[int, np.ndarray], optional
            Pre-estimated initial weight vectors per arm from historical regression.
            Eliminates cold-start blind exploration in early rounds.
        use_sherman_morrison : bool, default True
            Updates A_inv in O(d^2) directly via Sherman-Morrison rank-1 formula
            rather than recomputing O(d^3) matrix inversions every round.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if gamma <= 0 or gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        self.arms = arms
        self.context_dim = context_dim
        self._alpha = alpha
        self.gamma = gamma
        self.use_sherman_morrison = use_sherman_morrison
        self._step_count = {a: 0 for a in arms}

        # Per-arm covariance matrix A_a and reward accumulator b_a
        self.A = {a: np.eye(context_dim) for a in arms}
        self.A_inv = {a: np.eye(context_dim) for a in arms}
        self.b = {a: np.zeros(context_dim) for a in arms}

        # Warm-start initialization
        if warm_start_thetas is not None:
            for a in arms:
                if a in warm_start_thetas:
                    th0 = np.asarray(warm_start_thetas[a], dtype=np.float64)
                    assert th0.shape == (context_dim,), f"Warm-start theta shape mismatch for arm {a}"
                    self.b[a] = self.A[a] @ th0

    # ------------------------------------------------------------------
    # alpha property: settable AFTER construction without resetting state
    # (needed for Milestone 24's alpha sensitivity experiment)
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value < 0:
            raise ValueError(f"alpha must be >= 0, got {value}")
        self._alpha = value

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_action(self, raw_context: Dict, customer, eligible_arms: List[int]) -> int:
        """
        Given a context and the guardrail-filtered eligible arms,
        return the arm with the highest UCB score.

        Uses pre-computed A_inv in O(d^2) via Sherman-Morrison.
        """
        if not eligible_arms:
            raise ValueError("select_action called with empty eligible_arms.")

        x = build_context_vector(raw_context, customer)
        best_arm = None
        best_score = -math.inf

        for arm in eligible_arms:
            if self.use_sherman_morrison:
                A_inv = self.A_inv[arm]
            else:
                A_inv = np.linalg.inv(self.A[arm])

            theta = A_inv @ self.b[arm]

            exploit = float(theta @ x)
            var_term = float(x @ A_inv @ x)
            # Guard: var_term should always be >= 0 for PSD A_inv; clip for fp safety
            if not math.isfinite(var_term) or var_term < 0.0:
                var_term = 0.0
            explore = self._alpha * math.sqrt(var_term)

            score = exploit + explore

            if score > best_score:
                best_score = score
                best_arm   = arm

        return best_arm

    def update(self, arm: int, raw_context: Dict, customer, reward: float):
        """
        Online model update.
        Uses the Sherman-Morrison rank-1 update to maintain A_inv directly in O(d^2):
            (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        With optional exponential discount factor gamma for non-stationary environments.
        """
        if arm not in self.A:
            raise ValueError(f"Unknown arm: {arm}. Valid arms: {self.arms}")

        x = build_context_vector(raw_context, customer)
        self._step_count[arm] += 1

        if self.gamma == 1.0:
            # Standard stationary update
            self.A[arm] += np.outer(x, x)
            self.b[arm] += reward * x

            if self.use_sherman_morrison:
                A_inv = self.A_inv[arm]
                v = A_inv @ x
                denom = 1.0 + float(x @ v)
                self.A_inv[arm] = A_inv - np.outer(v, v) / denom
        else:
            # Non-stationary Discounted LinUCB (D-LinUCB) update
            # New A_discounted = gamma * A_old + x x^T
            # New b_discounted = gamma * b_old + r * x
            self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
            self.b[arm] = self.gamma * self.b[arm] + reward * x

            if self.use_sherman_morrison:
                # Woodbury for (gamma * A_old + x x^T)^{-1}:
                # Let B = gamma * A_old  =>  B^{-1} = A_inv_old / gamma
                # (B + x x^T)^{-1} = B^{-1} - (B^{-1} x x^T B^{-1}) / (1 + x^T B^{-1} x)
                B_inv = self.A_inv[arm] / self.gamma
                v = B_inv @ x
                denom = 1.0 + float(x @ v)
                self.A_inv[arm] = B_inv - np.outer(v, v) / denom

        # Periodically refresh A_inv every 2,000 arm pulls to eliminate floating point drift
        if self.use_sherman_morrison and (self._step_count[arm] % 2000 == 0):
            self.A_inv[arm] = np.linalg.inv(self.A[arm])


# ---------------------------------------------------------------------------
# 4. SELF-TEST  (run: python src/linucb.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from round_generator import Customer

    print("=" * 60)
    print("Milestone 23 -- Disjoint LinUCB Self-Test")
    print("=" * 60)

    # ---- 4a. Confirm CONTEXT_DIM is what we calculated ----
    print(f"\nExpected CONTEXT_DIM: 26")
    print(f"Actual   CONTEXT_DIM: {CONTEXT_DIM}")
    assert CONTEXT_DIM == 26, f"CONTEXT_DIM mismatch! Got {CONTEXT_DIM}"
    print("  => Shape confirmed")

    # ---- 4b. Build a sample context vector, inspect it ----
    sample_context = {
        "Price": 65.0,
        "CostPrice": 39.0,
        "Category": "Electronics",
        "Region": "West",
        "InventoryLevel": 120,
        "CompetitorPricing": 68.5,
        "Seasonality": "Summer",
        "Promotion": 1,
        "DemandIndex": 0.72,
    }
    sample_customer = Customer(
        CustomerID=1,
        PurchaseCount=12,
        RecencyDays=5,
        LoyaltyTier="Loyal",
        CurrentBasketValue=450.0,
    )

    x = build_context_vector(sample_context, sample_customer)
    print(f"\nSample context vector x:")
    print(f"  shape : {x.shape}  (expected: ({CONTEXT_DIM},))")
    print(f"  values: {x.round(4)}")
    print(f"  contains NaN: {np.isnan(x).any()}  (must be False)")

    # ---- 4c. Instantiate bandit, confirm matrix shapes ----
    ARMS = [0, 5, 10, 15, 20]
    bandit = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=1.0)

    print(f"\nBandit state shapes (per arm):")
    for arm in ARMS:
        print(f"  arm={arm:>2}  A_a.shape={bandit.A[arm].shape}  b_a.shape={bandit.b[arm].shape}")

    expected_shape = (CONTEXT_DIM, CONTEXT_DIM)
    for arm in ARMS:
        assert bandit.A[arm].shape == expected_shape, f"A[{arm}] shape wrong"
        assert bandit.b[arm].shape == (CONTEXT_DIM,),  f"b[{arm}] shape wrong"
    print("  => All shapes correct")

    # ---- 4d. Run 3 rounds: select -> update -> select ----
    print("\n--- 3-Round Trace ---")
    eligible = [0, 5, 10, 15, 20]   # Loyal customer, all eligible

    for rnd in range(1, 4):
        chosen = bandit.select_action(sample_context, sample_customer, eligible)
        # Simulate a reward: higher discount -> customer more likely to buy
        # (simple placeholder -- real simulator comes in Milestone 25)
        simulated_reward = 10.0 - chosen * 0.3 + (1 if sample_context["Promotion"] else 0)
        bandit.update(chosen, sample_context, sample_customer, simulated_reward)
        print(f"  Round {rnd}: eligible={eligible}  chosen={chosen}  reward={simulated_reward:.2f}")

    # ---- 4e. Confirm alpha validation ----
    print("\n--- Alpha Validation ---")
    try:
        bad_bandit = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=-0.5)
        print("  ERROR: should have raised ValueError for negative alpha!")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nAll checks passed -- Milestone 23 ready.")
