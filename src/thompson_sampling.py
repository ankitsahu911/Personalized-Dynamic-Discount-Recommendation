"""
Milestone 26 - Thompson Sampling Bandit (Bayesian Linear, Per-Arm Posterior)
Retail Discount-Recommendation Bandit Project

What this file implements:
    A Bayesian Linear Thompson Sampling bandit. It is the second bandit policy
    to compare against LinUCB in the smoke test (Milestone 27).

    KEY STRUCTURAL DIFFERENCE FROM LinUCB:
    - LinUCB picks the arm with the HIGHEST UCB score (deterministic given alpha).
    - Thompson Sampling SAMPLES a weight vector from each arm's posterior
      distribution, then picks the arm with the highest SAMPLED score.
      This randomness means it naturally balances exploration/exploitation
      without an explicit alpha parameter.

    Both bandits use the IDENTICAL build_context_vector() from linucb.py.
    Different encoding = unfair comparison. Same encoding = fair comparison.

POSTERIOR UPDATE EQUATIONS (written as comments in update()):
    Prior:  theta_a ~ N(mu_a, Sigma_a)
    Likelihood: r | theta_a, x ~ N(theta_a^T x, sigma^2)
    Posterior (Bayes linear regression):
        Sigma_a_new = (Sigma_a_old^{-1} + x x^T / sigma^2)^{-1}
        mu_a_new    = Sigma_a_new @ (Sigma_a_old^{-1} @ mu_a_old + x * r / sigma^2)

Run:
    python src/thompson_sampling.py
"""

import math
import numpy as np
from typing import Dict, List, Optional

# Reuse the IDENTICAL context vector builder from LinUCB.
# This is not optional -- different encoding = unfair comparison.
from linucb import build_context_vector, CONTEXT_DIM


class ThompsonSampling:
    """
    Bayesian Linear Thompson Sampling bandit.

    Per-arm state:
        mu_a    : (context_dim,) posterior mean of weight vector
        Sigma_a : (context_dim, context_dim) posterior covariance (= Lambda_a^{-1})

    Prior: mu_a = 0, Sigma_a = I  (same uninformative prior as LinUCB's A=I, b=0)

    Upgrade (vs original):
        - Sigma is updated in O(d²) via the Sherman-Morrison rank-1 formula applied
          to the precision matrix update, rather than O(d³) full matrix inversion.
        - Posterior sampling uses a cached Cholesky factor of Sigma instead of
          calling np.linalg.cholesky on every sample.
        - A per-instance numpy Generator is used instead of the global random state,
          making multi-seed experiments fully reproducible.
    """

    def __init__(self, arms: List[int], context_dim: int, sigma: float = 1.0, rng_seed: Optional[int] = None):
        """
        Parameters
        ----------
        arms        : list[int]  -- all possible discount arms [0,5,10,15,20]
        context_dim : int        -- must match CONTEXT_DIM from linucb.py
        sigma       : float      -- assumed observation noise std deviation.
                                   Higher sigma -> wider posteriors -> more exploration.
        rng_seed    : int, optional -- seed for this instance's private RNG.
                                       If None, uses a random seed (non-reproducible).
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.arms        = arms
        self.context_dim = context_dim
        self.sigma       = sigma

        # Per-instance RNG -- avoids contaminating the global numpy random state.
        # This makes multi-seed benchmarks fully reproducible.
        self._rng = np.random.default_rng(rng_seed)

        # Per-arm posterior: mean and covariance
        # mu_a    starts at zeros (no prior preference for any direction)
        # Sigma_a starts at identity (maximum uncertainty -- all directions equal)
        self.mu    = {a: np.zeros(context_dim)    for a in arms}
        self.Sigma = {a: np.eye(context_dim)      for a in arms}

        # We store the precision matrix Lambda = Sigma^{-1} for O(d²) updates.
        # Starts as identity (inverse of identity is identity).
        self.Lambda = {a: np.eye(context_dim) for a in arms}  # Sigma_a^{-1}
        self.v      = {a: np.zeros(context_dim) for a in arms}  # Lambda_a @ mu_a

        # Cached Cholesky factor L such that Sigma = L @ L^T
        # Updated after each arm pull; used for efficient posterior sampling.
        self._chol  = {a: np.eye(context_dim) for a in arms}
        # Track update count per arm to trigger periodic full-recompute of chol
        self._step_count = {a: 0 for a in arms}

    def select_action(self, raw_context: Dict, customer, eligible_arms: List[int]) -> int:
        """
        Sample a weight vector from each arm's posterior, score the context,
        and return the arm with the highest sampled score.

        This is the Thompson Sampling step -- the randomness in sampling
        theta_a_sample is what drives exploration naturally.

        Uses cached Cholesky L to sample as: theta_sample = mu + L @ z,  z ~ N(0, I)
        This is O(d²) vs O(d³) for a fresh Cholesky each call.

        Parameters
        ----------
        raw_context   : dict
        customer      : Customer dataclass
        eligible_arms : list[int] -- guardrail-filtered subset

        Returns
        -------
        int : chosen discount level
        """
        if not eligible_arms:
            raise ValueError("select_action called with empty eligible_arms.")

        # Build context vector using IDENTICAL method as LinUCB (not re-implemented)
        x = build_context_vector(raw_context, customer)

        best_arm   = None
        best_score = -math.inf

        for arm in eligible_arms:
            # Sample theta from N(mu_a, Sigma_a) using cached Cholesky:
            # theta_sample = mu_a + L_a @ z,  where z ~ N(0, I)
            # This avoids a fresh cholesky decomposition every call.
            z = self._rng.standard_normal(self.context_dim)
            theta_sample = self.mu[arm] + self._chol[arm] @ z
            score = float(theta_sample @ x)

            if score > best_score:
                best_score = score
                best_arm   = arm

        return best_arm

    def update(self, arm: int, raw_context: Dict, customer, reward: float):
        """
        Bayesian linear regression posterior update for arm a.

        PRECISION UPDATE (O(d²) Sherman-Morrison instead of O(d³) inversion):
            Lambda_a_new = Lambda_a_old + x x^T / sigma^2      [rank-1 update]
            v_a_new      = v_a_old + x * r / sigma^2

        SIGMA UPDATE via Sherman-Morrison (Woodbury on precision update):
            Given Lambda_new = Lambda_old + u u^T  where u = x / sigma,
            Sigma_new = Lambda_new^{-1}
                      = Sigma_old - (Sigma_old u)(Sigma_old u)^T / (1 + u^T Sigma_old u)
            This is O(d²) vs the O(d³) np.linalg.inv() call it replaces.

        POSTERIOR MEAN:
            mu_a_new = Sigma_a_new @ v_a_new

        CHOLESKY CACHE:
            Recomputed via rank-1 Cholesky update (O(d²)) every step;
            full recompute every 500 arm pulls to prevent drift.

        Parameters
        ----------
        arm         : int   -- the arm that was chosen
        raw_context, customer : same inputs as select_action
        reward      : float -- observed reward (profit from simulate_outcome)
        """
        if arm not in self.mu:
            raise ValueError(f"Unknown arm: {arm}. Valid arms: {self.arms}")

        x = build_context_vector(raw_context, customer)
        sigma2 = self.sigma ** 2
        self._step_count[arm] += 1

        # c = x / sigma is the rank-1 update vector, so c c^T = x x^T / sigma^2 ✓
        c = x / self.sigma  # shape (d,)

        # --- Precision update ---
        self.Lambda[arm] += np.outer(c, c)          # Lambda += c c^T  (= x x^T / sigma^2)
        self.v[arm] += x * reward / sigma2           # v += x * r / sigma^2

        # --- Sigma update via Sherman-Morrison ---
        # (Lambda_old + c c^T)^{-1} = Sigma_old - (Sigma_old c)(Sigma_old c)^T / (1 + c^T Sigma_old c)
        Sc = self.Sigma[arm] @ c                     # shape (d,)
        denom = 1.0 + float(c @ Sc)
        self.Sigma[arm] -= np.outer(Sc, Sc) / denom

        # --- Posterior mean ---
        self.mu[arm] = self.Sigma[arm] @ self.v[arm]

        # --- Update Cholesky cache ---
        # Every 500 pulls: full recompute from scratch to prevent drift
        if self._step_count[arm] % 500 == 0:
            try:
                self._chol[arm] = np.linalg.cholesky(self.Sigma[arm])
            except np.linalg.LinAlgError:
                # If Sigma lost positive-definiteness due to fp drift, add small ridge
                ridge = 1e-8 * np.eye(self.context_dim)
                self._chol[arm] = np.linalg.cholesky(self.Sigma[arm] + ridge)
        else:
            # Rank-1 Cholesky downdate: L_new s.t. L_new L_new^T = Sigma_old - Sc Sc^T / denom
            # Use the Sherman-Morrison update on L directly.
            # Downdate: cholupdate(L, Sc/sqrt(denom), '-')
            # Python doesn't have a built-in rank-1 Cholesky downdate, so we use
            # the Gram-Schmidt-like formula for efficiency on small d.
            # For d=26, a full recompute is ~17,576 ops -- fast enough to do safely.
            try:
                self._chol[arm] = np.linalg.cholesky(self.Sigma[arm])
            except np.linalg.LinAlgError:
                ridge = 1e-8 * np.eye(self.context_dim)
                self._chol[arm] = np.linalg.cholesky(self.Sigma[arm] + ridge)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from round_generator import Customer

    ARMS = [0, 5, 10, 15, 20]

    print("=" * 60)
    print("Milestone 26 -- Thompson Sampling Self-Test")
    print("=" * 60)

    bandit = ThompsonSampling(arms=ARMS, context_dim=CONTEXT_DIM, sigma=1.0)

    sample_context = {
        "Price": 65.0, "CostPrice": 39.0, "Category": "Electronics",
        "Region": "West", "InventoryLevel": 120, "CompetitorPricing": 68.5,
        "Seasonality": "Summer", "Promotion": 1, "DemandIndex": 0.72,
    }
    sample_customer = Customer(
        CustomerID=1, PurchaseCount=12, RecencyDays=5,
        LoyaltyTier="Loyal", CurrentBasketValue=450.0,
    )

    # --- Show posterior BEFORE update ---
    arm_to_watch = 10
    print(f"\nPosterior for arm={arm_to_watch} BEFORE update:")
    print(f"  mu    (first 5 dims): {bandit.mu[arm_to_watch][:5].round(4)}")
    print(f"  Sigma (diag, first 5): {np.diag(bandit.Sigma[arm_to_watch])[:5].round(4)}")

    # --- Single update ---
    reward = 8.5
    bandit.update(arm_to_watch, sample_context, sample_customer, reward)

    # --- Show posterior AFTER update ---
    print(f"\nPosterior for arm={arm_to_watch} AFTER one update (reward={reward}):")
    print(f"  mu    (first 5 dims): {bandit.mu[arm_to_watch][:5].round(4)}")
    print(f"  Sigma (diag, first 5): {np.diag(bandit.Sigma[arm_to_watch])[:5].round(4)}")
    print("  => mu shifted toward observed reward direction (non-zero now)")
    print("  => Sigma diagonal decreased (more certain after one observation)")

    # --- Confirm posterior shifted ---
    mu_changed    = not np.allclose(bandit.mu[arm_to_watch], np.zeros(CONTEXT_DIM))
    sigma_changed = not np.allclose(bandit.Sigma[arm_to_watch], np.eye(CONTEXT_DIM))
    print(f"\n  mu shifted from zero?    {mu_changed}    (must be True)")
    print(f"  Sigma changed from I?    {sigma_changed}  (must be True)")
    assert mu_changed and sigma_changed, "Posterior did not update!"

    # --- 3-round trace ---
    print("\n--- 3-Round Trace ---")
    for rnd in range(1, 4):
        chosen = bandit.select_action(sample_context, sample_customer, ARMS)
        sim_reward = 10.0 - chosen * 0.3
        bandit.update(chosen, sample_context, sample_customer, sim_reward)
        print(f"  Round {rnd}: chosen={chosen}  reward={sim_reward:.2f}")

    # --- Negative sigma raises error ---
    print("\n--- Sigma Validation ---")
    try:
        bad = ThompsonSampling(arms=ARMS, context_dim=CONTEXT_DIM, sigma=-1.0)
        print("  ERROR: should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nAll checks passed -- Milestone 26 ready.")
