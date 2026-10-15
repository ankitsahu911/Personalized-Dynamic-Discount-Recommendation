"""
Milestone 22 — Rule-Based Baseline
------------------------------------
Policy: use the customer's current demand level to pick a discount target,
then snap to the closest eligible action (same tie-break as Milestone 21).

Why a lookup table instead of a linear formula?
    A linear formula (e.g. target = 20 - demand_index * 10) would work here
    because we only have 3 levels. But a lookup table is preferred because:
      • It makes the business intent explicit and easy to audit.
      • Adding a new demand level (e.g. "Very High") requires one dict entry,
        not a formula change.
      • No chance of off-by-one errors when levels are not evenly spaced.

Demand-index convention (matches round_generator.py):
    0 = Low demand  → target 20 % (stimulate demand with a large discount)
    1 = Medium      → target 10 % (moderate push)
    2 = High        → target  0 % (no discount needed; product sells itself)

Tie-break rule: LOWER value wins (imported from fixed_baseline, not re-implemented).

Usage:
    from rule_based_baseline import rule_based_baseline

    ctx_low  = {"demand_index": 0}
    ctx_high = {"demand_index": 2}

    rule_based_baseline(ctx_low,  [0, 5, 10, 15, 20])  # → 20
    rule_based_baseline(ctx_high, [0, 5, 10, 15, 20])  # → 0
    rule_based_baseline(ctx_low,  [0, 5])               # → 5  (guardrail wins!)
"""

import logging
from typing import List, Optional

# Reuse the snap helper from Milestone 21 — do NOT re-implement it.
from fixed_baseline import _snap_to_closest

logger = logging.getLogger(__name__)

# Lookup table: demand_index → preferred discount target
_DEMAND_TARGET: dict = {
    0: 20,   # Low demand  → big discount to stimulate sales
    1: 10,   # Medium      → moderate discount
    2: 0,    # High demand → no discount needed
}


def rule_based_baseline(context: dict, eligible_actions: List[int]) -> Optional[int]:
    """
    Return the best available discount according to a demand-driven rule.

    Parameters
    ----------
    context : dict
        Must contain key ``"demand_index"`` (int: 0=Low, 1=Medium, 2=High).
    eligible_actions : list[int]
        Discounts pre-filtered by the guardrails.

    Returns
    -------
    int or None
        The chosen discount, or None if eligible_actions is empty.
    """
    if not eligible_actions:
        logger.warning("rule_based_baseline received an empty eligible_actions list — returning None.")
        return None

    demand_index = context.get("demand_index")
    if demand_index not in _DEMAND_TARGET:
        raise ValueError(
            f"rule_based_baseline: unrecognised demand_index={demand_index!r}. "
            f"Expected one of {list(_DEMAND_TARGET)}."
        )

    target = _DEMAND_TARGET[demand_index]

    # Exact match — fast path
    if target in eligible_actions:
        return target

    # Snap to nearest; lower wins ties (imported, not re-implemented)
    return _snap_to_closest(target, eligible_actions)


# ---------------------------------------------------------------------------
# Self-test (run: python src/rule_based_baseline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    all_discounts = [0, 5, 10, 15, 20]
    new_customer_discounts = [0, 5]   # guardrail blocks 10/15/20 for new customers

    test_cases = [
        # (context, eligible, expected, description)
        ({"demand_index": 0}, all_discounts,          20, "Low demand  + all eligible → 20"),
        ({"demand_index": 1}, all_discounts,          10, "Med demand  + all eligible → 10"),
        ({"demand_index": 2}, all_discounts,           0, "High demand + all eligible → 0"),
        ({"demand_index": 0}, new_customer_discounts,  5, "Low demand  + [0,5] only   → 5  (guardrail wins, NOT 20)"),
        ({"demand_index": 2}, new_customer_discounts,  0, "High demand + [0,5] only   → 0"),
        ({"demand_index": 0}, [],                   None, "Empty list  → None"),
    ]

    print(f"{'demand':>7} {'eligible':<20} {'expected':>8} {'got':>8} {'pass':>6}")
    print("-" * 65)
    all_pass = True
    for ctx, actions, expected, desc in test_cases:
        got = rule_based_baseline(ctx, actions)
        ok = got == expected
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"{ctx['demand_index']:>7} {str(actions):<20} {str(expected):>8} "
            f"{str(got):>8} {status:>6}"
        )
        if not ok:
            print(f"    FAIL: {desc}")

    print()

    # ------------------------------------------------------------------
    # End-to-end wiring: run against REAL guardrail + live_generator output
    # ------------------------------------------------------------------
    print("=" * 65)
    print("End-to-end wiring test (5 live rounds via guardrails + live_generator)")
    print("=" * 65)
    try:
        import itertools
        from live_generator import generate_live_rounds, build_demo_registry
        from guardrails import get_eligible_actions
        from fixed_baseline import fixed_baseline

        registry = build_demo_registry()
        live_gen = generate_live_rounds(registry)

        for i, (ctx, customer, _) in enumerate(itertools.islice(live_gen, 5)):
            eligible = get_eligible_actions(ctx, customer)

            # Bucket DemandIndex (float 0-1) into demand_index int 0/1/2.
            # live_generator produces DemandIndex as a float; rule_based_baseline
            # expects a bucketed int. We map: low float => high demand (index 2),
            # high float => low demand (index 0). This matches the domain intuition
            # that a high DemandIndex value means demand is high.
            raw_demand = ctx.get("DemandIndex", 0.5)
            if raw_demand >= 0.67:
                demand_index = 2   # High demand
            elif raw_demand >= 0.34:
                demand_index = 1   # Medium demand
            else:
                demand_index = 0   # Low demand
            bucketed_ctx = {**ctx, "demand_index": demand_index}

            rb_pick = rule_based_baseline(bucketed_ctx, eligible)
            fx_pick = fixed_baseline(eligible)

            assert rb_pick is None or rb_pick in eligible, \
                f"Round {i+1}: rule_based picked {rb_pick} which is NOT in eligible {eligible}!"
            assert fx_pick is None or fx_pick in eligible, \
                f"Round {i+1}: fixed_baseline picked {fx_pick} which is NOT in eligible {eligible}!"

            demand_label = ["Low", "Medium", "High"][demand_index]
            print(
                f"  Round {i+1} | Tier={customer.LoyaltyTier:<8} | Demand={demand_label:<6} "
                f"| Eligible={eligible} | RuleBased={rb_pick} | Fixed={fx_pick}"
            )

        print("\nAll picks are inside eligible_actions - OK")
        print("All tests passed" if all_pass else "SOME SELF-TESTS FAILED")

    except ImportError as e:
        print(f"  [Skipped wiring test -- import error: {e}]")
        print("All self-tests passed" if all_pass else "SOME SELF-TESTS FAILED")
