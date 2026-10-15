"""
Milestone 21 — Fixed-Target Baseline
-------------------------------------
Policy: always try to offer a fixed discount (default 10%).
If the target is not available, snap to the closest eligible discount.

Tie-break rule (must be consistent across all baselines):
    When two discounts are equidistant from the target,
    the LOWER value wins. This is the conservative choice — we prefer
    giving less discount when the benefit is identical.

Usage:
    from fixed_baseline import fixed_baseline

    fixed_baseline([0, 5])          # → 5   (10 not available; 5 is closer than 0)
    fixed_baseline([5, 10, 15])     # → 10  (exact match)
    fixed_baseline([10, 15, 20])    # → 10  (exact match)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _snap_to_closest(target: int, candidates: List[int]) -> int:
    """
    Return the element in *candidates* that is numerically closest to *target*.

    Tie-break rule: when two candidates are equidistant, the LOWER value wins.
    This function is intentionally exported so rule_based_baseline can import
    and reuse it rather than re-implement the same logic.
    """
    # min() with a custom key: primary sort = abs distance, secondary = value itself
    # Because min() returns the first minimum it encounters for equal keys, and
    # we sort ascending on value as the tiebreaker, we pair (distance, value) as
    # the key — Python compares tuples lexicographically, so equal distances
    # will fall back to comparing values, and the smaller value wins.
    return min(candidates, key=lambda x: (abs(x - target), x))


def fixed_baseline(eligible_actions: List[int], target: int = 10) -> Optional[int]:
    """
    Return the best available discount according to a fixed-target policy.

    Parameters
    ----------
    eligible_actions : list[int]
        Discounts pre-filtered by the guardrails (e.g. [0, 5, 10, 15, 20]).
    target : int, default 10
        The preferred discount percentage.

    Returns
    -------
    int or None
        The chosen discount, or None if eligible_actions is empty.
    """
    if not eligible_actions:
        logger.warning("fixed_baseline received an empty eligible_actions list — returning None.")
        return None

    # Exact match — fast path
    if target in eligible_actions:
        return target

    # Snap to nearest, lower wins ties
    return _snap_to_closest(target, eligible_actions)


# ---------------------------------------------------------------------------
# Self-test (run: python src/fixed_baseline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ([0, 5],       10, 5,  "10 not available; 5 is closer to 10 than 0 is"),
        ([5, 10, 15],  10, 10, "exact match"),
        ([10, 15, 20], 10, 10, "exact match"),
        ([0, 20],      10, 0,  "equidistant (|0-10|=|20-10|=10) → lower wins → 0"),
        ([],           10, None, "empty list → None"),
    ]

    print(f"{'eligible_actions':<20} {'target':>6} {'expected':>8} {'got':>8} {'pass':>6}")
    print("-" * 60)
    all_pass = True
    for actions, tgt, expected, note in test_cases:
        got = fixed_baseline(actions, target=tgt)
        ok = got == expected
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL <--"
        print(f"{str(actions):<20} {tgt:>6} {str(expected):>8} {str(got):>8} {status:>8}")
        if not ok:
            print(f"    NOTE: {note}")

    print()
    print("All tests passed" if all_pass else "SOME TESTS FAILED")
