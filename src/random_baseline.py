"""
Milestone 20 � Random Discount Baseline
Retail Discount-Recommendation Bandit Project

Per the Phase 4 doc, this one is fully self-contained -- it only needs to
know that eligible_actions is a list of ints, a subset of {0,5,10,15,20},
already filtered by whatever guardrails you built in Milestones 15-18. No
placeholders to flag this time; nothing here is invented against your
project's real interfaces.

A NOTE ON THE `seed` PARAMETER (read before you rely on it):
Passing the SAME seed on every call makes THAT call deterministic -- same
seed in, same value out, every time. That's useful for reproducing one
specific decision in a test or a log. It is NOT the way to get a
reproducible SEQUENCE of many different random draws (e.g. across a
10,000-round bandit run) -- for that, seed Python's global `random` module
ONCE at the start of your script/experiment (random.seed(42)) and then
call random_baseline(eligible_actions) without passing seed on every call,
same as the test below does.

Run:
    python milestone20_random_baseline.py
"""

import random
from collections import Counter
from typing import List, Optional


def random_baseline(eligible_actions: List[int], seed: Optional[int] = None) -> Optional[int]:
    """
    Uniformly random choice from eligible_actions.

    Args:
        eligible_actions: subset of {0, 5, 10, 15, 20}, already filtered
            by tier + margin guardrails elsewhere in the project. This
            function doesn't need to know how that filtering happened.
        seed: optional int. If given, uses a dedicated random.Random(seed)
            for THIS call only -- see the module docstring above for why
            that's different from seeding once for a whole experiment.

    Returns:
        A single int from eligible_actions, or None if eligible_actions is
        empty (logged as a warning, never raises).
    """
    if not eligible_actions:
        print("[random_baseline] WARNING: eligible_actions is empty -- nothing to choose. Returning None.")
        return None

    rng = random.Random(seed) if seed is not None else random
    return rng.choice(eligible_actions)


# ---------------------------------------------------------------------------
# TEST 1 -- 1,000 calls on a fixed list, check the distribution is ~uniform
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)  # seed the RUN once, not each call -- see note above

    fixed_eligible = [5, 10, 15]
    results = [random_baseline(fixed_eligible) for _ in range(1000)]
    counts = Counter(results)

    print("Test 1 -- 1,000 calls on eligible_actions=[5, 10, 15]:")
    for value in fixed_eligible:
        print(f"  {value}: {counts[value]}")
    print(f"  (expect roughly ~333 each if uniform)\n")

    # ------------------------------------------------------------------
    # TEST 2 -- empty list edge case
    # ------------------------------------------------------------------
    print("Test 2 -- empty eligible_actions:")
    result = random_baseline([])
    print(f"  returned: {result}  (expect None + a WARNING printed above)\n")

    # ------------------------------------------------------------------
    # BONUS -- wired into your actual Milestone 17/19 code, not a
    # hand-typed example list. Only runs if those files are present in
    # the same folder; safe to delete this block if you just want the
    # standalone baseline.
    # ------------------------------------------------------------------
    try:
        from round_generator import build_demo_registry
        from live_generator import generate_live_rounds

        print("Bonus -- random_baseline wired into your real get_eligible_actions() output:")
        registry = build_demo_registry(n=20)
        gen = generate_live_rounds(registry)
        for i in range(5):
            context, customer, eligible = next(gen)
            choice = random_baseline(eligible)
            print(
                f"  Round {i+1}: tier-eligible/margin-filtered={eligible} "
                f"-> random_baseline picked {choice}"
            )
    except ImportError:
        print("(Bonus integration test skipped -- Milestone 17/19 files not found in this folder.)")
