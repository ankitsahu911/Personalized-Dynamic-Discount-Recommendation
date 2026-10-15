"""
Milestone 27 - 100-Round Smoke Test: LinUCB vs Thompson Sampling
Retail Discount-Recommendation Bandit Project

What this file does:
    Runs both bandits independently for 100 rounds each using run_round()
    from simulator.py, then prints a side-by-side comparison:
        - Total conversions (purchased=True count)
        - Average profit per round
        - Discount choice distribution (are all arms being explored?)
        - Any NaNs in logs (signals a numerical bug)

    RED FLAGS checked automatically:
        [!] Any NaN values anywhere in logs
        [!] Only ONE arm chosen across all 100 rounds (exploration broken)
        [!] Every round = zero profit (simulator or guardrail wiring broken)

Run:
    python src/smoke_test.py
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from linucb import DisjointLinUCB, CONTEXT_DIM
from thompson_sampling import ThompsonSampling
from simulator import run_round
from live_generator import generate_live_rounds, build_demo_registry

ARMS      = [0, 5, 10, 15, 20]
N_ROUNDS  = 100
SEED      = 42


def run_smoke_test(bandit, bandit_name: str, registry, seed: int) -> dict:
    """Run N_ROUNDS rounds for one bandit, return summary stats."""
    random.seed(seed)
    gen  = generate_live_rounds(registry)
    logs = []

    for _ in range(N_ROUNDS):
        log = run_round(bandit, gen)
        logs.append(log)

    # --- Compute summary ---
    profits   = [l["profit"]   for l in logs]
    purchased = [l["purchased"] for l in logs]
    chosen    = [l["chosen_discount"] for l in logs]
    eligible  = [l["eligible_discounts"] for l in logs]

    total_conversions  = sum(purchased)
    avg_profit         = sum(profits) / len(profits)
    discount_dist      = Counter(chosen)
    any_nan            = any(
        p != p for p in profits   # NaN check: NaN != NaN is True in Python
    )

    # --- Red flag detection ---
    red_flags = []
    if any_nan:
        red_flags.append("NaN values found in profit logs")
    if len(discount_dist) == 1:
        red_flags.append(f"Only 1 arm chosen across all {N_ROUNDS} rounds -- exploration broken")
    if total_conversions == 0:
        red_flags.append("Zero conversions in 100 rounds -- simulator/guardrail wiring broken")
    if all(p == 0.0 for p in profits):
        red_flags.append("Every profit = 0.0 -- reward signal is dead")

    return {
        "name":               bandit_name,
        "total_conversions":  total_conversions,
        "avg_profit":         avg_profit,
        "discount_dist":      discount_dist,
        "any_nan":            any_nan,
        "red_flags":          red_flags,
        "logs":               logs,
    }


def print_summary(result: dict):
    print(f"\n{'=' * 55}")
    print(f"  {result['name']}")
    print(f"{'=' * 55}")
    print(f"  Total conversions  : {result['total_conversions']} / {N_ROUNDS}")
    print(f"  Avg profit/round   : Rs {result['avg_profit']:.4f}")
    print(f"  Contains NaN       : {result['any_nan']}")
    print(f"\n  Discount distribution (chosen arm counts):")
    for arm in ARMS:
        count = result['discount_dist'].get(arm, 0)
        bar   = "#" * count
        print(f"    {arm:>3}% : {count:>3}  {bar}")

    if result["red_flags"]:
        print(f"\n  RED FLAGS DETECTED:")
        for flag in result["red_flags"]:
            print(f"    [!] {flag}")
    else:
        print(f"\n  No red flags -- smoke test PASSED")


if __name__ == "__main__":
    print("=" * 55)
    print("Milestone 27 -- 100-Round Smoke Test")
    print(f"Both bandits, {N_ROUNDS} rounds each, seed={SEED}")
    print("=" * 55)

    # Each bandit gets its OWN fresh registry and generator instance
    registry_ucb = build_demo_registry()
    registry_ts  = build_demo_registry()

    linucb_bandit = DisjointLinUCB(arms=ARMS, context_dim=CONTEXT_DIM, alpha=1.0)
    ts_bandit     = ThompsonSampling(arms=ARMS, context_dim=CONTEXT_DIM, sigma=1.0)

    result_ucb = run_smoke_test(linucb_bandit, "DisjointLinUCB  (alpha=1.0)", registry_ucb, seed=SEED)
    result_ts  = run_smoke_test(ts_bandit,     "ThompsonSampling (sigma=1.0)", registry_ts,  seed=SEED + 1)

    print_summary(result_ucb)
    print_summary(result_ts)

    # --- Save logs for Milestone 28 ---
    import json, pathlib
    logs_dir = pathlib.Path(__file__).parent.parent / "reports"
    logs_dir.mkdir(exist_ok=True)

    # Logs contain non-serializable objects (Customer dict) -- serialize safely
    def serialize_log(log):
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in log.items()
                if k != "context"}   # context has full dict, keep it simple

    with open(logs_dir / "smoke_test_linucb_logs.json", "w") as f:
        json.dump([serialize_log(l) for l in result_ucb["logs"]], f, indent=2)
    with open(logs_dir / "smoke_test_ts_logs.json", "w") as f:
        json.dump([serialize_log(l) for l in result_ts["logs"]], f, indent=2)

    print(f"\nLogs saved to reports/ for Milestone 28 audit.")

    # --- Overall verdict ---
    all_clear = not result_ucb["red_flags"] and not result_ts["red_flags"]
    print(f"\n{'SMOKE TEST PASSED -- ready for Milestone 28' if all_clear else 'SMOKE TEST HAS RED FLAGS -- check above'}")
