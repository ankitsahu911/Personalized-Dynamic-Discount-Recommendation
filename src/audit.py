"""
Milestone 28 - Guardrail Compliance Audit
Retail Discount-Recommendation Bandit Project

What this file does:
    Audits every log entry from both bandits' smoke test runs to confirm
    that chosen_discount is ALWAYS a member of eligible_discounts for that
    exact round. A single violation would mean the bandit picked a discount
    that the guardrails had blocked -- a critical correctness failure.

    On ANY violation: prints the full offending entry (not just "violation found").
    At the end: prints a clear compliance summary for both bandits.

Run:
    python src/audit.py

    (Must run AFTER smoke_test.py, which saves the log files to reports/)
"""

import json
import sys
import os
import pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def audit_logs(logs: list, bandit_name: str) -> bool:
    """
    Check every log entry: chosen_discount must be in eligible_discounts.

    Parameters
    ----------
    logs        : list of dicts, each with 'chosen_discount' and 'eligible_discounts'
    bandit_name : str, for display only

    Returns
    -------
    bool : True if fully compliant, False if any violations found
    """
    total      = len(logs)
    violations = []

    for i, entry in enumerate(logs):
        chosen   = entry["chosen_discount"]
        eligible = entry["eligible_discounts"]

        if chosen not in eligible:
            violations.append((i + 1, entry))  # 1-indexed round number

    print(f"\n{'=' * 55}")
    print(f"  Audit: {bandit_name}")
    print(f"{'=' * 55}")

    if violations:
        print(f"\n  VIOLATIONS FOUND -- {len(violations)} / {total} rounds non-compliant")
        print(f"\n  Full offending entries:")
        for round_num, entry in violations:
            print(f"\n  --- Round {round_num} VIOLATION ---")
            print(f"    customer_tier     : {entry.get('customer_tier', 'N/A')}")
            print(f"    eligible_discounts: {entry['eligible_discounts']}")
            print(f"    chosen_discount   : {entry['chosen_discount']}")
            print(f"    purchased         : {entry.get('purchased', 'N/A')}")
            print(f"    profit            : {entry.get('profit', 'N/A')}")
        print(f"\n  RESULT: FAILED -- guardrail compliance broken")
        return False
    else:
        print(f"\n  {total} / {total} rounds compliant -- no violations")
        print(f"  RESULT: PASSED")
        return True


if __name__ == "__main__":
    reports_dir = pathlib.Path(__file__).parent.parent / "reports"

    linucb_log_path = reports_dir / "smoke_test_linucb_logs.json"
    ts_log_path     = reports_dir / "smoke_test_ts_logs.json"

    print("=" * 55)
    print("Milestone 28 -- Guardrail Compliance Audit")
    print("Verifying: chosen_discount IN eligible_discounts, every round")
    print("=" * 55)

    # Check log files exist
    if not linucb_log_path.exists() or not ts_log_path.exists():
        print("\nERROR: Log files not found in reports/")
        print("Run smoke_test.py first to generate them:")
        print("  python src/smoke_test.py")
        sys.exit(1)

    with open(linucb_log_path) as f:
        linucb_logs = json.load(f)
    with open(ts_log_path) as f:
        ts_logs = json.load(f)

    ucb_passed = audit_logs(linucb_logs, "DisjointLinUCB")
    ts_passed  = audit_logs(ts_logs,     "ThompsonSampling")

    print(f"\n{'=' * 55}")
    print(f"FINAL VERDICT")
    print(f"{'=' * 55}")
    if ucb_passed and ts_passed:
        print(f"  Both bandits: FULLY COMPLIANT")
        print(f"  Guardrails are respected in every single round.")
        print(f"  Milestone 28 PASSED.")
    else:
        failed = []
        if not ucb_passed: failed.append("DisjointLinUCB")
        if not ts_passed:  failed.append("ThompsonSampling")
        print(f"  FAILED for: {', '.join(failed)}")
        print(f"  Debug: check select_action() -- it must ONLY score eligible_arms,")
        print(f"         and eligible_arms must come from get_eligible_actions().")
        sys.exit(1)
