"""
Milestone 29 - Step 1: Pre-generate and Save Fixed Round Sequence
Retail Discount-Recommendation Bandit Project

Generates N_ROUNDS = 20,000 live rounds with MAIN_SEED = 42 and saves them
to data/rounds/training_rounds_seed42_n20000.pkl.

Why pickle (.pkl)?
    Pickle directly preserves Python dataclasses (Customer), nested context dicts,
    and Python int lists without schema casting, string parsing, or precision loss,
    and loads in ~0.1 seconds.

Round record schema:
    {
        "round_idx": int,
        "context": dict,
        "customer": Customer,
        "eligible_discounts": list[int],
        "round_seed": int,
    }
"""

import os
import pickle
import random
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_generator import generate_live_rounds, build_demo_registry
from round_generator import Customer

N_ROUNDS = 20000
MAIN_SEED = 42
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "rounds",
    "training_rounds_seed42_n20000.pkl",
)


def generate_and_save_rounds(n_rounds: int = N_ROUNDS, main_seed: int = MAIN_SEED, output_path: str = OUTPUT_PATH):
    print("=" * 65)
    print(f"Generating {n_rounds:,} fixed training rounds (MAIN_SEED = {main_seed})...")
    print("=" * 65)

    random.seed(main_seed)
    np.random.seed(main_seed)

    registry = build_demo_registry(n=50)
    gen = generate_live_rounds(registry)

    rounds = []
    for i in range(n_rounds):
        context, customer, eligible_discounts = next(gen)
        # Create a snapshot copy of customer state so subsequent rounds don't mutate earlier snapshots
        customer_snapshot = Customer(
            CustomerID=customer.CustomerID,
            PurchaseCount=customer.PurchaseCount,
            RecencyDays=customer.RecencyDays,
            LoyaltyTier=customer.LoyaltyTier,
            CurrentBasketValue=customer.CurrentBasketValue,
        )

        round_seed = int((main_seed * 100003 + i * 37) % (2**31 - 1))

        rounds.append({
            "round_idx": i + 1,
            "context": dict(context),
            "customer": customer_snapshot,
            "eligible_discounts": list(eligible_discounts),
            "round_seed": round_seed,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1:,} / {n_rounds:,} rounds...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(rounds, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(rounds):,} rounds to {output_path}")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    return output_path


def load_training_rounds(path: str = OUTPUT_PATH) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequence file not found at {path}. Run training_rounds.py first.")
    with open(path, "rb") as f:
        rounds = pickle.load(f)
    return rounds


if __name__ == "__main__":
    generate_and_save_rounds()
