"""
Milestone 31 - Save & Load Trained Bandit Models to Disk
Retail Discount-Recommendation Bandit Project

Saves and loads trained models using .npz (NumPy arrays) + .json (metadata & preprocessing).
NEVER uses pickle for models, ensuring security and cross-environment portability for Streamlit.

Files created:
    models/linucb_v1.npz
    models/linucb_v1_meta.json
    models/thompson_v1.npz
    models/thompson_v1_meta.json
"""

import json
import os
import time
from typing import Dict, List, Optional
import numpy as np

from linucb import (
    DisjointLinUCB,
    NUMERIC_STATS,
    CATEGORICAL_VALUES,
    CONTEXT_DIM,
    build_context_vector as default_build_context_vector,
)
from thompson_sampling import ThompsonSampling


def get_preprocessing_metadata() -> dict:
    """
    Returns full preprocessing specification required to construct the exact
    context vector x in downstream production/Streamlit apps.
    """
    ordered_feature_names = (
        list(NUMERIC_STATS.keys())
        + ["Promotion"]
        + [f"Category_{v}" for v in CATEGORICAL_VALUES["Category"]]
        + [f"Region_{v}" for v in CATEGORICAL_VALUES["Region"]]
        + [f"Seasonality_{v}" for v in CATEGORICAL_VALUES["Seasonality"]]
        + [f"LoyaltyTier_{v}" for v in CATEGORICAL_VALUES["LoyaltyTier"]]
        + ["Bias"]
    )

    return {
        "scaling_method": "z-score",
        "context_dim": CONTEXT_DIM,
        "feature_names": ordered_feature_names,
        "numeric_stats": {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in NUMERIC_STATS.items()},
        "categorical_values": {k: list(v) for k, v in CATEGORICAL_VALUES.items()},
    }


def save_linucb(
    model: DisjointLinUCB,
    path_prefix: str,
    n_rounds: int = 20000,
    main_seed: int = 42,
    sequence_file: str = "training_rounds_seed42_n20000.pkl",
    arm_pull_counts: Optional[Dict[int, int]] = None,
) -> None:
    """
    Save LinUCB model arrays to <path_prefix>.npz and metadata to <path_prefix>_meta.json.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
    arms = list(model.arms)
    n_arms = len(arms)
    d = model.context_dim

    # Stack A and b into arrays
    A_array = np.stack([model.A[a] for a in arms], axis=0)  # (n_arms, d, d)
    b_array = np.stack([model.b[a] for a in arms], axis=0)  # (n_arms, d)

    pulls = np.array([arm_pull_counts.get(a, 0) if arm_pull_counts else 0 for a in arms], dtype=np.int64)

    np.savez_compressed(
        f"{path_prefix}.npz",
        arms=np.array(arms),
        A=A_array,
        b=b_array,
        pull_counts=pulls,
    )

    meta = {
        "model_type": "DisjointLinUCB",
        "arms": arms,
        "context_dim": d,
        "alpha": float(model.alpha),
        "n_rounds": n_rounds,
        "main_seed": main_seed,
        "sequence_file": sequence_file,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing": get_preprocessing_metadata(),
    }

    with open(f"{path_prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[model_io] Saved LinUCB model to {path_prefix}.npz and {path_prefix}_meta.json")


def load_linucb(path_prefix: str) -> DisjointLinUCB:
    """
    Load a trained LinUCB model from disk.
    """
    npz_path = f"{path_prefix}.npz"
    meta_path = f"{path_prefix}_meta.json"

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing model files at {path_prefix}.npz / _meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(npz_path)
    arms = [int(a) for a in data["arms"]]
    context_dim = int(meta["context_dim"])
    alpha = float(meta["alpha"])

    model = DisjointLinUCB(arms=arms, context_dim=context_dim, alpha=alpha)

    for i, a in enumerate(arms):
        model.A[a] = data["A"][i].copy()
        model.b[a] = data["b"][i].copy()
        model.A_inv[a] = np.linalg.inv(model.A[a])

    # Attach preprocessing metadata
    model.preprocessing = meta.get("preprocessing", {})
    return model


def save_thompson(
    model: ThompsonSampling,
    path_prefix: str,
    n_rounds: int = 20000,
    ts_seed: int = 7,
    sequence_file: str = "training_rounds_seed42_n20000.pkl",
) -> None:
    """
    Save Thompson Sampling model arrays to <path_prefix>.npz and metadata to <path_prefix>_meta.json.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
    arms = list(model.arms)

    mu_array = np.stack([model.mu[a] for a in arms], axis=0)        # (n_arms, d)
    Sigma_array = np.stack([model.Sigma[a] for a in arms], axis=0)  # (n_arms, d, d)
    Lambda_array = np.stack([model.Lambda[a] for a in arms], axis=0)
    v_array = np.stack([model.v[a] for a in arms], axis=0)

    np.savez_compressed(
        f"{path_prefix}.npz",
        arms=np.array(arms),
        mu=mu_array,
        Sigma=Sigma_array,
        Lambda=Lambda_array,
        v=v_array,
    )

    meta = {
        "model_type": "ThompsonSampling",
        "arms": arms,
        "context_dim": model.context_dim,
        "sigma": float(model.sigma),
        "ts_seed": ts_seed,
        "n_rounds": n_rounds,
        "sequence_file": sequence_file,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing": get_preprocessing_metadata(),
    }

    with open(f"{path_prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[model_io] Saved Thompson Sampling model to {path_prefix}.npz and {path_prefix}_meta.json")


def load_thompson(path_prefix: str) -> ThompsonSampling:
    """
    Load a trained Thompson Sampling model from disk.
    """
    npz_path = f"{path_prefix}.npz"
    meta_path = f"{path_prefix}_meta.json"

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing model files at {path_prefix}.npz / _meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(npz_path)
    arms = [int(a) for a in data["arms"]]
    context_dim = int(meta["context_dim"])
    sigma = float(meta["sigma"])

    model = ThompsonSampling(arms=arms, context_dim=context_dim, sigma=sigma)

    for i, a in enumerate(arms):
        model.mu[a] = data["mu"][i].copy()
        model.Sigma[a] = data["Sigma"][i].copy()
        if "Lambda" in data:
            model.Lambda[a] = data["Lambda"][i].copy()
        if "v" in data:
            model.v[a] = data["v"][i].copy()

    model.preprocessing = meta.get("preprocessing", {})
    return model
