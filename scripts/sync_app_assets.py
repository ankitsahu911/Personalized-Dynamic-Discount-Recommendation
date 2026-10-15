"""
Developer script to sync and thin app assets into app/assets/
Ensures app is 100% self-contained and loads instantaneously.
"""

import os
import shutil
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "app", "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
DASHBOARD_DIR = os.path.join(ASSETS_DIR, "dashboard")


def sync_assets():
    print("=" * 80)
    print("SYNCING APPLICATION ASSETS INTO app/assets/")
    print("=" * 80)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    # 1. Model files
    model_files = ["linucb_v1.npz", "linucb_v1_meta.json"]
    for mf in model_files:
        src = os.path.join(PROJECT_ROOT, "models", mf)
        dst = os.path.join(MODELS_DIR, mf)
        assert os.path.exists(src), f"Missing source model file: {src}"
        shutil.copy2(src, dst)
        print(f"  Copied model: {mf} ({os.path.getsize(dst):,} bytes)")

    # 2. Dashboard files
    dash_sources = [
        (os.path.join(PROJECT_ROOT, "outputs", "dashboard_data", "cumulative_series_seed42.csv"), "cumulative_series_seed42.csv"),
        (os.path.join(PROJECT_ROOT, "outputs", "dashboard_data", "summary_kpis.json"), "summary_kpis.json"),
        (os.path.join(PROJECT_ROOT, "results", "comparison_online.csv"), "comparison_online.csv"),
        (os.path.join(PROJECT_ROOT, "results", "comparison_frozen.csv"), "comparison_frozen.csv"),
        (os.path.join(PROJECT_ROOT, "results", "comparison_lift.csv"), "comparison_lift.csv"),
    ]

    for src, fname in dash_sources:
        dst = os.path.join(DASHBOARD_DIR, fname)
        assert os.path.exists(src), f"Missing source dashboard file: {src}"
        shutil.copy2(src, dst)
        print(f"  Copied dashboard file: {fname} ({os.path.getsize(dst):,} bytes)")

    # 3. Build thinned multi-seed array from cumulative_by_seed.npz
    npz_src = os.path.join(PROJECT_ROOT, "results", "cumulative_by_seed.npz")
    if os.path.exists(npz_src):
        print("\nBuilding thinned multi-seed cumulative statistics from cumulative_by_seed.npz...")
        npz = np.load(npz_src)
        # Shape: (10 seeds, 5 policies, 20000 rounds)
        cum_profit = npz["cum_profit"]
        cum_rev = npz["cum_revenue"]
        cum_disc = npz["cum_discount_given"]
        cum_reg = npz["cum_regret"]
        policies = [str(p) for p in npz["policy_order"]]

        step = 20
        rounds_thinned = np.arange(1, 20001, step)
        indices = rounds_thinned - 1

        rows = []
        for p_idx, p_name in enumerate(policies):
            prof_p = cum_profit[:, p_idx, indices]  # (10, n_thinned)
            rev_p = cum_rev[:, p_idx, indices]
            disc_p = cum_disc[:, p_idx, indices]
            reg_p = cum_reg[:, p_idx, indices]

            for i_idx, r_num in enumerate(rounds_thinned):
                rows.append({
                    "round_idx": int(r_num),
                    "policy": p_name,
                    "profit_mean": float(np.mean(prof_p[:, i_idx])),
                    "profit_std": float(np.std(prof_p[:, i_idx])),
                    "revenue_mean": float(np.mean(rev_p[:, i_idx])),
                    "revenue_std": float(np.std(rev_p[:, i_idx])),
                    "discount_given_mean": float(np.mean(disc_p[:, i_idx])),
                    "discount_given_std": float(np.std(disc_p[:, i_idx])),
                    "regret_mean": float(np.mean(reg_p[:, i_idx])),
                    "regret_std": float(np.std(reg_p[:, i_idx])),
                })

        thinned_df = pd.DataFrame(rows)
        thinned_csv = os.path.join(DASHBOARD_DIR, "cumulative_mean_std_thinned.csv")
        thinned_df.to_csv(thinned_csv, index=False)
        print(f"  Generated thinned multi-seed summary: {thinned_csv} ({os.path.getsize(thinned_csv):,} bytes)")

    print("\nAsset synchronization complete.")


if __name__ == "__main__":
    sync_assets()
