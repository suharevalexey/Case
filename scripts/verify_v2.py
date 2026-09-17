import os
import sys
import json
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def verify():
    print("=" * 70)
    print("RUNNING VERIFICATION SUITE FOR VERSION 2.0")
    print("=" * 70)

    # 1. Check datasets
    df_A = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_evaluators_group_A.csv"))
    df_B = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_evaluators_group_B.csv"))
    df_comb = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_evaluators_combined.csv"))

    s_A = set(df_A["canonical_smiles"])
    s_B = set(df_B["canonical_smiles"])
    overlap = s_A.intersection(s_B)
    assert len(overlap) == 0, f"Violation: {len(overlap)} molecules overlap between D_A and D_B!"
    print(f"✓ [PASS] D_A ∩ D_B = ∅ verified: 0 overlapping molecules.")
    print(f"         Group A size: {len(s_A):,}")
    print(f"         Group B size: {len(s_B):,}")
    print(f"         Combined size: {len(df_comb):,}")
    assert len(df_comb) == len(df_A) + len(df_B), "Row counts do not match!"

    # 2. Check log_kp count
    kp_count = df_B["log_kp"].notna().sum()
    print(f"✓ [PASS] Group B log_kp labeled points: {kp_count} (target >= 270)")
    assert kp_count >= 270

    # 3. Check performance report
    with open(os.path.join(RESULTS_DIR, "evaluators_performance_report.json"), "r", encoding="utf-8") as f:
        perf = json.load(f)
    r2_val = perf["log_kp"]["R2"]
    mae_val = perf["log_kp"]["MAE"]
    print(f"✓ [PASS] Evaluator log_kp test R^2: {r2_val:.4f} (v1.0 was 0.4383), MAE: {mae_val:.4f}")
    assert r2_val > 0.50

    # 4. Check generated candidates
    df_gen = pd.read_csv(os.path.join(RESULTS_DIR, "generated_B0.csv"))
    assert len(df_gen) == 3000, f"Expected 3000 generated molecules, found {len(df_gen)}"
    pass_rate = df_gen["pass_constraints"].mean()
    print(f"✓ [PASS] Generated B0 molecules: {len(df_gen):,}, Overall Pass Constraints: {pass_rate:.2%}")

    # 5. Check metrics summary
    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_summary_B0.csv"))
    assert len(df_metrics) == 4, f"Expected 4 rows (3 seeds + 1 aggregated), found {len(df_metrics)}"
    print(f"✓ [PASS] Metrics summary has all 3 seeds and aggregated row.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify()
