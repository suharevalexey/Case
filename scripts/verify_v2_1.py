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
    print("RUNNING VERIFICATION SUITE FOR VERSION 2.1 (7 PROPERTIES)")
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

    # 2. Check property counts
    assert df_A["log_half_life"].notna().sum() == 75, "Expected 75 log_half_life points!"
    assert df_B["log_kp"].notna().sum() >= 270, "Expected >= 270 log_kp points!"
    print(f"✓ [PASS] Group A log_half_life points: {df_A['log_half_life'].notna().sum()}")
    print(f"✓ [PASS] Group B log_kp points: {df_B['log_kp'].notna().sum()}")

    # 3. Check performance report for all 7 properties
    with open(os.path.join(RESULTS_DIR, "evaluators_performance_report.json"), "r", encoding="utf-8") as f:
        perf = json.load(f)
    assert len(perf) == 7, f"Expected 7 evaluated properties, found {len(perf)}"
    assert perf["log_half_life"]["R2"] > 0.40, f"log_half_life R2 too low: {perf['log_half_life']['R2']}"
    assert perf["log_kp"]["R2"] > 0.50, f"log_kp R2 too low: {perf['log_kp']['R2']}"
    print(f"✓ [PASS] All 7 properties verified in performance report:")
    for prop, m in perf.items():
        if m["type"] == "regression":
            print(f"         • {prop:22s}: R^2 = {m['R2']:.4f}, MAE = {m['MAE']:.4f}")
        else:
            print(f"         • {prop:22s}: ROC-AUC = {m['ROC_AUC']:.4f}, Acc = {m['Accuracy']:.4f}")

    # 4. Check generated candidates
    df_gen = pd.read_csv(os.path.join(RESULTS_DIR, "generated_B0.csv"))
    assert len(df_gen) == 3000, f"Expected 3000 generated molecules, found {len(df_gen)}"
    pass_rate = df_gen["pass_constraints"].mean()
    print(f"✓ [PASS] Generated B0 molecules: {len(df_gen):,}, Overall Pass Constraints (7 properties): {pass_rate:.2%}")

    # 5. Check metrics summary
    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_summary_B0.csv"))
    assert len(df_metrics) == 4, f"Expected 4 rows (3 seeds + 1 aggregated), found {len(df_metrics)}"
    print(f"✓ [PASS] Metrics summary has all 3 seeds and aggregated row.")

    print("\n" + "=" * 70)
    print("ALL 7-PROPERTY VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify()
