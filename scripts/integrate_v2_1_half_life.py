import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FILE_A = os.path.join(PROCESSED_DIR, "dataset_evaluators_group_A.csv")
FILE_B = os.path.join(PROCESSED_DIR, "dataset_evaluators_group_B.csv")
FILE_COMB = os.path.join(PROCESSED_DIR, "dataset_evaluators_combined.csv")
RAW_M01 = os.path.join(BASE_DIR, "data", "raw", "group_A", "M01_photoswitches.csv")

def canonicalize_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None
    return None

def run():
    print("=" * 70)
    print("STEP 1: Extract Thermal Half-Life from M01 Photoswitches")
    print("=" * 70)
    df_p = pd.read_csv(RAW_M01)
    k_col = "rate of thermal isomerisation from Z-E in s-1"
    sub = df_p[df_p[k_col].notna()].copy()
    print(f"Total entries with thermal rate in M01: {len(sub)}")

    sub["can"] = sub["SMILES"].apply(canonicalize_smiles)
    sub = sub.dropna(subset=["can"]).copy()
    
    # Calculate log10(t_1/2 in seconds)
    sub["t_half_s"] = np.log(2) / sub[k_col].astype(float)
    sub["log_half_life"] = np.log10(sub["t_half_s"])

    # Aggregate by canonical smiles if duplicates
    h_agg = sub.groupby("can")["log_half_life"].median().to_dict()
    print(f"Unique canonical molecules with log_half_life: {len(h_agg)}")

    print("\n" + "=" * 70)
    print("STEP 2: Map to Group A Dataset")
    print("=" * 70)
    df_A = pd.read_csv(FILE_A)
    df_B = pd.read_csv(FILE_B)

    print(f"Current Group A size: {len(df_A):,}")
    print(f"Current Group B size: {len(df_B):,}")

    df_A["log_half_life"] = df_A["canonical_smiles"].map(h_agg)
    print(f"Group A molecules labeled with log_half_life: {df_A['log_half_life'].notna().sum()}")

    # Group B has no log_half_life (strictly NaN)
    df_B["log_half_life"] = np.nan

    # Check strict disjointness
    s_A = set(df_A["canonical_smiles"])
    s_B = set(df_B["canonical_smiles"])
    overlap = s_A.intersection(s_B)
    assert len(overlap) == 0, f"Disjointness violation: {len(overlap)} overlapping molecules!"
    print(f"✓ [PASS] D_A ∩ D_B = ∅ verified: 0 overlapping molecules.")

    # Reorder columns
    cols = [
        "molecule_id", "canonical_smiles", "source_group", "source_dataset",
        "absorption_max_nm", "log_extinction", "photochem_efficiency", "log_half_life",
        "log_kp", "skin_sensitization", "skin_irritation"
    ]
    df_A = df_A[cols]
    df_B = df_B[cols]
    df_comb = pd.concat([df_A, df_B], ignore_index=True)

    print("\n" + "=" * 70)
    print("STEP 3: Dataset Summary with 7 Properties")
    print("=" * 70)
    print(df_comb.groupby("source_group")[[
        "absorption_max_nm", "log_extinction", "photochem_efficiency", "log_half_life",
        "log_kp", "skin_sensitization", "skin_irritation"
    ]].count().T)

    df_A.to_csv(FILE_A, index=False)
    df_B.to_csv(FILE_B, index=False)
    df_comb.to_csv(FILE_COMB, index=False)
    print(f"Successfully saved {FILE_A}")
    print(f"Successfully saved {FILE_B}")
    print(f"Successfully saved {FILE_COMB}")

if __name__ == "__main__":
    run()
