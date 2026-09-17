import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from features import mol_to_features, compute_dataset_features

RAW_HUSKIN = os.path.join(BASE_DIR, "data", "raw", "group_B", "U10_huskinDB.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FILE_A = os.path.join(PROCESSED_DIR, "dataset_evaluators_group_A.csv")
FILE_B = os.path.join(PROCESSED_DIR, "dataset_evaluators_group_B.csv")
FILE_COMB = os.path.join(PROCESSED_DIR, "dataset_evaluators_combined.csv")
CACHE_NPZ = os.path.join(PROCESSED_DIR, "features_cache_1036.npz")

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

def run_integration():
    print("=" * 80)
    print("STEP 1: Load and Process HuskinDB")
    print("=" * 80)
    df_raw = pd.read_csv(RAW_HUSKIN)
    print(f"Loaded {len(df_raw)} raw rows from {RAW_HUSKIN}")

    huskin_records = []
    for _, row in df_raw.iterrows():
        smi = canonicalize_smiles(row.get("Smiles"))
        try:
            kp = float(row.get("logkp (cm/s)"))
            if not np.isnan(kp) and smi is not None:
                huskin_records.append({"canonical_smiles": smi, "log_kp": kp})
        except:
            pass

    df_h = pd.DataFrame(huskin_records)
    print(f"Parsed {len(df_h)} valid measurements with canonical SMILES.")
    df_h_agg = df_h.groupby("canonical_smiles", as_index=False)["log_kp"].median()
    print(f"Unique canonical molecules in HuskinDB: {len(df_h_agg)}")

    print("\n" + "=" * 80)
    print("STEP 2: Enforce Strict Disjointness D_A ∩ D_B = ∅")
    print("=" * 80)
    df_A = pd.read_csv(FILE_A)
    df_B = pd.read_csv(FILE_B)
    print(f"Current Group A molecules: {len(df_A):,}")
    print(f"Current Group B molecules: {len(df_B):,}")
    print(f"Current Group B with log_kp: {df_B['log_kp'].notna().sum():,}")

    smiles_A = set(df_A["canonical_smiles"])
    overlap_A = set(df_h_agg["canonical_smiles"]).intersection(smiles_A)
    print(f"Molecules overlapping with Group A (MUST EXCLUDE): {len(overlap_A)}")

    df_h_eligible = df_h_agg[~df_h_agg["canonical_smiles"].isin(overlap_A)].copy()
    print(f"Molecules eligible for Group B: {len(df_h_eligible)}")

    print("\n" + "=" * 80)
    print("STEP 3: Merge into Group B")
    print("=" * 80)
    h_dict = dict(zip(df_h_eligible["canonical_smiles"], df_h_eligible["log_kp"]))

    # Update existing molecules in Group B
    updated_existing = 0
    newly_labeled_existing = 0
    for idx, row in df_B.iterrows():
        smi = row["canonical_smiles"]
        if smi in h_dict:
            h_val = h_dict[smi]
            curr_kp = row["log_kp"]
            
            # Update source_dataset string
            sources = str(row["source_dataset"]).split(";")
            if "U10_huskinDB" not in sources:
                sources.append("U10_huskinDB")
                df_B.at[idx, "source_dataset"] = ";".join(sorted(sources))

            # Update log_kp
            if pd.isna(curr_kp):
                df_B.at[idx, "log_kp"] = h_val
                newly_labeled_existing += 1
            else:
                # Average existing and HuskinDB
                df_B.at[idx, "log_kp"] = np.mean([curr_kp, h_val])
                updated_existing += 1

    print(f"Existing Group B molecules newly labeled with log_kp: {newly_labeled_existing}")
    print(f"Existing Group B molecules with updated consensus log_kp: {updated_existing}")

    # Add brand new molecules from HuskinDB to Group B
    smiles_B_existing = set(df_B["canonical_smiles"])
    new_smiles = [smi for smi in df_h_eligible["canonical_smiles"] if smi not in smiles_B_existing]
    print(f"Brand new molecules to add to Group B: {len(new_smiles)}")

    new_rows = []
    for smi in new_smiles:
        new_rows.append({
            "molecule_id": "", # will be reassigned
            "canonical_smiles": smi,
            "source_group": "group_B",
            "source_dataset": "U10_huskinDB",
            "absorption_max_nm": np.nan,
            "log_extinction": np.nan,
            "photochem_efficiency": np.nan,
            "log_kp": h_dict[smi],
            "skin_sensitization": np.nan,
            "skin_irritation": np.nan
        })

    df_new_B = pd.DataFrame(new_rows)
    df_B_final = pd.concat([df_B, df_new_B], ignore_index=True)

    # Re-assign molecule IDs for Group B
    df_B_final["molecule_id"] = [f"MOL_B_{i+1:06d}" for i in range(len(df_B_final))]

    # Strict check: D_A ∩ D_B = ∅
    intersection = set(df_A["canonical_smiles"]).intersection(set(df_B_final["canonical_smiles"]))
    assert len(intersection) == 0, f"VIOLATION: D_A and D_B have {len(intersection)} overlapping molecules!"
    print(f"\n[VERIFIED] D_A ∩ D_B = ∅: Intersection size is {len(intersection)} molecules.")

    # Combine
    df_comb = pd.concat([df_A, df_B_final], ignore_index=True)

    print("\n" + "=" * 80)
    print("STEP 4: Summary of Datasets")
    print("=" * 80)
    print(f"Final Group A size: {len(df_A):,}")
    print(f"Final Group B size: {len(df_B_final):,}")
    print(f"Final Combined size: {len(df_comb):,}")
    print("\nLabeled target properties counts:")
    print(df_comb[["source_group", "absorption_max_nm", "log_extinction", "photochem_efficiency", "log_kp", "skin_sensitization", "skin_irritation"]].groupby("source_group").count())

    # Save CSVs
    df_B_final.to_csv(FILE_B, index=False)
    df_comb.to_csv(FILE_COMB, index=False)
    print(f"Saved updated {FILE_B}")
    print(f"Saved updated {FILE_COMB}")

    print("\n" + "=" * 80)
    print("STEP 5: Update Feature Cache (1036-dim)")
    print("=" * 80)
    if os.path.exists(CACHE_NPZ):
        print(f"Loading previous cache from {CACHE_NPZ}...")
        old_data = np.load(CACHE_NPZ)
        old_X = old_data["X"]
        old_mask = old_data["valid_mask"]
        print(f"Old cache shape: X={old_X.shape}, valid_mask={old_mask.shape}")
        
        # We know that the first len(old_X) molecules in df_comb are unchanged
        print(f"Computing 1036-dim features for {len(new_smiles)} new molecules...")
        new_X, new_mask = compute_dataset_features(new_smiles)
        
        updated_X = np.vstack([old_X, new_X])
        updated_mask = np.concatenate([old_mask, new_mask])
    else:
        print("Computing features for full combined dataset...")
        updated_X, updated_mask = compute_dataset_features(df_comb["canonical_smiles"])

    print(f"Updated cache shape: X={updated_X.shape}, valid_mask={updated_mask.shape}")
    assert len(updated_X) == len(df_comb), f"Feature matrix length {len(updated_X)} != dataset length {len(df_comb)}"
    np.savez_compressed(CACHE_NPZ, X=updated_X, valid_mask=updated_mask)
    print(f"Successfully saved updated feature cache to {CACHE_NPZ}")
    print("=" * 80)
    print("INTEGRATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    run_integration()
