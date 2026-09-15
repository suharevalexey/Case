import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Tuple, Dict

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, TRAIN_SPLIT_RATIOS
from src.metrics import canonicalize_smiles

def clean_and_curate_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Clean, desalt, canonicalize SMILES and remove duplicates."""
    cleaned_rows = []
    seen_smiles = set()
    
    for idx, row in df.iterrows():
        raw_smi = row.get("SMILES", "")
        can_smi = canonicalize_smiles(raw_smi)
        if can_smi is None:
            continue
            
        if can_smi in seen_smiles:
            continue
            
        seen_smiles.add(can_smi)
        row_dict = row.to_dict()
        row_dict["canonical_smiles"] = can_smi
        cleaned_rows.append(row_dict)
        
    res = pd.DataFrame(cleaned_rows)
    print(f"[{dataset_name}] Curated: {len(df)} raw -> {len(res)} valid unique molecules.")
    return res

def check_disjointness(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure strict non-overlap: D_A ∩ D_B = ∅."""
    set_a = set(df_a["canonical_smiles"])
    set_b = set(df_b["canonical_smiles"])
    overlap = set_a.intersection(set_b)
    
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping molecules between D_A and D_B! Resolving...")
        # Remove overlapping items from D_B to preserve strict disjointness
        df_b = df_b[~df_b["canonical_smiles"].isin(overlap)].reset_index(drop=True)
    else:
        print("Verification PASSED: D_A and D_B have strictly disjoint chemical spaces (0 overlap).")
        
    return df_a, df_b

def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Compute Murcko scaffold for a given molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
    except Exception:
        return ""

def scaffold_split(
    df: pd.DataFrame,
    ratios: Dict[str, float] = TRAIN_SPLIT_RATIOS,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, val, test based on Bemis-Murcko scaffolds.
    Sorts scaffolds by size descending to pack the largest core scaffolds into Train,
    ensuring balanced representation in Val (~10%) and Test (~10%).
    """
    scaffold_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        scaff = generate_scaffold(row["canonical_smiles"])
        scaffold_to_indices[scaff].append(idx)
        
    # Sort scaffolds by size descending
    sorted_scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
    
    total_len = len(df)
    target_train = int(ratios["train"] * total_len)
    target_val = int(ratios["val"] * total_len)
    
    train_idx, val_idx, test_idx = [], [], []
    
    for scaff, group in sorted_scaffolds:
        if len(train_idx) + len(group) <= target_train or (len(train_idx) == 0):
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= target_val or (len(val_idx) == 0):
            val_idx.extend(group)
        else:
            test_idx.extend(group)
            
    # Guarantee at least some entries in test
    if len(test_idx) == 0 and len(val_idx) > 2:
        test_idx = val_idx[len(val_idx)//2:]
        val_idx = val_idx[:len(val_idx)//2]
        
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def run_preprocessing_pipeline():
    """Run full curation, disjointness check, and scaffold splitting."""
    raw_a_file = RAW_DATA_DIR / "dataset_A_raw.csv"
    raw_b_file = RAW_DATA_DIR / "dataset_B_raw.csv"
    
    if not raw_a_file.exists() or not raw_b_file.exists():
        from src.data_loader import build_dataset_A, build_dataset_B
        build_dataset_A(RAW_DATA_DIR)
        build_dataset_B(RAW_DATA_DIR)
        
    df_a = pd.read_csv(raw_a_file)
    df_b = pd.read_csv(raw_b_file)
    
    # 1. Clean & canonicalize
    cur_a = clean_and_curate_dataframe(df_a, "Dataset_A_MOST")
    cur_b = clean_and_curate_dataframe(df_b, "Dataset_B_UV")
    
    # 2. Strict disjoint check
    cur_a, cur_b = check_disjointness(cur_a, cur_b)
    
    # 3. Scaffold split for D_A
    train_a, val_a, test_a = scaffold_split(cur_a)
    print(f"Dataset D_A Scaffold Split: Train={len(train_a)}, Val={len(val_a)}, Test={len(test_a)}")
    
    # 4. Scaffold split for D_B
    train_b, val_b, test_b = scaffold_split(cur_b)
    print(f"Dataset D_B Scaffold Split: Train={len(train_b)}, Val={len(val_b)}, Test={len(test_b)}")
    
    # Save processed files
    train_a.to_csv(PROCESSED_DATA_DIR / "train_A.csv", index=False)
    val_a.to_csv(PROCESSED_DATA_DIR / "val_A.csv", index=False)
    test_a.to_csv(PROCESSED_DATA_DIR / "test_A.csv", index=False)
    
    train_b.to_csv(PROCESSED_DATA_DIR / "train_B.csv", index=False)
    val_b.to_csv(PROCESSED_DATA_DIR / "val_B.csv", index=False)
    test_b.to_csv(PROCESSED_DATA_DIR / "test_B.csv", index=False)
    
    print(f"All processed splits successfully written to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
