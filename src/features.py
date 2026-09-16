import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

# Standard 1024-bit Morgan Generator (Radius 2, ECFP4 equivalent)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FractionCSP3",
    "NumAromaticRings",
    "RingCount",
    "HeavyAtomCount",
    "LabuteASA",
    "HallKierAlpha"
]

def mol_to_features(smiles_or_mol):
    """
    Computes 1024-bit Morgan fingerprint + 12 physicochemical RDKit descriptors.
    Total feature vector dimension: 1036.
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol
        
    if mol is None:
        return None
        
    # 1. Morgan Fingerprint (1024-bit)
    fp = MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
    
    # 2. Physicochemical Descriptors (12 features)
    try:
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            Descriptors.FractionCSP3(mol),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.RingCount(mol)),
            float(Descriptors.HeavyAtomCount(mol)),
            Descriptors.LabuteASA(mol),
            Descriptors.HallKierAlpha(mol)
        ], dtype=np.float32)
    except Exception:
        desc = np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
        
    return np.concatenate([fp, desc])

def compute_dataset_features(smiles_series):
    """
    Computes feature matrix for a series/list of SMILES.
    Returns (X, valid_mask)
    """
    features_list = []
    valid_mask = []
    
    for s in smiles_series:
        feat = mol_to_features(str(s))
        if feat is not None:
            features_list.append(feat)
            valid_mask.append(True)
        else:
            features_list.append(np.zeros(1024 + len(DESCRIPTOR_NAMES), dtype=np.float32))
            valid_mask.append(False)
            
    return np.vstack(features_list), np.array(valid_mask, dtype=bool)

def get_or_compute_cached_features(data_csv_path, cache_npz_path):
    """
    Loads features from .npz cache if exists, otherwise computes and caches them.
    """
    if os.path.exists(cache_npz_path):
        print(f"Loading cached features from {cache_npz_path}...")
        data = np.load(cache_npz_path)
        return data["X"], data["valid_mask"]
        
    print(f"Computing features for {data_csv_path}...")
    df = pd.read_csv(data_csv_path)
    X, valid_mask = compute_dataset_features(df["canonical_smiles"])
    
    os.makedirs(os.path.dirname(cache_npz_path), exist_ok=True)
    np.savez_compressed(cache_npz_path, X=X, valid_mask=valid_mask)
    print(f"Saved computed features to {cache_npz_path} (Shape: {X.shape})")
    return X, valid_mask

if __name__ == "__main__":
    test_smi = "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2=CC3=CC=CC=C3" # Benzylidene camphor
    feat = mol_to_features(test_smi)
    print(f"Test molecule feature vector shape: {feat.shape}")
    print(f"Fingerprint bits set: {int(feat[:1024].sum())}")
    print(f"Descriptors: {feat[1024:]}")
