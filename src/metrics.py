import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Contrib.SA_Score import sascorer
from typing import List, Tuple, Dict, Set, Optional

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        # Desalt / take largest fragment if multi-component
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None

def compute_sa_score(mol: Chem.Mol) -> float:
    """Calculate Ertl Synthetic Accessibility (SA) Score (1 = easy, 10 = difficult)."""
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 5.0

def compute_potts_guy_log_kp(mol: Chem.Mol) -> float:
    """
    Calculate skin permeability coefficient log Kp (cm/h) using the Potts & Guy (1992) equation:
    log Kp = -2.7 + 0.71 * logP - 0.0061 * MW
    Lower values (< -2.5) indicate limited systemic absorption / higher retention in skin/stratum corneum.
    """
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        log_kp = -2.7 + 0.71 * logp - 0.0061 * mw
        return float(log_kp)
    except Exception:
        return -2.5

def get_morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan Fingerprint (ECFP4)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

def compute_applicability_distance(mol: Chem.Mol, ref_fps: List) -> float:
    """
    Compute distance to Applicability Domain (AD):
    Defined as 1.0 - max(Tanimoto similarity to reference set), i.e., distance to nearest neighbor.
    Range [0, 1]. Lower is closer to training domain.
    """
    if not ref_fps:
        return 0.0
    fp = get_morgan_fp(mol)
    sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
    max_sim = max(sims) if sims else 0.0
    return float(1.0 - max_sim)

def evaluate_generation_metrics(
    generated_smiles: List[str],
    train_smiles_set: Set[str],
    max_diversity_sample: int = 500
) -> Dict[str, float]:
    """
    Compute standard molecular benchmark metrics:
    - Validity
    - Uniqueness
    - Novelty
    - Diversity
    - Mean SA Score
    """
    n_gen = len(generated_smiles)
    if n_gen == 0:
        return {
            "validity": 0.0, "uniqueness": 0.0, "novelty": 0.0,
            "diversity": 0.0, "mean_sa": 0.0
        }
        
    valid_mols = []
    canon_smiles = []
    
    for s in generated_smiles:
        can = canonicalize_smiles(s)
        if can is not None:
            canon_smiles.append(can)
            valid_mols.append(Chem.MolFromSmiles(can))
            
    n_valid = len(valid_mols)
    validity = n_valid / n_gen
    
    if n_valid == 0:
        return {
            "validity": 0.0, "uniqueness": 0.0, "novelty": 0.0,
            "diversity": 0.0, "mean_sa": 0.0
        }
        
    unique_smiles = set(canon_smiles)
    n_unique = len(unique_smiles)
    uniqueness = n_unique / n_valid
    
    # Novelty: fraction of unique generated molecules not present in training set
    novel_count = sum(1 for s in unique_smiles if s not in train_smiles_set)
    novelty = novel_count / n_unique if n_unique > 0 else 0.0
    
    # Diversity: 1 - mean pairwise Tanimoto similarity
    fps = [get_morgan_fp(m) for m in valid_mols[:max_diversity_sample]]
    n_fps = len(fps)
    if n_fps > 1:
        similarities = []
        for i in range(n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            similarities.extend(sims)
        diversity = 1.0 - (np.mean(similarities) if similarities else 0.0)
    else:
        diversity = 0.0
        
    # Mean SA Score
    sa_scores = [compute_sa_score(m) for m in valid_mols]
    mean_sa = float(np.mean(sa_scores)) if sa_scores else 0.0
    
    return {
        "validity": float(validity),
        "uniqueness": float(uniqueness),
        "novelty": float(novelty),
        "diversity": float(diversity),
        "mean_sa": mean_sa,
        "n_generated": n_gen,
        "n_valid": n_valid,
        "n_unique": n_unique
    }
