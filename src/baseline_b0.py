import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.config import BASELINE_GEN_COUNT, RANDOM_SEED, RESULTS_DIR, PROCESSED_DATA_DIR
from src.metrics import canonicalize_smiles, evaluate_generation_metrics
from src.evaluators import PropertyEvaluatorSuite

# Seed building blocks and scaffolds for Variant A (Hybrid MOST + UV absorbing molecules)
SEED_SCAFFOLDS = [
    # Norbornadiene (NBD) cores
    "C1=CC2C=CC1C2",
    "N#CC1=CC2C=CC1C2",
    "N#CC1=C(C#N)C2C=CC1C2",
    "COC(=O)C1=CC2C=CC1C2",
    "c1ccccc1C1=CC2C=CC1C2",
    "CC1=CC2C=CC1C2",
    "c1cc(sc1)C1=CC2C=CC1C2",
    # Quadricyclane (QC) cores
    "C1C2C3C1C4C2C34",
    "N#CC1C2C3C1C4(C#N)C2C34",
    # Photoswitch azo cores
    "c1ccc(N=Nc2ccccc2)cc1",
    "c1ccc(N=Nc2ccc(O)cc2)cc1",
    "c1ccc(N=Nc2ccc(N(C)C)cc2)cc1",
    # UV filter cores (for recombination/mutation)
    "O=C(c1ccccc1)c1ccccc1",
    "O=C(O)/C=C/c1ccccc1",
    "O=C(O)c1ccccc1O",
    "O=C(c1ccccc1)CC(=O)c1ccccc1"
]

# Common functional groups / auxochromes to attach
SUBSTITUENTS = [
    "[H]", "C", "CC", "C(C)(C)C", "OC", "OCC", "O", 
    "C#N", "N(=O)=O", "N(C)C", "C(=O)OC", "C(=O)OCC(CC)CCCC", 
    "c1ccccc1", "c1ccc(OC)cc1", "c1ccc(N(C)C)cc1", "c1cc(sc1)", 
    "F", "Cl", "C(F)(F)F", "S(=O)(=O)O", "C(=O)c1ccccc1"
]

def mutate_molecule(smiles: str, rng: random.Random) -> str:
    """Apply a simple stochastic chemical mutation to a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
        
    rwmol = Chem.RWMol(mol)
    mutation_type = rng.choice(["add_substituent", "change_bond", "substitute_atom", "recombine"])
    
    try:
        if mutation_type == "add_substituent":
            candidate_atoms = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetTotalNumHs() > 0]
            if candidate_atoms:
                target_idx = rng.choice(candidate_atoms)
                sub_smi = rng.choice(SUBSTITUENTS)
                if sub_smi != "[H]":
                    sub_mol = Chem.MolFromSmiles(sub_smi)
                    if sub_mol is not None:
                        combo = Chem.CombineMols(rwmol, sub_mol)
                        rwcombo = Chem.RWMol(combo)
                        sub_start_idx = rwmol.GetNumAtoms()
                        rwcombo.AddBond(target_idx, sub_start_idx, Chem.BondType.SINGLE)
                        mut_mol = rwcombo.GetMol()
                        can = canonicalize_smiles(Chem.MolToSmiles(mut_mol))
                        if can is not None:
                            return can
                            
        elif mutation_type == "substitute_atom":
            c_atoms = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetDegree() <= 2]
            if c_atoms:
                idx = rng.choice(c_atoms)
                new_elem = rng.choice([7, 8, 16])
                rwmol.GetAtomWithIdx(idx).SetAtomicNum(new_elem)
                can = canonicalize_smiles(Chem.MolToSmiles(rwmol.GetMol()))
                if can is not None:
                    return can
                    
        elif mutation_type == "recombine":
            other_seed = rng.choice(SEED_SCAFFOLDS)
            mol2 = Chem.MolFromSmiles(other_seed)
            if mol2 is not None:
                cands1 = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetTotalNumHs() > 0]
                cands2 = [a.GetIdx() for a in mol2.GetAtoms() if a.GetTotalNumHs() > 0]
                if cands1 and cands2:
                    idx1 = rng.choice(cands1)
                    idx2 = rng.choice(cands2)
                    combo = Chem.CombineMols(rwmol, mol2)
                    rwcombo = Chem.RWMol(combo)
                    rwcombo.AddBond(idx1, rwmol.GetNumAtoms() + idx2, Chem.BondType.SINGLE)
                    can = canonicalize_smiles(Chem.MolToSmiles(rwcombo.GetMol()))
                    if can is not None:
                        return can
    except Exception:
        pass
        
    return smiles

class BaselineGeneratorB0:
    """
    Baseline Strategy B0: Stochastic molecular mutator with naive rejection sampling.
    Represents an unguided generative baseline to benchmark joint multi-objective optimization.
    """
    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
    def generate(self, n_target: int = BASELINE_GEN_COUNT) -> List[str]:
        """Generate at least n_target unique valid SMILES."""
        print(f"[B0 Baseline] Starting stochastic generation of {n_target} molecules...")
        unique_generated = set()
        pool = list(SEED_SCAFFOLDS)
        
        attempts = 0
        max_attempts = n_target * 15
        
        while len(unique_generated) < n_target and attempts < max_attempts:
            attempts += 1
            parent = self.rng.choice(pool)
            mutated = mutate_molecule(parent, self.rng)
            can = canonicalize_smiles(mutated)
            
            if can is not None and can not in unique_generated:
                mol = Chem.MolFromSmiles(can)
                if mol is not None and 7 <= mol.GetNumHeavyAtoms() <= 50:
                    unique_generated.add(can)
                    if len(pool) < 200:
                        pool.append(can)
                        
            if len(unique_generated) % 250 == 0 and len(unique_generated) > 0:
                print(f"  [B0 Baseline] Generated {len(unique_generated)} / {n_target} molecules...")
                
        result = list(unique_generated)[:n_target]
        print(f"[B0 Baseline] Generation complete: {len(result)} unique valid molecules generated.")
        return result

def run_baseline_b0_pipeline(evaluator_suite: PropertyEvaluatorSuite) -> pd.DataFrame:
    """Run baseline B0 generation, evaluate 8 criteria + AD, and save generated_b0.csv."""
    b0 = BaselineGeneratorB0(seed=RANDOM_SEED)
    gen_smiles = b0.generate(BASELINE_GEN_COUNT)
    
    # Load training sets to compute novelty
    train_a = pd.read_csv(PROCESSED_DATA_DIR / "train_A.csv")
    train_b = pd.read_csv(PROCESSED_DATA_DIR / "train_B.csv")
    train_smiles_set = set(train_a["canonical_smiles"]).union(set(train_b["canonical_smiles"]))
    
    # Standard benchmark generation metrics
    metrics = evaluate_generation_metrics(gen_smiles, train_smiles_set)
    print("\n--- B0 Baseline Generation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
    # Evaluate candidates with 8-criteria EvaluatorSuite
    print("\nEvaluating candidates across 4 MOST + 4 UV/Skin criteria...")
    df_eval = evaluator_suite.evaluate_candidates(gen_smiles)
    
    # Add method ID and novelty column
    df_eval["method_id"] = "B0_baseline"
    df_eval["novelty"] = df_eval["SMILES"].apply(lambda s: bool(s not in train_smiles_set))
    
    # Reorder columns matching Section 10 specification exactly:
    cols_order = [
        "SMILES", "method_id",
        # Group A properties (MOST)
        "pred_A_wavelength", "pred_A_delta_h", "pred_A_pss", "pred_A_log_k", "synthetic_accessibility",
        # Group B properties (UV / Skin Safety)
        "pred_B_wavelength", "pred_B_log_eps", "potts_guy_log_kp", "molecular_weight", "log_p",
        # Reliability and Distance
        "uncertainty_AD", "novelty", "dist_to_DA", "dist_to_DB",
        # Flags
        "pass_group_A", "pass_group_B", "pass_ad", "pass_constraints"
    ]
    
    df_final = df_eval[cols_order].copy()
    
    # Calculate Joint Success Rate (JSR)
    jsr = df_final["pass_constraints"].mean()
    pass_a = df_final["pass_group_A"].mean()
    pass_b = df_final["pass_group_B"].mean()
    pass_ad = df_final["pass_ad"].mean()
    
    print("\n>>> B0 Baseline Multi-Criteria Results <<<")
    print(f"  Pass Group A (all 4 MOST criteria):        {pass_a * 100:.2f}%")
    print(f"  Pass Group B (all 4 UV/Skin criteria):     {pass_b * 100:.2f}%")
    print(f"  Within Applicability Domain (reliable AD): {pass_ad * 100:.2f}%")
    print(f"  Joint Success Rate (JSR - ALL criteria):   {jsr * 100:.2f}%\n")
    
    out_file = RESULTS_DIR / "generated_b0.csv"
    df_final.to_csv(out_file, index=False)
    print(f"Generated B0 candidates saved to {out_file}")
    
    return df_final

if __name__ == "__main__":
    suite = PropertyEvaluatorSuite()
    suite.load_models()
    run_baseline_b0_pipeline(suite)
