import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from features import MORGAN_GEN
from evaluators import PropertyEvaluatorSuite, MODELS_META

# Library of functional substituents frequently observed in UV filters and photoswitch auxochromes
AUXOCHROMES_AND_SUBSTITUENTS = [
    "OC",             # -OCH3 methoxy
    "O",              # -OH hydroxy
    "F",              # -F fluoro
    "Cl",             # -Cl chloro
    "C",              # -CH3 methyl
    "C(C)(C)C",       # -tBu tert-butyl
    "C#N",            # -CN cyano
    "C(F)(F)F",       # -CF3 trifluoromethyl
    "C(=O)OC",        # -COOCH3 ester
    "N(C)C",          # -N(Me)2 dimethylamino
    "OCC(CC)CCCC",    # 2-ethylhexoxy (classic cosmetic lipophilic/barrier tail)
    "CCCC",           # butyl tail
    "c1ccccc1",       # phenyl
    "c1ccc(OC)cc1",   # 4-methoxyphenyl
    "c1ccc(C(C)(C)C)cc1", # 4-tert-butylphenyl
]

def cut_molecule_single_bonds(mol):
    """
    Finds non-ring single bonds between non-hydrogen, non-terminal atoms for crossover.
    """
    bonds_to_cut = []
    for bond in mol.GetBonds():
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.GetAtomicNum() > 1 and a2.GetAtomicNum() > 1:
                if a1.GetDegree() > 1 and a2.GetDegree() > 1:
                    bonds_to_cut.append(bond.GetIdx())
    if not bonds_to_cut:
        return None
    bond_idx = random.choice(bonds_to_cut)
    broken = Chem.FragmentOnBonds(mol, [bond_idx], dummyLabels=[(1, 2)])
    frags = Chem.GetMolFrags(broken, asMols=True)
    if len(frags) == 2:
        return frags
    return None

def recombine_crossover(mol_A, mol_B):
    """
    Performs stochastic single-bond crossover between a fragment from mol_A (Domain A)
    and a fragment from mol_B (Domain B).
    """
    frags_A = cut_molecule_single_bonds(mol_A)
    frags_B = cut_molecule_single_bonds(mol_B)
    if not frags_A or not frags_B:
        return None
    
    fA = random.choice(frags_A)
    fB = random.choice(frags_B)
    
    dummy_A = [a.GetIdx() for a in fA.GetAtoms() if a.GetAtomicNum() == 0]
    dummy_B = [a.GetIdx() for a in fB.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_A or not dummy_B:
        return None
    
    combo = Chem.CombineMols(fA, fB)
    ed = Chem.EditableMol(combo)
    
    shift = fA.GetNumAtoms()
    dA_idx = dummy_A[0]
    dB_idx = dummy_B[0] + shift
    
    neighbor_A = combo.GetAtomWithIdx(dA_idx).GetNeighbors()[0].GetIdx()
    neighbor_B = combo.GetAtomWithIdx(dB_idx).GetNeighbors()[0].GetIdx()
    
    ed.AddBond(neighbor_A, neighbor_B, Chem.BondType.SINGLE)
    ed.RemoveAtom(max(dA_idx, dB_idx))
    ed.RemoveAtom(min(dA_idx, dB_idx))
    
    prod = ed.GetMol()
    try:
        Chem.SanitizeMol(prod)
        return prod
    except Exception:
        return None

def mutate_functional_group(mol):
    """
    Applies random point mutation: attaches a functional group or auxochrome to an aromatic position.
    """
    m = Chem.RWMol(mol)
    c_indices = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIsAromatic() and a.GetTotalNumHs() > 0]
    if not c_indices:
        return None
    target_idx = random.choice(c_indices)
    sub_smi = random.choice(AUXOCHROMES_AND_SUBSTITUENTS)
    sub_mol = Chem.MolFromSmiles(sub_smi)
    if not sub_mol:
        return None
        
    shift = m.GetNumAtoms()
    for a in sub_mol.GetAtoms():
        m.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in sub_mol.GetBonds():
        m.AddBond(b.GetBeginAtomIdx() + shift, b.GetEndAtomIdx() + shift, b.GetBondType())
        
    m.AddBond(target_idx, shift, Chem.BondType.SINGLE)
    try:
        res = m.GetMol()
        Chem.SanitizeMol(res)
        return res
    except Exception:
        return None

class BaselineGeneratorB0:
    """
    Baseline Generative Strategy B0:
    Stochastic fragment recombination (crossover of D_A photoswitch motifs with D_B UV/skin motifs)
    followed by naive sequential filtering (Group A surrogate filter -> Group B surrogate filter).
    """
    def __init__(self, data_dir=os.path.join(BASE_DIR, "data", "processed")):
        print("Initializing Baseline Generator B0...")
        df_A = pd.read_csv(os.path.join(data_dir, "dataset_evaluators_group_A.csv"))
        df_B = pd.read_csv(os.path.join(data_dir, "dataset_evaluators_group_B.csv"))
        
        smiles_A = df_A["canonical_smiles"].dropna().sample(min(3000, len(df_A)), random_state=42).tolist()
        smiles_B = df_B["canonical_smiles"].dropna().sample(min(3000, len(df_B)), random_state=42).tolist()
        
        self.mols_A = [Chem.MolFromSmiles(s) for s in smiles_A]
        self.mols_A = [m for m in self.mols_A if m is not None]
        
        self.mols_B = [Chem.MolFromSmiles(s) for s in smiles_B]
        self.mols_B = [m for m in self.mols_B if m is not None]
        
        print(f"B0 parent pools loaded: {len(self.mols_A)} molecules from D_A, {len(self.mols_B)} molecules from D_B.")

    def generate_candidate(self, p_mut=0.35):
        """
        Generates a single candidate molecule via stochastic recombination and optional mutation.
        """
        mA = random.choice(self.mols_A)
        mB = random.choice(self.mols_B)
        
        prod = recombine_crossover(mA, mB)
        if prod is None:
            # Fallback: mutate parent mA or mB directly
            base_mol = random.choice([mA, mB])
            prod = mutate_functional_group(base_mol)
            if prod is None:
                return None
                
        if random.random() < p_mut:
            mut_prod = mutate_functional_group(prod)
            if mut_prod is not None:
                prod = mut_prod

        # Basic chemical filters
        try:
            Chem.SanitizeMol(prod)
            mw = Descriptors.MolWt(prod)
            if 150.0 <= mw <= 800.0:
                return prod
        except Exception:
            pass
        return None

    def run_generation(self, seed, target_unique_count=1000, max_attempts=15000):
        """
        Executes generation for a specific random seed until target_unique_count valid unique
        molecules are obtained, tracking total generation attempts to measure Validity.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        generated_mols = []
        unique_smiles_set = set()
        
        attempts = 0
        t0 = time.time()
        
        print(f"\n--- Starting Generation B0 (Seed = {seed}, Target = {target_unique_count}) ---")
        while len(unique_smiles_set) < target_unique_count and attempts < max_attempts:
            attempts += 1
            mol = self.generate_candidate()
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi not in unique_smiles_set:
                    unique_smiles_set.add(smi)
                    generated_mols.append(smi)
                    
            if attempts % 2000 == 0:
                print(f"  Attempts: {attempts:,} | Unique valid generated: {len(unique_smiles_set):,} ({len(unique_smiles_set)/attempts*100:.1f}%)")

        elapsed = time.time() - t0
        validity = len(generated_mols) / attempts if attempts > 0 else 0.0
        uniqueness = len(unique_smiles_set) / len(generated_mols) if generated_mols else 0.0
        
        print(f"Generation completed in {elapsed:.2f} s: {len(generated_mols):,} valid molecules ({validity*100:.1f}% validity, {uniqueness*100:.1f}% uniqueness).")
        return generated_mols, attempts, validity, uniqueness

def compute_internal_diversity(smiles_list, sample_limit=1000):
    """
    Computes Internal Diversity according to Section 6:
    Diversity = 1 - 2 / (N * (N - 1)) * sum_{i < j} Tanimoto(x_i, x_j)
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list[:sample_limit]]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0
        
    fps = [MORGAN_GEN.GetFingerprint(m) for m in mols]
    n = len(fps)
    sims = []
    for i in range(n - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        
    mean_tanimoto = float(np.mean(sims))
    return round(1.0 - mean_tanimoto, 4)

def run_experiment_b0(seeds=[42, 101, 2024], n_samples=1000):
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    generator = BaselineGeneratorB0()
    evaluator_suite = PropertyEvaluatorSuite()
    
    all_runs_dfs = []
    metrics_summary = {}

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"  RUNNING BASELINE B0 - RANDOM SEED: {seed}")
        print(f"=======================================================")
        
        smiles_list, attempts, validity, uniqueness = generator.run_generation(
            seed=seed, 
            target_unique_count=n_samples, 
            max_attempts=n_samples * 12
        )
        
        print(f"Evaluating batch of {len(smiles_list)} molecules for seed {seed}...")
        t_eval_0 = time.time()
        df_eval = evaluator_suite.evaluate_batch(smiles_list)
        t_eval_1 = time.time()
        print(f"Evaluation complete in {t_eval_1 - t_eval_0:.2f} s.")
        
        df_eval["method_id"] = "B0"
        df_eval["seed"] = seed
        
        # Section 6 Metrics Calculation
        novelty_rate = float(df_eval["novelty"].mean())
        internal_div = compute_internal_diversity(df_eval["SMILES"].tolist())
        
        # Success rates
        jsr_oracle = float(df_eval["pass_oracle_all"].mean())
        jsr_surrogate = float(df_eval["pass_surrogate_all"].mean())
        pass_constraints_rate = float(df_eval["pass_constraints"].mean())
        
        # Sequential gating attrition rates
        pass_surr_A_rate = float(df_eval["pass_surrogate_A"].mean())
        pass_surr_B_rate = float(df_eval["pass_surrogate_B"].mean())
        pass_orc_A_rate = float(df_eval["pass_oracle_A"].mean())
        pass_orc_B_rate = float(df_eval["pass_oracle_B"].mean())

        # SAScore statistics
        sa_mean = float(df_eval["synthetic_accessibility"].mean())
        sa_std = float(df_eval["synthetic_accessibility"].std())
        sa_pass = float((df_eval["synthetic_accessibility"] <= 5.0).mean())

        # Applicability Domain statistics
        mean_dist_A = float(df_eval["dist_to_D_A"].mean())
        mean_dist_B = float(df_eval["dist_to_D_B"].mean())
        mean_sim_A = float(df_eval["max_sim_D_A"].mean())
        mean_sim_B = float(df_eval["max_sim_D_B"].mean())
        in_ad_A = float((df_eval["dist_to_D_A"] <= 0.70).mean())
        in_ad_B = float((df_eval["dist_to_D_B"] <= 0.70).mean())
        in_both_ad = float(((df_eval["dist_to_D_A"] <= 0.70) & (df_eval["dist_to_D_B"] <= 0.70)).mean())

        run_metrics = {
            "method": "B0",
            "seed": seed,
            "N_generated_attempts": attempts,
            "N_valid": len(smiles_list),
            "Validity": round(validity, 4),
            "Uniqueness": round(uniqueness, 4),
            "Novelty": round(novelty_rate, 4),
            "Internal_Diversity": round(internal_div, 4),
            "JSR_Oracle (Joint Success Rate)": round(jsr_oracle, 4),
            "JSR_Surrogate": round(jsr_surrogate, 4),
            "Overall_Pass_Constraints": round(pass_constraints_rate, 4),
            "Pass_Surrogate_A": round(pass_surr_A_rate, 4),
            "Pass_Surrogate_B": round(pass_surr_B_rate, 4),
            "Pass_Oracle_A": round(pass_orc_A_rate, 4),
            "Pass_Oracle_B": round(pass_orc_B_rate, 4),
            "SAScore_Mean": round(sa_mean, 4),
            "SAScore_Std": round(sa_std, 4),
            "SAScore_Pass_Rate (<=5.0)": round(sa_pass, 4),
            "AD_Mean_Dist_to_D_A": round(mean_dist_A, 4),
            "AD_Mean_Dist_to_D_B": round(mean_dist_B, 4),
            "AD_Mean_MaxSim_D_A": round(mean_sim_A, 4),
            "AD_Mean_MaxSim_D_B": round(mean_sim_B, 4),
            "AD_Fraction_in_D_A (dist<=0.70)": round(in_ad_A, 4),
            "AD_Fraction_in_D_B (dist<=0.70)": round(in_ad_B, 4),
            "AD_Fraction_in_Both_AD": round(in_both_ad, 4),
        }
        
        metrics_summary[f"seed_{seed}"] = run_metrics
        all_runs_dfs.append(df_eval)
        
        print(f"\n--- Metrics for Seed {seed} ---")
        for k, v in run_metrics.items():
            print(f"  {k}: {v}")

    # Combine all runs into a single unified generated.csv
    combined_df = pd.concat(all_runs_dfs, ignore_index=True)
    
    # Save primary artifact: results/generated_B0.csv
    out_csv_path = os.path.join(BASE_DIR, "results", "generated_B0.csv")
    combined_df.to_csv(out_csv_path, index=False)
    print(f"\nSaved {len(combined_df):,} generated molecules to: {out_csv_path}")

    # Aggregated metrics across all seeds (mean +/- std)
    metric_keys = [k for k in list(metrics_summary.values())[0].keys() if isinstance(list(metrics_summary.values())[0][k], (int, float))]
    aggregated = {"method": "B0_aggregated", "n_seeds": len(seeds), "total_candidates": len(combined_df)}
    for k in metric_keys:
        vals = [metrics_summary[f"seed_{s}"][k] for s in seeds]
        aggregated[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        aggregated[f"{k}_std"] = round(float(np.std(vals)), 4)
    metrics_summary["aggregated"] = aggregated

    # Save metrics JSON & CSV
    out_json_path = os.path.join(BASE_DIR, "results", "metrics_B0.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics summary to: {out_json_path}")
    
    # Also save tabular summary
    summary_df = pd.DataFrame([metrics_summary[f"seed_{s}"] for s in seeds] + [aggregated])
    out_summary_csv = os.path.join(BASE_DIR, "results", "metrics_summary_B0.csv")
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"Saved summary table to: {out_summary_csv}")

    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Generative Strategy B0")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of unique candidates per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024], help="Random seeds")
    args = parser.parse_args()
    
    run_experiment_b0(seeds=args.seeds, n_samples=args.n_samples)
