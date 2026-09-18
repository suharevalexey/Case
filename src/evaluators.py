import os
import sys
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig, DataStructs
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from features import mol_to_features, MORGAN_GEN
from applicability_domain import ApplicabilityDomainManager
from evaluator_model import EnsembleEvaluatorWrapper

MODELS_DIR = os.path.join(BASE_DIR, "models", "evaluators")
ORACLE_DIR = os.path.join(BASE_DIR, "models", "oracle")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Model registry with explicit Group A and Group B designations and target constraint boundaries
MODELS_META = {
    "group_A_absorption_max_nm": {
        "group": "A",
        "target": "absorption_max_nm",
        "type": "regression",
        "unit": "nm",
        "constraint_min": 290.0,
        "constraint_max": 420.0,
    },
    "group_A_log_extinction": {
        "group": "A",
        "target": "log_extinction",
        "type": "regression",
        "unit": "log10(M^-1 cm^-1)",
        "constraint_min": 3.80,
        "constraint_max": float("inf"),
    },
    "group_A_quantum_yield": {
        "group": "A",
        "target": "quantum_yield",
        "type": "regression",
        "unit": "quantum_yield",
        "constraint_min": 0.25,
        "constraint_max": float("inf"),
    },
    "group_A_log_half_life": {
        "group": "A",
        "target": "log_half_life",
        "type": "regression",
        "unit": "log10(s)",
        "constraint_min": 3.56,
        "constraint_max": float("inf"),
    },
    "group_B_log_kp": {
        "group": "B",
        "target": "log_kp",
        "type": "regression",
        "unit": "log10(cm/s)",
        "constraint_min": float("-inf"),
        "constraint_max": -5.00,
    },
    "group_B_skin_sensitization": {
        "group": "B",
        "target": "skin_sensitization",
        "type": "classification",
        "unit": "probability_sensitizer",
        "constraint_min": float("-inf"),
        "constraint_max": 0.50,
    },
    "group_B_skin_irritation": {
        "group": "B",
        "target": "skin_irritation",
        "type": "classification",
        "unit": "probability_irritant",
        "constraint_min": float("-inf"),
        "constraint_max": 0.50,
    },
}

def _ensure_file_from_github(local_path, rel_github_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    raw_url = f"https://raw.githubusercontent.com/suharevalexey/Case/main/{rel_github_path.replace(os.sep, '/')}"
    print(f"[GitHub Sync] Downloading: {rel_github_path}...")
    import urllib.request
    req = urllib.request.Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"[GitHub Sync] Ready: {rel_github_path} ({os.path.getsize(local_path):,} bytes)")
    return local_path

class PropertyEvaluatorSuite:
    """
    Unified evaluator suite for all 6 properties across Group A and Group B.
    Loads Guidance Surrogate models (with epistemic uncertainty) and Independent Test Oracles.
    Also computes distances to training domains D_A and D_B (Applicability Domain) and SAScore.
    """
    def __init__(self, base_dir=None):
        print("Loading Property Evaluator Suite...")
        if base_dir is None:
            if os.path.exists("/content/Case"):
                base_dir = "/content/Case"
            elif os.path.exists("models/evaluators"):
                base_dir = os.path.abspath(".")
            elif os.path.exists("Case/models/evaluators"):
                base_dir = os.path.abspath("Case")
            elif os.path.exists("/content"):
                base_dir = "/content/Case"
            else:
                base_dir = BASE_DIR

        models_dir = os.path.join(base_dir, "models", "evaluators")
        oracle_dir = os.path.join(base_dir, "models", "oracle")
        data_dir = os.path.join(base_dir, "data", "processed")

        self.surrogates = {}
        self.oracles = {}
        
        for key in MODELS_META.keys():
            surr_path = os.path.join(models_dir, f"{key}_evaluator.joblib")
            orc_path = os.path.join(oracle_dir, f"{key}_oracle.joblib")
            
            if not (os.path.exists(surr_path) and os.path.getsize(surr_path) > 0):
                legacy_surr = os.path.join(models_dir, f"{key.replace('quantum_yield', 'photochem_efficiency')}_evaluator.joblib")
                if os.path.exists(legacy_surr) and os.path.getsize(legacy_surr) > 0:
                    surr_path = legacy_surr
                else:
                    _ensure_file_from_github(surr_path, f"models/evaluators/{key}_evaluator.joblib")
            self.surrogates[key] = joblib.load(surr_path)
                
            if not (os.path.exists(orc_path) and os.path.getsize(orc_path) > 0):
                legacy_orc = os.path.join(oracle_dir, f"{key.replace('quantum_yield', 'photochem_efficiency')}_oracle.joblib")
                if os.path.exists(legacy_orc) and os.path.getsize(legacy_orc) > 0:
                    orc_path = legacy_orc
                else:
                    _ensure_file_from_github(orc_path, f"models/oracle/{key}_oracle.joblib")
            self.oracles[key] = joblib.load(orc_path)
                
        print(f"Successfully loaded {len(self.surrogates)} Group A/B surrogate models and {len(self.oracles)} oracle models.")

        # Load reference datasets for Applicability Domain & Novelty
        path_A = os.path.join(data_dir, "dataset_evaluators_group_A.csv")
        path_B = os.path.join(data_dir, "dataset_evaluators_group_B.csv")
        if not (os.path.exists(path_A) and os.path.getsize(path_A) > 0):
            _ensure_file_from_github(path_A, "data/processed/dataset_evaluators_group_A.csv")
        if not (os.path.exists(path_B) and os.path.getsize(path_B) > 0):
            _ensure_file_from_github(path_B, "data/processed/dataset_evaluators_group_B.csv")

        df_A = pd.read_csv(path_A)
        df_B = pd.read_csv(path_B)
        
        smiles_A_series = df_A["canonical_smiles"].dropna()
        smiles_B_series = df_B["canonical_smiles"].dropna()
        
        self.known_smiles_set = set(smiles_A_series).union(set(smiles_B_series))
        print(f"Reference database size: {len(self.known_smiles_set):,} unique training molecules.")

        self.ad_manager = ApplicabilityDomainManager(
            smiles_A=smiles_A_series.sample(min(5000, len(smiles_A_series)), random_state=42),
            smiles_B=smiles_B_series
        )

    def evaluate_batch(self, smiles_list):
        """
        Evaluates a batch of SMILES strings with high vectorized efficiency.
        Returns a DataFrame containing all required metrics for generated.csv:
          - Canonical SMILES
          - Novelty flag
          - Applicability domain distances (dist_to_D_A, dist_to_D_B, max_sim_D_A, max_sim_D_B)
          - Synthetic Accessibility (SAScore)
          - Surrogate predictions and uncertainties for Group A and Group B
          - Independent Oracle predictions for Group A and Group B
          - Pass flags for Omega_A, Omega_B, and Oracle joint success
        """
        valid_items = []
        for s in smiles_list:
            if not isinstance(s, str) or not s.strip():
                continue
            mol = Chem.MolFromSmiles(s.strip())
            if mol is not None:
                valid_items.append((s, mol))

        if not valid_items:
            return pd.DataFrame()

        canon_smiles = [Chem.MolToSmiles(m) for s, m in valid_items]
        valid_mols = [m for s, m in valid_items]

        # 1. Novelty check vs D_A U D_B
        novelty = [int(s not in self.known_smiles_set) for s in canon_smiles]

        # 2. Synthetic Accessibility (Ertl SAScore)
        sa_scores = []
        for m in valid_mols:
            try:
                sa = float(sascorer.calculateScore(m))
            except Exception:
                sa = 10.0
            sa_scores.append(round(sa, 4))

        # 3. Applicability Domain Distances
        fps = [MORGAN_GEN.GetFingerprint(m) for m in valid_mols]
        sims_A = [max(DataStructs.BulkTanimotoSimilarity(fp, self.ad_manager.fps_A)) if self.ad_manager.fps_A else 0.0 for fp in fps]
        sims_B = [max(DataStructs.BulkTanimotoSimilarity(fp, self.ad_manager.fps_B)) if self.ad_manager.fps_B else 0.0 for fp in fps]
        dists_A = [round(1.0 - s, 4) for s in sims_A]
        dists_B = [round(1.0 - s, 4) for s in sims_B]

        # 4. Feature Extraction
        feats = [mol_to_features(m) for m in valid_mols]
        X = np.array(feats)

        res_df = pd.DataFrame({
            "SMILES": canon_smiles,
            "novelty": novelty,
            "dist_to_D_A": dists_A,
            "dist_to_D_B": dists_B,
            "max_sim_D_A": [round(s, 4) for s in sims_A],
            "max_sim_D_B": [round(s, 4) for s in sims_B],
            "synthetic_accessibility": sa_scores,
        })

        # 5. Surrogate Predictions & Uncertainties
        pass_surr_A = np.ones(len(valid_mols), dtype=bool)
        pass_surr_B = np.ones(len(valid_mols), dtype=bool)

        for key, meta in MODELS_META.items():
            surr = self.surrogates[key]
            pred, std = surr.predict_with_uncertainty(X)
            res_df[f"pred_{key}"] = np.round(pred, 4)
            res_df[f"unc_{key}"] = np.round(std, 4)
            c_pass = (pred >= meta["constraint_min"]) & (pred <= meta["constraint_max"])
            if meta["group"] == "A":
                pass_surr_A = pass_surr_A & c_pass
            else:
                pass_surr_B = pass_surr_B & c_pass

        res_df["pass_surrogate_A"] = pass_surr_A.astype(int)
        res_df["pass_surrogate_B"] = pass_surr_B.astype(int)
        res_df["pass_surrogate_all"] = (pass_surr_A & pass_surr_B).astype(int)

        # 6. Independent Oracle Predictions
        pass_orc_A = np.ones(len(valid_mols), dtype=bool)
        pass_orc_B = np.ones(len(valid_mols), dtype=bool)

        for key, meta in MODELS_META.items():
            orc = self.oracles[key]
            if hasattr(orc, "predict_proba"):
                o_val = orc.predict_proba(X)[:, 1]
            else:
                o_val = orc.predict(X)
            res_df[f"oracle_{key}"] = np.round(o_val, 4)
            c_pass = (o_val >= meta["constraint_min"]) & (o_val <= meta["constraint_max"])
            if meta["group"] == "A":
                pass_orc_A = pass_orc_A & c_pass
            else:
                pass_orc_B = pass_orc_B & c_pass

        res_df["pass_oracle_A"] = pass_orc_A.astype(int)
        res_df["pass_oracle_B"] = pass_orc_B.astype(int)
        res_df["pass_oracle_all"] = (pass_orc_A & pass_orc_B).astype(int)

        # 7. Final overall constraint pass
        res_df["pass_constraints"] = ((res_df["pass_oracle_all"] == 1) & (res_df["synthetic_accessibility"] <= 5.0)).astype(int)

        # Backwards-compatible aliases for legacy property names
        if "pred_group_A_quantum_yield" in res_df.columns:
            res_df["pred_group_A_photochem_efficiency"] = res_df["pred_group_A_quantum_yield"]
            res_df["unc_group_A_photochem_efficiency"] = res_df["unc_group_A_quantum_yield"]
            res_df["oracle_group_A_photochem_efficiency"] = res_df["oracle_group_A_quantum_yield"]

        return res_df

    def evaluate_molecule(self, smiles):
        df = self.evaluate_batch([smiles])
        if len(df) == 0:
            return None
        return df.iloc[0].to_dict()

if __name__ == "__main__":
    suite = PropertyEvaluatorSuite()
    avobenzone = "CC(C)(C)c1ccc(C(=O)CC(=O)c2ccc(OC)cc2)cc1"
    res = suite.evaluate_molecule(avobenzone)
    print("\n--- Test Evaluation for Avobenzone ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
