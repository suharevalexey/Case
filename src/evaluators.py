import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, CRITERIA
from src.metrics import canonicalize_smiles, get_morgan_fp, compute_sa_score, compute_potts_guy_log_kp

def extract_molecular_features(mol: Chem.Mol, n_bits: int = 1024) -> np.ndarray:
    """Extract combined ECFP4 fingerprint + 2D physicochemical descriptors."""
    # 1. ECFP4 fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    fp_arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    
    # 2. Key 2D physicochemical descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    fraction_sp3 = Descriptors.FractionCSP3(mol)
    
    phys_desc = np.array([mw, logp, tpsa, rotb, hbd, hba, aromatic_rings, fraction_sp3], dtype=np.float32)
    return np.concatenate([fp_arr, phys_desc])

def featurize_smiles_list(smiles_list: List[str], n_bits: int = 1024) -> Tuple[np.ndarray, List[int]]:
    """Featurize a list of SMILES, returning feature matrix and valid indices."""
    feats = []
    valid_idx = []
    for idx, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            f = extract_molecular_features(mol, n_bits=n_bits)
            feats.append(f)
            valid_idx.append(idx)
    return np.array(feats, dtype=np.float32), valid_idx

class PropertyEvaluatorSuite:
    """
    Main Evaluator Suite containing:
    1. Surrogate Evaluator A (MOST: lambda_max, delta_h)
    2. Surrogate Evaluator B (UV / Skin: lambda_max, log_kp)
    3. Independent Oracle Evaluators (for unbiased post-eval test verification)
    4. Applicability Domain (AD) Distance Evaluator
    """
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Surrogate reward models
        self.model_A_wl = None
        self.model_A_dh = None
        self.model_B_wl = None
        self.model_B_logkp = None
        
        # Independent test oracles (distinct architecture / random state)
        self.oracle_A_wl = None
        self.oracle_B_wl = None
        
        # Reference fingerprints for Applicability Domain
        self.train_fps_A = []
        self.train_fps_B = []
        
    def fit_and_evaluate(self):
        """Train both internal surrogates and independent test oracles on scaffold splits."""
        train_a = pd.read_csv(PROCESSED_DATA_DIR / "train_A.csv")
        val_a = pd.read_csv(PROCESSED_DATA_DIR / "val_A.csv")
        test_a = pd.read_csv(PROCESSED_DATA_DIR / "test_A.csv")
        
        train_b = pd.read_csv(PROCESSED_DATA_DIR / "train_B.csv")
        val_b = pd.read_csv(PROCESSED_DATA_DIR / "val_B.csv")
        test_b = pd.read_csv(PROCESSED_DATA_DIR / "test_B.csv")
        
        # Store training Morgan fingerprints for AD
        self.train_fps_A = [get_morgan_fp(Chem.MolFromSmiles(s)) for s in train_a["canonical_smiles"]]
        self.train_fps_B = [get_morgan_fp(Chem.MolFromSmiles(s)) for s in train_b["canonical_smiles"]]
        
        # --- Featurize D_A ---
        X_train_A, idx_tr_A = featurize_smiles_list(train_a["canonical_smiles"].tolist())
        y_train_A_wl = train_a["lambda_max"].iloc[idx_tr_A].values
        y_train_A_dh = train_a["delta_h"].iloc[idx_tr_A].values
        
        X_val_A, idx_v_A = featurize_smiles_list(val_a["canonical_smiles"].tolist())
        y_val_A_wl = val_a["lambda_max"].iloc[idx_v_A].values
        y_val_A_dh = val_a["delta_h"].iloc[idx_v_A].values
        
        X_test_A, idx_te_A = featurize_smiles_list(test_a["canonical_smiles"].tolist())
        y_test_A_wl = test_a["lambda_max"].iloc[idx_te_A].values
        y_test_A_dh = test_a["delta_h"].iloc[idx_te_A].values
        
        # Train Surrogate A models (XGBoost)
        print("Training Surrogate Model A (MOST)...")
        self.model_A_wl = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_A_wl.fit(X_train_A, y_train_A_wl)
        
        self.model_A_dh = XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_A_dh.fit(X_train_A, y_train_A_dh)
        
        # Train Independent Oracle A (Random Forest)
        print("Training Independent Test Oracle A...")
        self.oracle_A_wl = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_SEED + 99)
        self.oracle_A_wl.fit(X_train_A, y_train_A_wl)
        
        # Evaluate Model A
        pred_test_A_wl = self.model_A_wl.predict(X_test_A)
        pred_test_A_dh = self.model_A_dh.predict(X_test_A)
        oracle_pred_A_wl = self.oracle_A_wl.predict(X_test_A)
        
        print(f"  Surrogate A (lambda_max) Test MAE: {mean_absolute_error(y_test_A_wl, pred_test_A_wl):.2f} nm, R2: {r2_score(y_test_A_wl, pred_test_A_wl):.3f}")
        print(f"  Surrogate A (delta_h)    Test MAE: {mean_absolute_error(y_test_A_dh, pred_test_A_dh):.2f} kJ/mol, R2: {r2_score(y_test_A_dh, pred_test_A_dh):.3f}")
        print(f"  Oracle A    (lambda_max) Test MAE: {mean_absolute_error(y_test_A_wl, oracle_pred_A_wl):.2f} nm, R2: {r2_score(y_test_A_wl, oracle_pred_A_wl):.3f}")
        
        # --- Featurize D_B ---
        X_train_B, idx_tr_B = featurize_smiles_list(train_b["canonical_smiles"].tolist())
        y_train_B_wl = train_b["lambda_max"].iloc[idx_tr_B].values
        y_train_B_logkp = train_b["log_kp"].iloc[idx_tr_B].values
        
        X_test_B, idx_te_B = featurize_smiles_list(test_b["canonical_smiles"].tolist())
        y_test_B_wl = test_b["lambda_max"].iloc[idx_te_B].values
        y_test_B_logkp = test_b["log_kp"].iloc[idx_te_B].values
        
        # Train Surrogate B models (XGBoost)
        print("Training Surrogate Model B (UV/Skin)...")
        self.model_B_wl = XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_B_wl.fit(X_train_B, y_train_B_wl)
        
        self.model_B_logkp = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_B_logkp.fit(X_train_B, y_train_B_logkp)
        
        # Train Independent Oracle B (Random Forest)
        print("Training Independent Test Oracle B...")
        self.oracle_B_wl = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=RANDOM_SEED + 99)
        self.oracle_B_wl.fit(X_train_B, y_train_B_wl)
        
        # Evaluate Model B
        pred_test_B_wl = self.model_B_wl.predict(X_test_B)
        oracle_pred_B_wl = self.oracle_B_wl.predict(X_test_B)
        print(f"  Surrogate B (UV lambda_max) Test MAE: {mean_absolute_error(y_test_B_wl, pred_test_B_wl):.2f} nm, R2: {r2_score(y_test_B_wl, pred_test_B_wl):.3f}")
        print(f"  Oracle B    (UV lambda_max) Test MAE: {mean_absolute_error(y_test_B_wl, oracle_pred_B_wl):.2f} nm, R2: {r2_score(y_test_B_wl, oracle_pred_B_wl):.3f}")
        
        # Save all models
        joblib.dump(self.model_A_wl, self.models_dir / "surrogate_A_wl.joblib")
        joblib.dump(self.model_A_dh, self.models_dir / "surrogate_A_dh.joblib")
        joblib.dump(self.oracle_A_wl, self.models_dir / "oracle_A_wl.joblib")
        joblib.dump(self.model_B_wl, self.models_dir / "surrogate_B_wl.joblib")
        joblib.dump(self.model_B_logkp, self.models_dir / "surrogate_B_logkp.joblib")
        joblib.dump(self.oracle_B_wl, self.models_dir / "oracle_B_wl.joblib")
        joblib.dump({"fps_A": self.train_fps_A, "fps_B": self.train_fps_B}, self.models_dir / "ad_fingerprints.joblib")
        print("All evaluators and oracles successfully saved to disk.")

    def load_models(self):
        """Load trained models from disk."""
        self.model_A_wl = joblib.load(self.models_dir / "surrogate_A_wl.joblib")
        self.model_A_dh = joblib.load(self.models_dir / "surrogate_A_dh.joblib")
        self.oracle_A_wl = joblib.load(self.models_dir / "oracle_A_wl.joblib")
        self.model_B_wl = joblib.load(self.models_dir / "surrogate_B_wl.joblib")
        self.model_B_logkp = joblib.load(self.models_dir / "surrogate_B_logkp.joblib")
        self.oracle_B_wl = joblib.load(self.models_dir / "oracle_B_wl.joblib")
        fps_dict = joblib.load(self.models_dir / "ad_fingerprints.joblib")
        self.train_fps_A = fps_dict["fps_A"]
        self.train_fps_B = fps_dict["fps_B"]

    def evaluate_candidates(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Evaluate candidate molecules across all endpoints, Applicability Domain distances,
        SA score, and joint constraint satisfaction.
        """
        if self.model_A_wl is None:
            self.load_models()
            
        rows = []
        for s in smiles_list:
            can_smi = canonicalize_smiles(s)
            if can_smi is None:
                continue
            mol = Chem.MolFromSmiles(can_smi)
            if mol is None:
                continue
                
            feats = extract_molecular_features(mol).reshape(1, -1)
            
            # Predict properties
            p_A_wl = float(self.model_A_wl.predict(feats)[0])
            p_A_dh = float(self.model_A_dh.predict(feats)[0])
            p_B_wl = float(self.model_B_wl.predict(feats)[0])
            p_B_logkp = float(self.model_B_logkp.predict(feats)[0])
            
            # Potts-Guy empirical log Kp and SA score
            calc_log_kp = compute_potts_guy_log_kp(mol)
            sa_val = compute_sa_score(mol)
            
            # Applicability Domain distances
            fp = get_morgan_fp(mol)
            sims_A = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_A)
            dist_A = float(1.0 - max(sims_A)) if sims_A else 1.0
            
            sims_B = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_B)
            dist_B = float(1.0 - max(sims_B)) if sims_B else 1.0
            
            # Combined uncertainty / AD distance
            uncertainty_ad = (dist_A + dist_B) / 2.0
            
            # Joint Constraint Satisfaction:
            # 1. MOST: lambda_max in [310, 420] nm, Delta H >= 40.0 kJ/mol
            pass_most = (CRITERIA["most_wavelength_min"] <= p_A_wl <= CRITERIA["most_wavelength_max"]) and (p_A_dh >= CRITERIA["most_delta_h_min"])
            # 2. UV / Skin: lambda_max >= 330 nm, log Kp <= -2.5, SA <= 4.5
            pass_uv_skin = (p_B_wl >= CRITERIA["uv_wavelength_min"]) and (p_B_logkp <= CRITERIA["log_kp_max"]) and (sa_val <= CRITERIA["sa_score_max"])
            # 3. Reliability: within AD (not extrapolated)
            pass_ad = (dist_A <= CRITERIA["ad_distance_max"]) or (dist_B <= CRITERIA["ad_distance_max"])
            
            pass_all = pass_most and pass_uv_skin and pass_ad
            
            rows.append({
                "SMILES": can_smi,
                "pred_A_wavelength": round(p_A_wl, 2),
                "pred_A_delta_h": round(p_A_dh, 2),
                "pred_B_wavelength": round(p_B_wl, 2),
                "pred_B_log_kp": round(p_B_logkp, 2),
                "potts_guy_log_kp": round(calc_log_kp, 2),
                "synthetic_accessibility": round(sa_val, 2),
                "dist_to_DA": round(dist_A, 3),
                "dist_to_DB": round(dist_B, 3),
                "uncertainty_AD": round(uncertainty_ad, 3),
                "pass_constraints": bool(pass_all)
            })
            
        return pd.DataFrame(rows)

if __name__ == "__main__":
    suite = PropertyEvaluatorSuite()
    suite.fit_and_evaluate()
