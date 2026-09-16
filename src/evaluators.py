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
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, CRITERIA
from src.metrics import canonicalize_smiles, get_morgan_fp, compute_sa_score, compute_potts_guy_log_kp

def extract_molecular_features(mol: Chem.Mol, n_bits: int = 1024) -> np.ndarray:
    """Extract combined ECFP4 fingerprint + 2D physicochemical descriptors."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    fp_arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    
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
    Evaluator Suite implementing 4 criteria for Group A and 4 criteria for Group B:
    Group A (MOST):
      1. Wavelength lambda_max (nm)
      2. Photostationary State PSS (%)
      3. Thermal rate log10(k_thermal / s^-1)
      4. Synthetic Accessibility SA Score
    Group B (UV / Skin / Safety):
      1. UV absorption lambda_max (nm)
      2. Molar extinction coefficient log10(eps / M^-1 cm^-1)
      3. Skin permeability log Kp (cm/h)
      4. Film barrier size & lipophilicity (MW in [250, 600], logP in [1.5, 5.5])
    """
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Group A Surrogate models
        self.model_A_wl = None
        self.model_A_dh = None
        self.model_A_pss = None
        self.model_A_k = None
        
        # Group B Surrogate models
        self.model_B_wl = None
        self.model_B_eps = None
        
        # Independent Test Oracles (Section 7)
        self.oracle_A_wl = None
        self.oracle_B_wl = None
        
        # Applicability Domain reference fingerprints
        self.train_fps_A = []
        self.train_fps_B = []
        
    def fit_and_evaluate(self):
        """Train models strictly on real training splits."""
        train_a = pd.read_csv(PROCESSED_DATA_DIR / "train_A.csv")
        test_a = pd.read_csv(PROCESSED_DATA_DIR / "test_A.csv")
        
        train_b = pd.read_csv(PROCESSED_DATA_DIR / "train_B.csv")
        test_b = pd.read_csv(PROCESSED_DATA_DIR / "test_B.csv")
        
        # Store training fingerprints for AD
        self.train_fps_A = [get_morgan_fp(Chem.MolFromSmiles(s)) for s in train_a["canonical_smiles"]]
        self.train_fps_B = [get_morgan_fp(Chem.MolFromSmiles(s)) for s in train_b["canonical_smiles"]]
        
        # --- 1. Train Group A Evaluators (MOST) ---
        print("Training Evaluators for Group A (MOST)...")
        X_train_A, idx_tr_A = featurize_smiles_list(train_a["canonical_smiles"].tolist())
        X_test_A, idx_te_A = featurize_smiles_list(test_a["canonical_smiles"].tolist())
        
        # Criteria A1: lambda_max
        y_train_A_wl = train_a["lambda_max"].iloc[idx_tr_A].values
        y_test_A_wl = test_a["lambda_max"].iloc[idx_te_A].values
        self.model_A_wl = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_A_wl.fit(X_train_A, y_train_A_wl)
        
        # Oracle A: lambda_max
        self.oracle_A_wl = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_SEED + 99)
        self.oracle_A_wl.fit(X_train_A, y_train_A_wl)
        
        pred_te_A_wl = self.model_A_wl.predict(X_test_A)
        print(f"  [Criteria A1] lambda_max Test MAE: {mean_absolute_error(y_test_A_wl, pred_te_A_wl):.2f} nm, R2: {r2_score(y_test_A_wl, pred_te_A_wl):.3f}")
        
        # Criteria A2: Energy Storage Capacity Delta H_storage (kJ/mol)
        y_train_A_dh = train_a["delta_h"].iloc[idx_tr_A].values
        y_test_A_dh = test_a["delta_h"].iloc[idx_te_A].values
        self.model_A_dh = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_A_dh.fit(X_train_A, y_train_A_dh)
        pred_te_A_dh = self.model_A_dh.predict(X_test_A)
        print(f"  [Criteria A2] Delta H_storage Test MAE: {mean_absolute_error(y_test_A_dh, pred_te_A_dh):.2f} kJ/mol, R2: {r2_score(y_test_A_dh, pred_te_A_dh):.3f}")
        
        # Criteria A3: PSS (trained on rows where PSS was measured)
        train_pss = train_a.iloc[idx_tr_A].dropna(subset=["pss_z"])
        if len(train_pss) > 10:
            X_tr_pss, _ = featurize_smiles_list(train_pss["canonical_smiles"].tolist())
            y_tr_pss = train_pss["pss_z"].values
            self.model_A_pss = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)
            self.model_A_pss.fit(X_tr_pss, y_tr_pss)
            print(f"  [Criteria A3] PSS Evaluator trained on {len(train_pss)} points.")
        else:
            self.model_A_pss = None
            
        # Criteria A4: Thermal isomerization rate log10(k_thermal)
        train_k = train_a.iloc[idx_tr_A].dropna(subset=["k_thermal"])
        train_k = train_k[train_k["k_thermal"] > 0]
        if len(train_k) > 10:
            X_tr_k, _ = featurize_smiles_list(train_k["canonical_smiles"].tolist())
            y_tr_k = np.log10(train_k["k_thermal"].values)
            self.model_A_k = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)
            self.model_A_k.fit(X_tr_k, y_tr_k)
            print(f"  [Criteria A4] Thermal Rate Evaluator trained on {len(train_k)} points.")
        else:
            self.model_A_k = None
            
        # --- 2. Train Group B Evaluators (UV / Skin) ---
        print("\nTraining Evaluators for Group B (UV Protection / Skin Safety)...")
        X_train_B, idx_tr_B = featurize_smiles_list(train_b["canonical_smiles"].tolist())
        X_test_B, idx_te_B = featurize_smiles_list(test_b["canonical_smiles"].tolist())
        
        # Criteria B1: lambda_max (UV absorption)
        y_train_B_wl = train_b["lambda_max"].iloc[idx_tr_B].values
        y_test_B_wl = test_b["lambda_max"].iloc[idx_te_B].values
        self.model_B_wl = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=RANDOM_SEED)
        self.model_B_wl.fit(X_train_B, y_train_B_wl)
        
        # Oracle B: lambda_max
        self.oracle_B_wl = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_SEED + 99)
        self.oracle_B_wl.fit(X_train_B, y_train_B_wl)
        
        pred_te_B_wl = self.model_B_wl.predict(X_test_B)
        print(f"  [Criteria B1] UV lambda_max Test MAE: {mean_absolute_error(y_test_B_wl, pred_te_B_wl):.2f} nm, R2: {r2_score(y_test_B_wl, pred_te_B_wl):.3f}")
        
        # Criteria B2: Molar extinction coefficient log_eps
        train_eps = train_b.iloc[idx_tr_B].dropna(subset=["log_eps"])
        if len(train_eps) > 50:
            X_tr_eps, _ = featurize_smiles_list(train_eps["canonical_smiles"].tolist())
            y_tr_eps = train_eps["log_eps"].values
            self.model_B_eps = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=RANDOM_SEED)
            self.model_B_eps.fit(X_tr_eps, y_tr_eps)
            print(f"  [Criteria B2] Molar Extinction log(eps) Evaluator trained on {len(train_eps)} experimental points.")
        else:
            self.model_B_eps = None
            
        # Save models to disk
        joblib.dump(self.model_A_wl, self.models_dir / "model_A_wl.joblib")
        joblib.dump(self.model_A_dh, self.models_dir / "model_A_dh.joblib")
        joblib.dump(self.model_A_pss, self.models_dir / "model_A_pss.joblib")
        joblib.dump(self.model_A_k, self.models_dir / "model_A_k.joblib")
        joblib.dump(self.oracle_A_wl, self.models_dir / "oracle_A_wl.joblib")
        
        joblib.dump(self.model_B_wl, self.models_dir / "model_B_wl.joblib")
        joblib.dump(self.model_B_eps, self.models_dir / "model_B_eps.joblib")
        joblib.dump(self.oracle_B_wl, self.models_dir / "oracle_B_wl.joblib")
        joblib.dump({"fps_A": self.train_fps_A, "fps_B": self.train_fps_B}, self.models_dir / "ad_fingerprints.joblib")
        print("\nAll Evaluators & Oracles successfully saved to models/.")

    def load_models(self):
        """Load trained models from disk."""
        self.model_A_wl = joblib.load(self.models_dir / "model_A_wl.joblib")
        self.model_A_dh = joblib.load(self.models_dir / "model_A_dh.joblib")
        self.model_A_pss = joblib.load(self.models_dir / "model_A_pss.joblib")
        self.model_A_k = joblib.load(self.models_dir / "model_A_k.joblib")
        self.oracle_A_wl = joblib.load(self.models_dir / "oracle_A_wl.joblib")
        
        self.model_B_wl = joblib.load(self.models_dir / "model_B_wl.joblib")
        self.model_B_eps = joblib.load(self.models_dir / "model_B_eps.joblib")
        self.oracle_B_wl = joblib.load(self.models_dir / "oracle_B_wl.joblib")
        
        fps_dict = joblib.load(self.models_dir / "ad_fingerprints.joblib")
        self.train_fps_A = fps_dict["fps_A"]
        self.train_fps_B = fps_dict["fps_B"]

    def evaluate_candidates(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Evaluate candidate molecules across all 4 criteria of Group A and 4 criteria of Group B,
        plus Applicability Domain reliability and joint constraint satisfaction.
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
            
            # --- Group A Evaluators (MOST) ---
            # Criteria A1: lambda_max (nm)
            p_A_wl = float(self.model_A_wl.predict(feats)[0])
            pass_A1 = (CRITERIA["most_wavelength_min"] <= p_A_wl <= CRITERIA["most_wavelength_max"])
            
            # Criteria A2: Energy storage capacity Delta H_storage (kJ/mol)
            p_A_dh = float(self.model_A_dh.predict(feats)[0]) if self.model_A_dh else 65.0
            pass_A2 = (p_A_dh >= CRITERIA["most_delta_h_min"])
            
            # Criteria A3: PSS (%) or thermal kinetic barrier
            p_A_pss = float(self.model_A_pss.predict(feats)[0]) if self.model_A_pss else 75.0
            p_A_log_k = float(self.model_A_k.predict(feats)[0]) if self.model_A_k else -4.0
            pass_A3 = (p_A_pss >= CRITERIA["most_pss_min"]) or (p_A_log_k <= CRITERIA["most_log_k_max"])
            
            # Criteria A4: SA Score
            sa_score = compute_sa_score(mol)
            pass_A4 = (sa_score <= CRITERIA["most_sa_max"])
            
            pass_group_A = pass_A1 and pass_A2 and pass_A3 and pass_A4
            
            # --- Group B Evaluators (UV / Skin / Safety) ---
            # Criteria B1: UV lambda_max (nm)
            p_B_wl = float(self.model_B_wl.predict(feats)[0])
            pass_B1 = (CRITERIA["uv_wavelength_min"] <= p_B_wl <= CRITERIA["uv_wavelength_max"])
            
            # Criteria B2: Molar Extinction log10(eps)
            p_B_log_eps = float(self.model_B_eps.predict(feats)[0]) if self.model_B_eps else 4.2
            pass_B2 = (p_B_log_eps >= CRITERIA["uv_log_eps_min"])
            
            # Criteria B3: Skin permeability log Kp (cm/h)
            calc_log_kp = compute_potts_guy_log_kp(mol)
            pass_B3 = (calc_log_kp <= CRITERIA["log_kp_max"])
            
            # Criteria B4: Film size & lipophilicity barrier (MW & LogP)
            mw_val = float(Descriptors.MolWt(mol))
            logp_val = float(Descriptors.MolLogP(mol))
            pass_B4 = (CRITERIA["mw_min"] <= mw_val <= CRITERIA["mw_max"]) and (CRITERIA["logp_min"] <= logp_val <= CRITERIA["logp_max"])
            
            pass_group_B = pass_B1 and pass_B2 and pass_B3 and pass_B4
            
            # --- Applicability Domain (AD) ---
            fp = get_morgan_fp(mol)
            sims_A = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_A)
            dist_A = float(1.0 - max(sims_A)) if sims_A else 1.0
            
            sims_B = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_B)
            dist_B = float(1.0 - max(sims_B)) if sims_B else 1.0
            
            uncertainty_ad = (dist_A + dist_B) / 2.0
            pass_ad = (dist_A <= CRITERIA["ad_distance_max"]) or (dist_B <= CRITERIA["ad_distance_max"])
            
            # Joint Success Rate (all 8 criteria + AD)
            pass_all = pass_group_A and pass_group_B and pass_ad
            
            rows.append({
                "SMILES": can_smi,
                # Group A properties
                "pred_A_wavelength": round(p_A_wl, 2),
                "pred_A_delta_h": round(p_A_dh, 2),
                "pred_A_pss": round(p_A_pss, 1),
                "pred_A_log_k": round(p_A_log_k, 2),
                "synthetic_accessibility": round(sa_score, 2),
                # Group B properties
                "pred_B_wavelength": round(p_B_wl, 2),
                "pred_B_log_eps": round(p_B_log_eps, 2),
                "potts_guy_log_kp": round(calc_log_kp, 2),
                "molecular_weight": round(mw_val, 1),
                "log_p": round(logp_val, 2),
                # Reliability & AD
                "dist_to_DA": round(dist_A, 3),
                "dist_to_DB": round(dist_B, 3),
                "uncertainty_AD": round(uncertainty_ad, 3),
                # Constraint Flags
                "pass_group_A": bool(pass_group_A),
                "pass_group_B": bool(pass_group_B),
                "pass_ad": bool(pass_ad),
                "pass_constraints": bool(pass_all)
            })
            
        return pd.DataFrame(rows)

if __name__ == "__main__":
    suite = PropertyEvaluatorSuite()
    suite.fit_and_evaluate()
