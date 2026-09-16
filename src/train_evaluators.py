import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, accuracy_score
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from features import get_or_compute_cached_features
from evaluator_model import EnsembleEvaluatorWrapper

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_evaluators_combined.csv")
CACHE_PATH = os.path.join(BASE_DIR, "data", "processed", "features_cache_1036.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models", "evaluators")
ORACLE_DIR = os.path.join(BASE_DIR, "models", "oracle")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ORACLE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PROPERTIES = [
    {
        "name": "absorption_max_nm",
        "type": "regression",
        "group": "group_A",
        "description": "Peak absorption wavelength lambda_max (nm)",
        "target_col": "absorption_max_nm"
    },
    {
        "name": "log_extinction",
        "type": "regression",
        "group": "group_A",
        "description": "Log10 molar extinction coefficient (M^-1 cm^-1)",
        "target_col": "log_extinction"
    },
    {
        "name": "photochem_efficiency",
        "type": "regression",
        "group": "group_A",
        "description": "Photochemical energy storage quantum yield / PSS (0..1)",
        "target_col": "photochem_efficiency"
    },
    {
        "name": "log_kp",
        "type": "regression",
        "group": "group_B",
        "description": "Human skin permeability coefficient logKp (cm/s)",
        "target_col": "log_kp"
    },
    {
        "name": "skin_sensitization",
        "type": "classification",
        "group": "group_B",
        "description": "Skin sensitization potential (0: Inactive/Safe, 1: Active/Sensitizer)",
        "target_col": "skin_sensitization"
    },
    {
        "name": "skin_irritation",
        "type": "classification",
        "group": "group_B",
        "description": "Skin irritation potential (0: Non-irritant, 1: Irritant)",
        "target_col": "skin_irritation"
    }
]

def main():
    print("=== STARTING TRAINING OF PROPERTY EVALUATORS ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df):,} total molecules.")
    
    t0 = time.time()
    X, valid_mask = get_or_compute_cached_features(DATA_PATH, CACHE_PATH)
    print(f"Features ready in {time.time()-t0:.2f}s. Feature matrix shape: {X.shape}")

    metrics_report = {}

    for prop in PROPERTIES:
        prop_name = prop["name"]
        prop_type = prop["type"]
        print("\n" + "="*60)
        print(f"Training Evaluator for: {prop_name} ({prop['description']})")
        print("="*60)

        series = df[prop["target_col"]]
        mask = series.notna() & valid_mask
        idx_labeled = np.where(mask)[0]
        
        y = series.iloc[idx_labeled].values
        X_sub = X[idx_labeled]
        n_samples = len(y)
        print(f"Total labeled samples for {prop_name}: {n_samples:,}")

        if n_samples < 20:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y, test_size=0.20, random_state=42,
            stratify=y if prop_type == "classification" else None
        )

        if prop_type == "regression":
            model = RandomForestRegressor(
                n_estimators=100, max_depth=20, min_samples_leaf=2,
                n_jobs=-1, random_state=42
            )
            model.fit(X_train, y_train)

            wrapper = EnsembleEvaluatorWrapper(model, prop_type, prop_name)
            y_pred, y_std = wrapper.predict_with_uncertainty(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mean_unc = float(np.mean(y_std))

            print(f"Holdout Test Results: R^2 = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}, Mean Uncertainty = {mean_unc:.4f}")
            metrics_report[prop_name] = {
                "type": prop_type,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "R2": float(r2),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "mean_uncertainty": mean_unc
            }

            oracle = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=123, n_jobs=-1)
            oracle.fit(X_train, y_train)
            oracle_r2 = r2_score(y_test, oracle.predict(X_test))
            group = prop["group"]
            joblib.dump(oracle, os.path.join(ORACLE_DIR, f"{group}_{prop_name}_oracle.joblib"))

        else:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=16, min_samples_leaf=2,
                n_jobs=-1, random_state=42, class_weight="balanced"
            )
            model.fit(X_train, y_train)

            wrapper = EnsembleEvaluatorWrapper(model, prop_type, prop_name)
            y_prob, y_std = wrapper.predict_with_uncertainty(X_test)
            y_pred_class = (y_prob >= 0.5).astype(int)

            roc_auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred_class)
            acc = accuracy_score(y_test, y_pred_class)
            mean_unc = float(np.mean(y_std))

            print(f"Holdout Test Results: ROC-AUC = {roc_auc:.4f}, F1 = {f1:.4f}, Accuracy = {acc:.4f}, Mean Uncertainty = {mean_unc:.4f}")
            metrics_report[prop_name] = {
                "type": prop_type,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "ROC_AUC": float(roc_auc),
                "F1": float(f1),
                "Accuracy": float(acc),
                "mean_uncertainty": mean_unc
            }

            oracle = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=123, n_jobs=-1)
            oracle.fit(X_train, y_train)
            oracle_auc = roc_auc_score(y_test, oracle.predict_proba(X_test)[:, 1])
            group = prop["group"]
            print(f"Independent Oracle Test ROC-AUC: {oracle_auc:.4f}")
            joblib.dump(oracle, os.path.join(ORACLE_DIR, f"{group}_{prop_name}_oracle.joblib"))

        group = prop["group"]
        model_save_path = os.path.join(MODELS_DIR, f"{group}_{prop_name}_evaluator.joblib")
        joblib.dump(wrapper, model_save_path)
        print(f"Saved Guidance Surrogate Evaluator to {model_save_path}")

    report_json_path = os.path.join(RESULTS_DIR, "evaluators_performance_report.json")
    report_csv_path = os.path.join(RESULTS_DIR, "evaluators_performance_report.csv")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    pd.DataFrame.from_dict(metrics_report, orient="index").to_csv(report_csv_path)
    print("\n" + "="*70)
    print("=== ALL 6 EVALUATORS TRAINED AND SAVED SUCCESSFULLY ===")
    print("="*70)

if __name__ == "__main__":
    main()
