import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42

# Property Evaluation Criteria (Omega_A and Omega_B)
CRITERIA = {
    # Group A: MOST (Molecular Solar Thermal Energy Storage)
    "most_wavelength_min": 310.0,  # nm (Photoswitch absorption threshold)
    "most_wavelength_max": 420.0,  # nm
    "most_delta_h_min": 40.0,       # kJ/mol (Isomerization energy storage)
    
    # Group B: Sunscreen UV protection & skin safety
    "uv_wavelength_min": 330.0,    # nm (UVA coverage proxy)
    "log_kp_max": -2.5,            # cm/h (Skin permeability limit, low absorption into dermis)
    "sa_score_max": 4.5,           # Synthetic accessibility threshold (1 = easy, 10 = impossible)
    
    # Applicability Domain (AD)
    "ad_distance_max": 0.65,       # Max Tanimoto distance to training nearest neighbor
}

# Training / Generation Budget
TRAIN_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
BASELINE_GEN_COUNT = 1000
