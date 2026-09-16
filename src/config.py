import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"
SOURCES_A_DIR = SOURCES_DIR / "group_A"
SOURCES_B_DIR = SOURCES_DIR / "group_B"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for d in [SOURCES_A_DIR, SOURCES_B_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Property Evaluation Criteria (Omega_A and Omega_B) - 4 criteria per group
# ---------------------------------------------------------------------------
CRITERIA = {
    # --- GROUP A: MOST / Photoswitches (Molecular Solar Thermal Energy Storage) ---
    # 1. Absorption window: matching terrestrial solar irradiance (UVB/UVA activation)
    "most_wavelength_min": 310.0,  # nm
    "most_wavelength_max": 420.0,  # nm
    
    # 2. Specific energy storage capacity (Slides 8, 11, 12)
    # Delta H_storage >= 40.0 kJ/mol (or >= 0.3 MJ/kg) calculated via valence isomerization strain
    "most_delta_h_min": 40.0,      # kJ/mol
    
    # 3. Thermal isomerization kinetic barrier / half-life stability at skin temp ~32-35°C (Slide 13)
    # log10(k_thermal) <= -3.0 corresponds to t_1/2 >= 1-8 hours storage
    "most_log_k_max": -3.0,        # log10(s^-1)
    "most_pss_min": 60.0,          # % (Z PhotoStationaryState >= 60%)
    
    # 4. Synthetic accessibility score (Slides 17, 22, Section 6)
    "most_sa_max": 4.5,            # Ertl SA score <= 4.5 (1 = trivial, 10 = intractable)

    # --- GROUP B: UV Protection / Skin Barrier & Safety ---
    # 1. Broad-spectrum UV peak absorption (Slides 8, 10; FDA 21 CFR §201.327 / ISO 24443)
    "uv_wavelength_min": 330.0,    # nm (UVA-1/UVA-2 broad coverage)
    "uv_wavelength_max": 400.0,    # nm (UVA cutoff, avoiding visible discoloration)
    
    # 2. Molar extinction coefficient: high photon shielding efficiency (Slide 7: A = eps * c * l)
    "uv_log_eps_min": 4.0,         # log10(eps / M^-1 cm^-1) >= 4.0 (eps >= 10,000 M^-1 cm^-1)
    
    # 3. Dermal retention / low skin permeability: OECD 428 / Martinez 2023 (Slide 16)
    # J_ss = K_p * Delta C. log Kp <= -2.5 cm/h ensures stratum corneum barrier localization
    "log_kp_max": -2.5,            # cm/h
    
    # 4. Film-forming barrier & 500-Da rule: uniform matrix without skin diffusion (Slides 15, 16)
    "mw_min": 250.0,               # Da (prevents volatile loss)
    "mw_max": 600.0,               # Da (prevents crystallization defect CV_h)
    "logp_min": 1.5,               # Lipophilicity lower bound (water resistance)
    "logp_max": 5.5,               # Lipophilicity upper bound (sebum compatibility)
    
    # --- Applicability Domain (AD) ---
    "ad_distance_max": 0.65,       # Max Tanimoto distance to nearest training neighbor
}

# Training / Generation Budget
TRAIN_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
BASELINE_GEN_COUNT = 1000
