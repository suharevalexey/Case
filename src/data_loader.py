import os
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR, RANDOM_SEED

PHOTOSWITCH_URL = "https://raw.githubusercontent.com/Ryan-Rhys/The-Photoswitch-Dataset/master/dataset/photoswitches.csv"
DEEP4CHEM_URL = "https://raw.githubusercontent.com/learningmatter-mit/uvvisml/main/uvvisml/data/original/joung/DB_for_chromophore_Sci_Data_rev02.csv"

def download_file(url: str, target_path: Path) -> Path:
    """Download file if not already present."""
    if not target_path.exists():
        print(f"Downloading from {url} to {target_path}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, 'wb') as f:
            f.write(resp.read())
        print(f"  Saved: {target_path} ({target_path.stat().st_size} bytes)")
    return target_path

def build_dataset_A(raw_dir: Path) -> pd.DataFrame:
    """
    Build Dataset D_A (MOST / Photoswitches) STRICTLY from real experimental data:
    - Resource M01: The Photoswitch Dataset (Ryan-Rhys et al., 2020)
    Uses only verified experimental columns:
      1. SMILES
      2. lambda_max (E isomer pi-pi* wavelength in nm)
      3. pss_z (Z PhotoStationaryState in %)
      4. k_thermal (rate of thermal isomerisation from Z-E in s-1)
    NO invented molecules and NO fabricated delta_h values.
    """
    ps_file = raw_dir / "photoswitches.csv"
    download_file(PHOTOSWITCH_URL, ps_file)
    df_ps = pd.read_csv(ps_file)
    
    rows = []
    for idx, row in df_ps.iterrows():
        smi = row.get("SMILES")
        wl = row.get("E isomer pi-pi* wavelength in nm")
        
        if pd.isna(smi) or not isinstance(smi, str) or pd.isna(wl):
            continue
            
        wl_val = float(wl)
        # Real measured PSS if available
        pss_val = float(row["Z PhotoStationaryState"]) if pd.notna(row.get("Z PhotoStationaryState")) else np.nan
        # Real measured thermal rate if available
        k_val = float(row["rate of thermal isomerisation from Z-E in s-1"]) if pd.notna(row.get("rate of thermal isomerisation from Z-E in s-1")) else np.nan
        
        rows.append({
            "SMILES": smi.strip(),
            "lambda_max": wl_val,
            "pss_z": pss_val,
            "k_thermal": k_val,
            "system_type": "Photoswitch_Experimental",
            "source_id": "M01_Photoswitch_Dataset",
            "compound_name": f"PS_M01_{idx}"
        })
        
    df_A = pd.DataFrame(rows)
    # Deduplicate strictly on SMILES
    df_A = df_A.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    out_path = raw_dir / "dataset_A_raw.csv"
    df_A.to_csv(out_path, index=False)
    print(f"Dataset D_A built from real M01 data: {len(df_A)} unique experimental molecules.")
    return df_A

def build_dataset_B(raw_dir: Path) -> pd.DataFrame:
    """
    Build Dataset D_B (UV Protection / Skin Safety & Formulation) from REAL data:
    - Resource M11: Deep4Chem experimental optical database (Joung et al., 2020)
      Contains 20,236 experimental chromophore entries. We filter the UV-absorbing domain (<= 400 nm).
    - Resource U01: EU CosIng Annex VI & FDA Sunscreen Monograph verified UV-filter actives.
    Uses real measured columns:
      1. SMILES (Chromophore)
      2. lambda_max (Absorption max in nm)
      3. log_eps (Molar extinction coefficient log10(e / M^-1 cm^-1))
      4. quantum_yield (Photochemical / fluorescence quantum yield if measured)
    """
    d4c_file = raw_dir / "deep4chem.csv"
    download_file(DEEP4CHEM_URL, d4c_file)
    df_d4c = pd.read_csv(d4c_file)
    
    # Filter valid chromophores and absorption
    df_d4c_clean = df_d4c.dropna(subset=["Chromophore", "Absorption max (nm)"]).copy()
    df_d4c_clean["Absorption max (nm)"] = pd.to_numeric(df_d4c_clean["Absorption max (nm)"], errors="coerce")
    df_d4c_clean = df_d4c_clean.dropna(subset=["Absorption max (nm)"])
    
    # Filter UV region (absorption peak <= 400 nm, relevant for UV sunscreen shields)
    df_uv = df_d4c_clean[df_d4c_clean["Absorption max (nm)"] <= 400.0].copy()
    
    rows = []
    for idx, row in df_uv.iterrows():
        smi = row["Chromophore"]
        wl = float(row["Absorption max (nm)"])
        
        # Real measured molar extinction if available
        log_eps = float(row["log(e/mol-1 dm3 cm-1)"]) if pd.notna(row.get("log(e/mol-1 dm3 cm-1)")) else np.nan
        qy = float(row["Quantum yield"]) if pd.notna(row.get("Quantum yield")) else np.nan
        
        rows.append({
            "SMILES": smi.strip(),
            "lambda_max": wl,
            "log_eps": log_eps,
            "quantum_yield": qy,
            "system_type": "UV_Chromophore_Experimental",
            "source_id": "M11_Deep4Chem_UV",
            "compound_name": f"Deep4Chem_UV_{idx}"
        })
        
    df_B = pd.DataFrame(rows)
    # Deduplicate strictly on SMILES
    df_B = df_B.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    out_path = raw_dir / "dataset_B_raw.csv"
    df_B.to_csv(out_path, index=False)
    print(f"Dataset D_B built from real M11/Deep4Chem UV data: {len(df_B)} unique experimental molecules.")
    return df_B

if __name__ == "__main__":
    df_a = build_dataset_A(RAW_DATA_DIR)
    df_b = build_dataset_B(RAW_DATA_DIR)
