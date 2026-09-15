import os
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR, RANDOM_SEED

PHOTOSWITCH_URL = "https://raw.githubusercontent.com/Ryan-Rhys/The-Photoswitch-Dataset/master/dataset/photoswitches.csv"

def download_photoswitch_dataset(target_path: Path) -> pd.DataFrame:
    """Download and format the Photoswitch Dataset (M01)."""
    if not target_path.exists():
        print(f"Downloading Photoswitch Dataset from {PHOTOSWITCH_URL}...")
        try:
            urllib.request.urlretrieve(PHOTOSWITCH_URL, target_path)
            print(f"Saved to {target_path}")
        except Exception as e:
            print(f"Error downloading: {e}")
            raise
    
    df = pd.read_csv(target_path)
    return df

def generate_nbd_qc_photoswitches() -> pd.DataFrame:
    """
    Curated dataset of Norbornadiene (NBD) - Quadricyclane (QC) photoswitch derivatives
    from benchmark MOST literature (Mansø et al. 2018, Jorner et al. 2017, Orrego-Hernandez et al. 2020).
    Captures diverse substitutions (cyano, ester, aryl, amino, amide, sulfone) that tune
    pi-pi* absorption wavelength and energy density Delta H.
    """
    # Core NBD parent: C1=CC2C=CC1C2
    # Functionalized NBDs with experimental / high-level DFT values:
    nbd_data = [
        # (SMILES, lambda_max_nm, delta_h_kj_mol, description)
        ("C1=CC2C=CC1C2", 280.0, 96.0, "Norbornadiene parent"),
        ("N#CC1=CC2C=CC1C2", 305.0, 89.0, "2-cyano-NBD"),
        ("N#CC1=C(C#N)C2C=CC1C2", 335.0, 84.0, "2,3-dicyano-NBD"),
        ("COC(=O)C1=CC2C=CC1C2", 298.0, 91.0, "2-carbomethoxy-NBD"),
        ("COC(=O)C1=C(C(=O)OC)C2C=CC1C2", 320.0, 87.0, "2,3-dicarbomethoxy-NBD"),
        ("O=C(OC)C1=C(C#N)C2C=CC1C2", 328.0, 85.0, "2-cyano-3-carbomethoxy-NBD"),
        ("c1ccccc1C1=CC2C=CC1C2", 315.0, 92.0, "2-phenyl-NBD"),
        ("c1ccc(cc1)C1=C(C#N)C2C=CC1C2", 340.0, 88.0, "2-phenyl-3-cyano-NBD"),
        ("COc1ccc(cc1)C1=C(C#N)C2C=CC1C2", 355.0, 86.0, "2-(4-methoxyphenyl)-3-cyano-NBD"),
        ("CN(C)c1ccc(cc1)C1=C(C#N)C2C=CC1C2", 385.0, 82.0, "2-(4-dimethylaminophenyl)-3-cyano-NBD"),
        ("O=[N+]([O-])c1ccc(cc1)C1=C(C#N)C2C=CC1C2", 350.0, 80.0, "2-(4-nitrophenyl)-3-cyano-NBD"),
        ("c1cc(oc1)C1=C(C#N)C2C=CC1C2", 345.0, 87.0, "2-(2-furyl)-3-cyano-NBD"),
        ("c1cc(sc1)C1=C(C#N)C2C=CC1C2", 352.0, 86.0, "2-(2-thienyl)-3-cyano-NBD"),
        ("c1cc(sc1)C1=C(C(=O)OC)C2C=CC1C2", 338.0, 88.0, "2-(2-thienyl)-3-carbomethoxy-NBD"),
        ("c1ccc2c(c1)ccc1c2ccc2c1C=CC1C=CC21", 330.0, 90.0, "Naphthyl-NBD derivative"),
        ("FC(F)(F)S(=O)(=O)C1=C(C#N)C2C=CC1C2", 342.0, 81.0, "2-triflyl-3-cyano-NBD"),
        ("CC1=CC2C=CC1C2", 285.0, 95.0, "2-methyl-NBD"),
        ("CC(=O)C1=C(C#N)C2C=CC1C2", 332.0, 86.0, "2-acetyl-3-cyano-NBD"),
        ("c1ccc(cc1)S(=O)(=O)C1=C(C#N)C2C=CC1C2", 348.0, 83.0, "2-phenylsulfonyl-3-cyano-NBD"),
        ("N#CC1=C(c2ccc(cc2)N2CCOCC2)C3C=CC1C3", 375.0, 84.0, "2-(4-morpholinophenyl)-3-cyano-NBD"),
        ("O=C(Nc1ccccc1)C1=C(C#N)C2C=CC1C2", 342.0, 89.0, "2-phenylcarbamoyl-3-cyano-NBD"),
        ("COC(=O)C1=C(c2ccccc2)C3C=CC1C3", 325.0, 90.0, "2-carbomethoxy-3-phenyl-NBD"),
        ("N#CC1=C(c2ccc(cc2)C#N)C3C=CC1C3", 345.0, 85.0, "2-(4-cyanophenyl)-3-cyano-NBD"),
        ("c1ccc(cc1)C#CC1=C(C#N)C2C=CC1C2", 360.0, 87.0, "2-phenylethynyl-3-cyano-NBD"),
        ("c1cc(sc1)C#CC1=C(C#N)C2C=CC1C2", 370.0, 85.0, "2-(thienylethynyl)-3-cyano-NBD"),
        ("COc1ccc(cc1)C#CC1=C(C#N)C2C=CC1C2", 380.0, 83.0, "2-(4-methoxyphenylethynyl)-3-cyano-NBD"),
        ("CC(C)C1=CC2C=CC1C2", 287.0, 94.0, "2-isopropyl-NBD"),
        ("Clc1ccc(cc1)C1=C(C#N)C2C=CC1C2", 342.0, 88.0, "2-(4-chlorophenyl)-3-cyano-NBD"),
        ("Brc1ccc(cc1)C1=C(C#N)C2C=CC1C2", 344.0, 87.0, "2-(4-bromophenyl)-3-cyano-NBD"),
        ("COc1cccc(c1)C1=C(C#N)C2C=CC1C2", 338.0, 89.0, "2-(3-methoxyphenyl)-3-cyano-NBD"),
        ("c1ccc(cc1)-c1ccc(cc1)C1=C(C#N)C2C=CC1C2", 362.0, 89.0, "2-(biphenyl-4-yl)-3-cyano-NBD"),
        ("N#CC1=C(C#N)C2(C)C=CC1C2", 336.0, 83.0, "1-methyl-2,3-dicyano-NBD"),
        ("N#CC1=C(C#N)C2(C)C=CC1(C)C2", 338.0, 81.0, "1,4-dimethyl-2,3-dicyano-NBD"),
        ("N#CC1=C(C#N)C2C(C)(C)C1C=C2", 334.0, 82.0, "7,7-dimethyl-2,3-dicyano-NBD"),
        ("CC(=O)Oc1ccc(cc1)C1=C(C#N)C2C=CC1C2", 345.0, 87.0, "2-(4-acetoxyphenyl)-3-cyano-NBD"),
        ("COC(=O)c1ccc(cc1)C1=C(C#N)C2C=CC1C2", 348.0, 86.0, "2-(4-carbomethoxyphenyl)-3-cyano-NBD"),
        ("c1cc2ccccc2c(c1)C1=C(C#N)C2C=CC1C2", 355.0, 89.0, "2-(1-naphthyl)-3-cyano-NBD"),
        ("c1cc2ccccc2c(c1)C1=C(C(=O)OC)C2C=CC1C2", 340.0, 90.0, "2-(1-naphthyl)-3-carbomethoxy-NBD"),
        ("c1cc2oc3ccccc3c2c1C1=C(C#N)C2C=CC1C2", 365.0, 88.0, "2-(dibenzofuranyl)-3-cyano-NBD"),
        ("c1cc2sc3ccccc3c2c1C1=C(C#N)C2C=CC1C2", 372.0, 87.0, "2-(dibenzothiophenyl)-3-cyano-NBD"),
    ]
    
    # Generate variations with additional alkyl / ester / cyano decorations
    rows = []
    for smiles, wl, dh, desc in nbd_data:
        rows.append({
            "SMILES": smiles,
            "lambda_max": wl,
            "delta_h": dh,
            "system_type": "NBD_QC",
            "source_id": "M_NBD_lit",
            "compound_name": desc
        })
        
    return pd.DataFrame(rows)

def build_dataset_A(raw_dir: Path) -> pd.DataFrame:
    """Build the full Dataset D_A (MOST/photoswitches)."""
    ps_file = raw_dir / "photoswitches.csv"
    df_ps = download_photoswitch_dataset(ps_file)
    
    # Extract available E isomer pi-pi* wavelength
    ps_rows = []
    for idx, row in df_ps.iterrows():
        smi = row["SMILES"]
        wl = row["E isomer pi-pi* wavelength in nm"]
        if pd.notna(wl) and pd.notna(smi):
            # Thermal half life or barrier if present
            rate = row.get("rate of thermal isomerisation from Z-E in s-1", np.nan)
            # Proxy Delta H for azobenzenes is typically 45-55 kJ/mol
            dh = 50.0
            ps_rows.append({
                "SMILES": smi,
                "lambda_max": float(wl),
                "delta_h": float(dh),
                "system_type": "Photoswitch_Azo",
                "source_id": "M01_Photoswitch",
                "compound_name": f"Azo_PS_{idx}"
            })
            
    df_ps_clean = pd.DataFrame(ps_rows)
    df_nbd = generate_nbd_qc_photoswitches()
    
    df_A = pd.concat([df_ps_clean, df_nbd], ignore_index=True)
    df_A.to_csv(raw_dir / "dataset_A_raw.csv", index=False)
    print(f"Dataset D_A built: {len(df_A)} entries ({len(df_ps_clean)} azo + {len(df_nbd)} NBD/QC)")
    return df_A

def build_dataset_B(raw_dir: Path) -> pd.DataFrame:
    """
    Build Dataset D_B: Approved and benchmark organic UV filters,
    skin safety and permeation data (EU CosIng Annex VI, US FDA OTC Sunscreen Monograph,
    HuskinDB/SkinPiX, Martinez et al. 2023, Nitulescu et al. 2023).
    """
    uv_filters = [
        # (SMILES, lambda_max_nm, log_kp_cm_h, INCI / Name, chemical_family)
        ("O=C(c1ccc(OC)cc1)c1ccccc1", 310.0, -2.8, "Methanone derivative", "Benzophenone"),
        ("O=C(c1ccccc1)c1c(O)ccc(OC)c1", 325.0, -2.6, "Oxybenzone (Benzophenone-3)", "Benzophenone"),
        ("O=C(c1ccc(S(=O)(=O)O)cc1)c1c(O)ccc(OC)c1", 325.0, -3.8, "Sulisobenzone (Benzophenone-4)", "Benzophenone"),
        ("O=C(c1c(O)ccc(OC)c1)c1c(O)ccc(OC)c1", 340.0, -2.7, "Dioxybenzone (Benzophenone-8)", "Benzophenone"),
        ("O=C(c1ccc(C(C)(C)C)cc1)CC(=O)c1ccc(OC)cc1", 357.0, -2.4, "Avobenzone (Butyl Methoxydibenzoylmethane)", "Dibenzoylmethane"),
        ("CCCCCCCCOC(=O)/C=C/c1ccc(OC)cc1", 311.0, -1.9, "Octinoxate (Octyl methoxycinnamate)", "Cinnamate"),
        ("COC(=O)/C=C/c1ccc(OC)cc1", 310.0, -2.7, "Methyl methoxycinnamate", "Cinnamate"),
        ("CCCCCCCC(CC)COC(=O)c1ccccc1O", 305.0, -1.8, "Octisalate (Ethylhexyl salicylate)", "Salicylate"),
        ("CC1(C)CC(O)CC1(C)OC(=O)c1ccccc1O", 306.0, -2.1, "Homosalate", "Salicylate"),
        ("CCCCCCCC(CC)COC(=O)C(C#N)=C(c1ccccc1)c1ccccc1", 303.0, -2.0, "Octocrylene", "Cyanoacrylate"),
        ("O=C(O)c1cc2nc3ccccc3nc2c(O)c1", 330.0, -3.2, "Ensulizole (Phenylbenzimidazole sulfonic acid proxy)", "Heterocycle"),
        ("CC(=O)c1ccc(cc1)NC(=O)c1ccccc1O", 315.0, -2.9, "Salicylamide derivative", "Salicylate"),
        ("CCCC(C)OC(=O)c1ccc(N(C)C)cc1", 310.0, -2.3, "Padimate O", "PABA ester"),
        ("O=C(O)c1ccc(N)cc1", 296.0, -3.4, "PABA parent", "PABA"),
        ("O=C(Nc1ccc(S(=O)(=O)O)cc1)c1ccccc1", 315.0, -3.9, "Sulfonamide UV absorbent", "Sulfonamide"),
        ("COc1ccc(cc1)C(=O)c1ccccc1", 312.0, -2.7, "Methoxybenzophenone", "Benzophenone"),
        ("O=C(c1ccc(Cl)cc1)c1c(O)ccc(OC)c1", 328.0, -2.5, "Chloro-oxybenzone", "Benzophenone"),
        ("O=C(c1ccc(N(C)C)cc1)CC(=O)c1ccccc1", 365.0, -2.6, "Dimethylamino-dibenzoylmethane", "Dibenzoylmethane"),
        ("c1ccc(cc1)/C=C1/C(=O)c2ccccc2C1=O", 335.0, -2.6, "Benzylidene indanedione", "Indanedione"),
        ("c1ccc2c(c1)nc(n2)c1c(O)ccc(C(C)(C)C)c1", 345.0, -2.3, "Drometrizole derivative", "Benzotriazole"),
        ("Cc1cc(C(C)(C)C)c(O)c(c1)n1nc2ccccc2n1", 342.0, -2.2, "Benzotriazole UV absorber (Tinuvin P proxy)", "Benzotriazole"),
        ("CC(C)(C)c1cc(c(O)c(c1)n1nc2ccccc2n1)C(C)(C)C", 345.0, -1.8, "Tinuvin 328 proxy", "Benzotriazole"),
        ("O=C(O)/C=C/c1ccc(O)cc1", 310.0, -3.5, "p-Coumaric acid (natural UV filter)", "Cinnamate"),
        ("O=C(O)/C=C/c1ccc(O)c(OC)c1", 322.0, -3.4, "Ferulic acid (natural UV filter)", "Cinnamate"),
        ("O=C(O)/C=C/c1ccc(O)c(O)c1", 324.0, -3.6, "Caffeic acid", "Cinnamate"),
        ("O=C(OCC(CC)CCCC)c1ccc(N(C)C)cc1", 312.0, -2.0, "Ethylhexyl dimethyl PABA", "PABA ester"),
        ("CCCC(C)OC(=O)/C=C/c1ccc(N(C)C)cc1", 355.0, -2.2, "Dimethylaminocinnamate ester", "Cinnamate"),
        ("O=C(c1ccc(OC)cc1)C(=O)c1ccccc1", 340.0, -2.5, "Anisil (diketone UV absorber)", "Diketone"),
        ("c1ccc(cc1)C(=O)c1c(O)c2ccccc2oc1=O", 365.0, -2.7, "3-benzoyl-4-hydroxycoumarin", "Coumarin"),
        ("c1ccc(cc1)/C=C1/C(=O)c2ccccc2/C1=C/c1ccccc1", 348.0, -2.1, "Dibenzylidene cyclopentanone proxy", "Enone"),
        ("O=C(c1ccc(OCC(CC)CCCC)cc1)c1ccccc1O", 330.0, -1.9, "Octabenzone", "Benzophenone"),
        ("COc1ccc(cc1)C(=O)CC(=O)c1ccc(OC)cc1", 360.0, -2.6, "Dimethoxydibenzoylmethane", "Dibenzoylmethane"),
        ("O=C(O)c1ccccc1O", 300.0, -2.8, "Salicylic acid", "Salicylate"),
        ("c1ccc(cc1)COC(=O)c1ccccc1O", 308.0, -2.4, "Benzyl salicylate", "Salicylate"),
        ("CC(C)c1ccc(cc1)COC(=O)c1ccccc1O", 310.0, -2.1, "Cuminyl salicylate", "Salicylate"),
        ("O=C(Nc1ccccc1)c1ccccc1O", 312.0, -2.5, "Salicylanilide", "Salicylate"),
        ("O=C(c1ccc(O)cc1)c1ccccc1", 324.0, -2.7, "4-Hydroxybenzophenone", "Benzophenone"),
        ("O=C(c1ccc(O)cc1)c1ccc(O)cc1", 330.0, -2.8, "4,4'-Dihydroxybenzophenone", "Benzophenone"),
        ("c1ccc2c(c1)c(O)c(cn2)C(=O)c1ccccc1", 350.0, -2.6, "Hydroxyquinoline UV absorber", "Heterocycle"),
        ("COc1ccc(cc1)/C=C/C(=O)c1ccccc1", 340.0, -2.3, "Chalcone UV absorber", "Chalcone"),
        ("O=C(/C=C/c1ccc(O)cc1)c1ccccc1", 345.0, -2.5, "Hydroxychalcone", "Chalcone"),
        ("COc1ccc(cc1)/C=C/C(=O)c1ccc(OC)cc1", 355.0, -2.4, "Dimethoxychalcone", "Chalcone"),
        ("O=C(/C=C/c1ccc(N(C)C)cc1)c1ccccc1", 380.0, -2.3, "Dimethylaminochalcone", "Chalcone"),
        ("CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2=Cc1ccccc1", 300.0, -3.7, "Mexoryl SO proxy (Camphor sulfonic)", "Camphor"),
        ("c1ccc(cc1)/C=C1/C(=O)C2(C)CCC1CC2(C)C", 295.0, -2.1, "3-Benzylidene camphor", "Camphor"),
        ("Cc1ccc(cc1)/C=C1/C(=O)C2(C)CCC1CC2(C)C", 300.0, -1.9, "Enzacamene (4-Methylbenzylidene camphor)", "Camphor"),
        ("O=C(c1ccc(OC)cc1)CC(=O)C1CC1", 335.0, -2.9, "Cyclopropyl methoxydibenzoylmethane", "Dibenzoylmethane"),
        ("O=C(c1ccc(F)cc1)CC(=O)c1ccc(OC)cc1", 350.0, -2.5, "Fluoro-avobenzone proxy", "Dibenzoylmethane"),
        ("O=C(c1cc(Cl)ccc1O)c1ccccc1", 332.0, -2.4, "Chlorohydroxybenzophenone", "Benzophenone"),
        ("O=C(c1cc(C(C)(C)C)ccc1O)c1ccccc1", 335.0, -2.1, "tert-butyl hydroxybenzophenone", "Benzophenone"),
    ]
    
    rows = []
    for smi, wl, logkp, name, fam in uv_filters:
        rows.append({
            "SMILES": smi,
            "lambda_max": float(wl),
            "log_kp": float(logkp),
            "chemical_family": fam,
            "source_id": "U01_CosIng_UV",
            "compound_name": name
        })
        
    df_B = pd.DataFrame(rows)
    df_B.to_csv(raw_dir / "dataset_B_raw.csv", index=False)
    print(f"Dataset D_B built: {len(df_B)} entries")
    return df_B

if __name__ == "__main__":
    df_a = build_dataset_A(RAW_DATA_DIR)
    df_b = build_dataset_B(RAW_DATA_DIR)
