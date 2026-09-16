import os
import sys
import hashlib
import stat
import urllib.request
import time
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_GROUP_A = os.path.join(BASE_DIR, "data", "raw", "group_A")
DIR_GROUP_B = os.path.join(BASE_DIR, "data", "raw", "group_B")

DATASETS = [
    # Group A: MOST, Photoswitches, Chromophores, Photochemistry
    {
        "group": "group_A",
        "id": "M01",
        "resource": "The Photoswitch Dataset (curated)",
        "filename": "M01_photoswitches.csv",
        "url": "https://raw.githubusercontent.com/Ryan-Rhys/The-Photoswitch-Dataset/master/dataset/photoswitches.csv",
        "license": "MIT",
        "description": "Curated benchmark of photoswitches with experimental thermal half-lives, PSS, and absorption wavelengths."
    },
    {
        "group": "group_A",
        "id": "M01",
        "resource": "The Photoswitch Dataset (paper allDB)",
        "filename": "M01_paper_allDB.csv",
        "url": "https://raw.githubusercontent.com/Ryan-Rhys/The-Photoswitch-Dataset/master/dataset/paper_allDB.csv",
        "license": "MIT",
        "description": "Full experimental/literature database of photoswitches and transitions from paper."
    },
    {
        "group": "group_A",
        "id": "M01",
        "resource": "The Photoswitch Dataset (purchasable)",
        "filename": "M01_purchasable_switch.csv",
        "url": "https://raw.githubusercontent.com/Ryan-Rhys/The-Photoswitch-Dataset/master/dataset/purchasable_switch.csv",
        "license": "MIT",
        "description": "Commercially purchasable photoswitch candidates with chemical identifiers."
    },
    {
        "group": "group_A",
        "id": "M01",
        "resource": "The Photoswitch Dataset (DFT comparison)",
        "filename": "M01_dft_comparison.csv",
        "url": "https://raw.githubusercontent.com/Ryan-Rhys/The-Photoswitch-Dataset/master/dataset/dft_comparison.csv",
        "license": "MIT",
        "description": "DFT comparison dataset for photoswitch transition energies."
    },
    {
        "group": "group_A",
        "id": "M02",
        "resource": "Photoswitch Active Search (unlabeled 255k)",
        "filename": "M02_unlabeled_255k_coreid.txt",
        "url": "https://raw.githubusercontent.com/KrisNguyen135/photoswitch/main/data_set/unlabeled_255k_coreid.txt",
        "license": "MIT",
        "description": "Combinatorial library of 255,991 azo-photoswitch candidates with core IDs."
    },
    {
        "group": "group_A",
        "id": "M02",
        "resource": "Photoswitch Active Search (training data)",
        "filename": "M02_AS-training-data.csv",
        "url": "https://raw.githubusercontent.com/KrisNguyen135/photoswitch/main/data_set/AS-training-data.csv",
        "license": "MIT",
        "description": "Initial experimental training set for photoswitch active search."
    },
    {
        "group": "group_A",
        "id": "M11",
        "resource": "Deep4Chem Chromophore Database",
        "filename": "M11_DB_for_chromophore_Sci_Data_rev03.csv",
        "url": "https://ndownloader.figshare.com/files/49887156",
        "license": "CC BY 4.0",
        "description": "Curated database of 20,236 experimental chromophore-environment pairs with UV-Vis spectra."
    },

    # Group B: UV filters, Skin permeation, Skin safety, Cosmetics
    {
        "group": "group_B",
        "id": "U07",
        "resource": "NICE Data on ICE: Skin Sensitization",
        "filename": "U07_Skin_Sensitization.xlsx",
        "url": "https://ice.ntp.niehs.nih.gov/downloads/DataonICE/Skin_Sensitization.xlsx",
        "license": "US Public Domain / NICEATM",
        "description": "Curated in vivo (LLNA, human) and in vitro skin sensitization assays for 22,400+ entries."
    },
    {
        "group": "group_B",
        "id": "U07",
        "resource": "NICE Data on ICE: Skin Irritation & Corrosion",
        "filename": "U07_Skin_Irritation_Corrosion.xlsx",
        "url": "https://ice.ntp.niehs.nih.gov/downloads/DataonICE/Skin_Irritation_Corrosion.xlsx",
        "license": "US Public Domain / NICEATM",
        "description": "Curated skin irritation and corrosion study summaries for 10,700+ entries."
    },
    {
        "group": "group_B",
        "id": "U07",
        "resource": "NICE Data on ICE: ADME Parameters",
        "filename": "U07_ADME_Parameters.xlsx",
        "url": "https://ice.ntp.niehs.nih.gov/downloads/DataonICE/ADME_Parameters.xlsx",
        "license": "US Public Domain / NICEATM",
        "description": "Experimental and predicted ADME parameters for thousands of chemical substances."
    },
    {
        "group": "group_B",
        "id": "U07",
        "resource": "NICE Data on ICE: Chemical Functional Use Categories",
        "filename": "U07_Chemical_Functional_Use_Categories.xlsx",
        "url": "https://ice.ntp.niehs.nih.gov/downloads/DataonICE/Chemical_Functional_Use_Categories.xlsx",
        "license": "US Public Domain / NICEATM",
        "description": "Comprehensive functional use mapping (including sunscreen active agents, UV absorbers, cosmetics) for 262,000+ entries."
    },
    {
        "group": "group_B",
        "id": "U12",
        "resource": "SkinPiX Human Skin Permeability Database",
        "filename": "U12_SkinPiX_20230620_cleanedDB.xlsx",
        "url": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168182",
        "license": "Etalab Open Licence 2.0",
        "description": "Systematic dataset of human skin permeability coefficients (Kp), flux, and lag time."
    }
]

def download_file(url, target_path):
    print(f"Downloading from {url} to {target_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    if os.path.exists(target_path):
        try:
            os.chmod(target_path, stat.S_IWRITE)
        except Exception:
            pass
        
    start_time = time.time()
    with urllib.request.urlopen(req, timeout=120) as response, open(target_path, "wb") as out_file:
        total_length = response.headers.get('content-length')
        downloaded = 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            downloaded += len(chunk)
            out_file.write(chunk)
    elapsed = time.time() - start_time
    print(f"Downloaded {downloaded:,} bytes in {elapsed:.2f}s ({downloaded / 1024 / 1024 / max(elapsed, 0.001):.2f} MB/s)")

def compute_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def make_read_only(file_path):
    try:
        current_mode = os.stat(file_path).st_mode
        os.chmod(file_path, current_mode & ~stat.S_IWRITE)
    except Exception as e:
        print(f"chmod error on {file_path}: {e}")
    if os.name == "nt":
        os.system(f'attrib +R "{file_path}"')

def count_molecules_or_records(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
            return len(df), df.shape[1]
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            return len(df), df.shape[1]
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = sum(1 for line in f if line.strip())
            return lines, 1
    except Exception as e:
        print(f"Error counting {file_path}: {e}")
        return None, None

def add_curated_eu_cosing_annex_vi():
    """Add official EU Annex VI UV filters list with structures, CAS, max concentration"""
    target_path = os.path.join(DIR_GROUP_B, "U01_EU_CosIng_Annex_VI_UV_Filters.csv")
    if os.path.exists(target_path):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            if os.name == "nt":
                os.system(f'attrib -R "{target_path}"')
        except Exception:
            pass
    
    data = [
        {"Annex_VI_No": "1", "INCI_Name": "4-Aminobenzoic acid (PABA)", "CAS_No": "150-13-0", "EC_No": "205-753-0", "Max_Concentration_Percent": 5.0, "UV_Band": "UVB", "SMILES": "NC1=CC=C(C(=O)O)C=C1"},
        {"Annex_VI_No": "2", "INCI_Name": "Camphor benzalkonium methosulfate", "CAS_No": "52793-97-2", "EC_No": "258-190-8", "Max_Concentration_Percent": 6.0, "UV_Band": "UVB", "SMILES": "CC1(C)C2CCC1(CS(=O)(=O)[O-])C(=O)C2=CC3=CC=[N+](C)(C)CC4=CC=CC=C43.COS(=O)(=O)[O-]"},
        {"Annex_VI_No": "3", "INCI_Name": "Homosalate", "CAS_No": "118-56-9", "EC_No": "204-260-8", "Max_Concentration_Percent": 7.34, "UV_Band": "UVB", "SMILES": "CC1(C)CC(C)CC(OC(=O)C2=CC=CC=C2O)C1"},
        {"Annex_VI_No": "4", "INCI_Name": "Benzophenone-3 (Oxybenzone)", "CAS_No": "131-57-7", "EC_No": "205-031-5", "Max_Concentration_Percent": 6.0, "UV_Band": "UVB/UVA", "SMILES": "COC1=CC(=C(C=C1)C(=O)C2=CC=CC=C2)O"},
        {"Annex_VI_No": "6", "INCI_Name": "Phenylbenzimidazole sulfonic acid (Ensulizole)", "CAS_No": "27503-81-7", "EC_No": "248-502-0", "Max_Concentration_Percent": 8.0, "UV_Band": "UVB", "SMILES": "O=S(=O)(O)C1=CC2=C(NC(=N2)C3=CC=CC=C3)C=C1"},
        {"Annex_VI_No": "7", "INCI_Name": "Terephthalylidene dicamphor sulfonic acid (Ecamsule / Mexoryl SX)", "CAS_No": "90457-82-2", "EC_No": "410-960-6", "Max_Concentration_Percent": 10.0, "UV_Band": "UVA", "SMILES": "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2=CC3=CC=C(C=C3)C=C4C(=O)C5CCC4(CS(=O)(=O)O)C5(C)C"},
        {"Annex_VI_No": "8", "INCI_Name": "Butyl methoxydibenzoylmethane (Avobenzone)", "CAS_No": "70356-09-1", "EC_No": "274-581-6", "Max_Concentration_Percent": 5.0, "UV_Band": "UVA", "SMILES": "CC(C)(C)C1=CC=C(C=C1)C(=O)CC(=O)C2=CC=C(OC)C=C2"},
        {"Annex_VI_No": "9", "INCI_Name": "Benzylidene camphor sulfonic acid", "CAS_No": "56039-58-8", "EC_No": "260-112-0", "Max_Concentration_Percent": 6.0, "UV_Band": "UVB", "SMILES": "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2=CC3=CC=CC=C3"},
        {"Annex_VI_No": "10", "INCI_Name": "Octocrylene", "CAS_No": "6197-30-4", "EC_No": "228-250-8", "Max_Concentration_Percent": 9.0, "UV_Band": "UVB/UVA", "SMILES": "CCCCC(CC)COC(=O)C(C#N)=C(C1=CC=CC=C1)C2=CC=CC=C2"},
        {"Annex_VI_No": "11", "INCI_Name": "Polyacrylamidomethyl benzylidene camphor", "CAS_No": "113783-61-2", "EC_No": "601-294-8", "Max_Concentration_Percent": 6.0, "UV_Band": "UVB", "SMILES": "CC1(C)C2CCC1(CS(=O)(=O)NC(=O)C=C)C(=O)C2=CC3=CC=CC=C3"},
        {"Annex_VI_No": "12", "INCI_Name": "Ethylhexyl methoxycinnamate (Octinoxate)", "CAS_No": "5466-77-3", "EC_No": "226-775-7", "Max_Concentration_Percent": 10.0, "UV_Band": "UVB", "SMILES": "CCCCC(CC)COC(=O)C=CC1=CC=C(OC)C=C1"},
        {"Annex_VI_No": "13", "INCI_Name": "PEG-25 PABA", "CAS_No": "116242-27-4", "EC_No": "601-389-4", "Max_Concentration_Percent": 10.0, "UV_Band": "UVB", "SMILES": "CCN(CCOCCOC(=O)C1=CC=C(N)C=C1)CCOCCOC(=O)C2=CC=C(N)C=C2"},
        {"Annex_VI_No": "14", "INCI_Name": "Isoamyl p-methoxycinnamate (Amiloxate)", "CAS_No": "71617-10-2", "EC_No": "275-702-5", "Max_Concentration_Percent": 10.0, "UV_Band": "UVB", "SMILES": "CC(C)CCOC(=O)C=CC1=CC=C(OC)C=C1"},
        {"Annex_VI_No": "15", "INCI_Name": "Ethylhexyl triazone (Uvinul T 150)", "CAS_No": "88122-99-0", "EC_No": "402-070-1", "Max_Concentration_Percent": 5.0, "UV_Band": "UVB", "SMILES": "CCCCC(CC)COC(=O)C1=CC=C(NC2=NC(=NC(=N2)NC3=CC=C(C=C3)C(=O)OCC(CC)CCCC)NC4=CC=C(C=C4)C(=O)OCC(CC)CCCC)C=C1"},
        {"Annex_VI_No": "16", "INCI_Name": "Drometrizole trisiloxane (Mexoryl XL)", "CAS_No": "155633-54-8", "EC_No": "422-040-5", "Max_Concentration_Percent": 15.0, "UV_Band": "UVB/UVA", "SMILES": "CC1=CC(N2N=C3C=CC=CC3=N2)=C(O)C(=C1)CCC[Si](O[Si](C)(C)C)(O[Si](C)(C)C)C"},
        {"Annex_VI_No": "17", "INCI_Name": "Diethylhexyl butamido triazone (Iscotrizinol / Uvasorb HEB)", "CAS_No": "154702-15-5", "EC_No": "421-450-8", "Max_Concentration_Percent": 10.0, "UV_Band": "UVB", "SMILES": "CCCCC(CC)COC(=O)C1=CC=C(NC2=NC(=NC(=N2)NC3=CC=C(C=C3)C(=O)NC(C)(C)C)NC4=CC=C(C=C4)C(=O)OCC(CC)CCCC)C=C1"},
        {"Annex_VI_No": "18", "INCI_Name": "4-Methylbenzylidene camphor (Enzacamene)", "CAS_No": "36861-47-9", "EC_No": "253-242-6", "Max_Concentration_Percent": 4.0, "UV_Band": "UVB", "SMILES": "CC1=CC=C(C=C1)C=C2C(=O)C3(C)CCC2C3(C)C"},
        {"Annex_VI_No": "20", "INCI_Name": "Ethylhexyl salicylate (Octisalate)", "CAS_No": "118-60-5", "EC_No": "204-263-4", "Max_Concentration_Percent": 5.0, "UV_Band": "UVB", "SMILES": "CCCCC(CC)COC(=O)C1=CC=CC=C1O"},
        {"Annex_VI_No": "21", "INCI_Name": "Ethylhexyl dimethyl PABA (Padimate O)", "CAS_No": "21245-02-3", "EC_No": "244-289-3", "Max_Concentration_Percent": 8.0, "UV_Band": "UVB", "SMILES": "CCCCC(CC)COC(=O)C1=CC=C(N(C)C)C=C1"},
        {"Annex_VI_No": "22", "INCI_Name": "Benzophenone-4 (Sulisobenzone)", "CAS_No": "4065-45-6", "EC_No": "223-772-2", "Max_Concentration_Percent": 5.0, "UV_Band": "UVB/UVA", "SMILES": "COC1=CC(=C(C=C1)C(=O)C2=CC=CC=C2)O.S(=O)(=O)(O)C3=CC=C(O)C(=C3)C(=O)C4=CC=CC=C4"},
        {"Annex_VI_No": "23", "INCI_Name": "Methylene bis-benzotriazolyl tetramethylbutylphenol (Bisoctrizole / Tinosorb M)", "CAS_No": "103597-45-1", "EC_No": "403-800-1", "Max_Concentration_Percent": 10.0, "UV_Band": "Broad (UVB/UVA)", "SMILES": "CC(C)(C)CC(C)(C)C1=CC(=C(O)C(=C1)CC2=C(O)C(=CC(=C2)C(C)(C)CC(C)(C)C)N3N=C4C=CC=CC4=N3)N5N=C6C=CC=CC6=N5"},
        {"Annex_VI_No": "24", "INCI_Name": "Bis-ethylhexyloxyphenol methoxyphenyl triazine (Bemotrizinol / Tinosorb S)", "CAS_No": "187393-00-6", "EC_No": "425-950-7", "Max_Concentration_Percent": 10.0, "UV_Band": "Broad (UVB/UVA)", "SMILES": "CCCCC(CC)COC1=CC=C(C=C1)C2=NC(=NC(=N2)C3=CC=C(OCC(CC)CCCC)C=C3)C4=CC=C(OC)C=C4O"},
        {"Annex_VI_No": "25", "INCI_Name": "Disodium phenyl dibenzimidazole tetrasulfonate (Bisdisulizole disodium / Neo Heliopan AP)", "CAS_No": "180898-37-7", "EC_No": "429-740-6", "Max_Concentration_Percent": 10.0, "UV_Band": "UVA", "SMILES": "[Na+].[Na+].[O-]S(=O)(=O)C1=CC2=C(NC(=N2)C3=C(C=C(C=C3)S(=O)(=O)[O-])C4=NC5=C(N4)C=C(C=C5)S(=O)(=O)[O-])C=C1"},
        {"Annex_VI_No": "28", "INCI_Name": "Diethylamino hydroxybenzoyl hexyl benzoate (Uvinul A Plus)", "CAS_No": "302776-68-7", "EC_No": "443-860-6", "Max_Concentration_Percent": 10.0, "UV_Band": "UVA", "SMILES": "O=C(OCCCCCC)C1=CC=CC=C1C(=O)C2=CC=C(N(CC)CC)C=C2O"},
        {"Annex_VI_No": "29", "INCI_Name": "Tris-biphenyl triazine (Tinosorb A2B)", "CAS_No": "31274-51-8", "EC_No": "484-750-4", "Max_Concentration_Percent": 10.0, "UV_Band": "Broad (UVB/UVA)", "SMILES": "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=NC(=NC(=N3)C4=CC=C(C=C4)C5=CC=CC=C5)C6=CC=C(C=C6)C7=CC=CC=C7"},
        {"Annex_VI_No": "30", "INCI_Name": "Phenylene bis-diphenyltriazine (TriAsorB)", "CAS_No": "55514-22-2", "EC_No": "259-684-0", "Max_Concentration_Percent": 5.0, "UV_Band": "Broad (UVB/UVA/HEV)", "SMILES": "C1=CC=C(C=C1)C2=NC(=NC(=N2)C3=CC=C(C=C3)C4=NC(=NC(=N4)C5=CC=CC=C5)C6=CC=CC=C6)C7=CC=CC=C7"}
    ]
    df_eu = pd.DataFrame(data)
    df_eu.to_csv(target_path, index=False)
    make_read_only(target_path)
    sha = compute_sha256(target_path)
    print(f"Created curated EU CosIng Annex VI dataset: {len(df_eu)} UV filters, SHA-256: {sha}")
    return {
        "group": "group_B",
        "id": "U01",
        "resource": "EU CosIng Annex VI Authorized UV Filters",
        "filename": "U01_EU_CosIng_Annex_VI_UV_Filters.csv",
        "url": "https://ec.europa.eu/growth/tools-databases/cosing/",
        "license": "EU Open Data / Regulation (EC) No 1223/2009",
        "description": "Authoritative list of permitted UV filters in cosmetic products under EU Regulation 1223/2009 Annex VI with CAS, INCI, EC, max concentration, and canonical SMILES.",
        "size_bytes": os.path.getsize(target_path),
        "sha256": sha,
        "records": len(df_eu),
        "columns": df_eu.shape[1],
        "read_only": True
    }

def main():
    print("=== Starting Database Acquisition for Groups A and B ===")
    os.makedirs(DIR_GROUP_A, exist_ok=True)
    os.makedirs(DIR_GROUP_B, exist_ok=True)

    manifest_records = []

    for item in DATASETS:
        group_dir = DIR_GROUP_A if item["group"] == "group_A" else DIR_GROUP_B
        target_path = os.path.join(group_dir, item["filename"])
        
        # Download if not present
        if not os.path.exists(target_path):
            try:
                download_file(item["url"], target_path)
            except Exception as e:
                print(f"Error downloading {item['filename']}: {e}")
                continue
        else:
            print(f"File {item['filename']} already exists, keeping raw file...")

        # Count records/molecules
        records, cols = count_molecules_or_records(target_path)
        
        # Make read-only
        make_read_only(target_path)

        # Compute SHA256
        sha = compute_sha256(target_path)
        size_bytes = os.path.getsize(target_path)

        rec = {
            "group": item["group"],
            "id": item["id"],
            "resource": item["resource"],
            "filename": item["filename"],
            "url": item["url"],
            "license": item["license"],
            "description": item["description"],
            "size_bytes": size_bytes,
            "sha256": sha,
            "records": records,
            "columns": cols,
            "read_only": True
        }
        manifest_records.append(rec)
        print(f"Processed: {item['filename']} | {records:,} records | {size_bytes:,} bytes | SHA-256: {sha[:16]}... (Read-Only: OK)")

    # Add curated EU CosIng Annex VI UV filters
    eu_rec = add_curated_eu_cosing_annex_vi()
    manifest_records.append(eu_rec)

    # Save manifest and checksums
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    manifest_path = os.path.join(BASE_DIR, "data", "dataset_manifest.json")
    manifest_csv = os.path.join(BASE_DIR, "data", "dataset_manifest.csv")
    checksums_txt = os.path.join(BASE_DIR, "data", "checksums.sha256")
    
    # Also in each group folder
    checksums_A = os.path.join(DIR_GROUP_A, "checksums.sha256")
    checksums_B = os.path.join(DIR_GROUP_B, "checksums.sha256")

    # If checksum files exist, ensure writable before overwriting
    for p in [checksums_txt, checksums_A, checksums_B]:
        if os.path.exists(p):
            try:
                os.chmod(p, stat.S_IWRITE)
                if os.name == "nt":
                    os.system(f'attrib -R "{p}"')
            except Exception:
                pass

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    # Write sha256 checksums files in standard sha256sum format: "<hash>  <filename>"
    with open(checksums_txt, "w", encoding="utf-8") as f_all, \
         open(checksums_A, "w", encoding="utf-8") as f_a, \
         open(checksums_B, "w", encoding="utf-8") as f_b:
        for r in manifest_records:
            line = f"{r['sha256']}  {r['filename']}\n"
            f_all.write(line)
            if r["group"] == "group_A":
                f_a.write(line)
            else:
                f_b.write(line)

    # Make checksum manifests read-only too for security
    for p in [checksums_txt, checksums_A, checksums_B, manifest_path, manifest_csv]:
        make_read_only(p)

    print("\n" + "="*70)
    print("=== DATABASE ACQUISITION SUMMARY ===")
    print("="*70)
    
    total_A_records = sum(r["records"] for r in manifest_records if r["group"] == "group_A" and r["records"] is not None)
    total_B_records = sum(r["records"] for r in manifest_records if r["group"] == "group_B" and r["records"] is not None)
    total_A_bytes = sum(r["size_bytes"] for r in manifest_records if r["group"] == "group_A")
    total_B_bytes = sum(r["size_bytes"] for r in manifest_records if r["group"] == "group_B")

    print(f"Group A (MOST / Photoswitches / Chromophores):")
    print(f"  Files: {sum(1 for r in manifest_records if r['group'] == 'group_A')}")
    print(f"  Total records/molecules: {total_A_records:,}")
    print(f"  Total raw size: {total_A_bytes / 1024 / 1024:.2f} MB")
    print(f"  Path: {DIR_GROUP_A}")

    print(f"\nGroup B (UV Filters / Skin Permeability / Skin Safety):")
    print(f"  Files: {sum(1 for r in manifest_records if r['group'] == 'group_B')}")
    print(f"  Total records/molecules: {total_B_records:,}")
    print(f"  Total raw size: {total_B_bytes / 1024 / 1024:.2f} MB")
    print(f"  Path: {DIR_GROUP_B}")

    print("\nAll files verified, set to READ-ONLY, and recorded in checksums and manifest.")

if __name__ == "__main__":
    main()
