import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

class ApplicabilityDomainManager:
    """
    Manages Applicability Domain (AD) calculation for disjoint sets D_A and D_B.
    Computes nearest-neighbor Tanimoto distances to training domains D_A and D_B,
    as required by Section 6 of the assignment.
    """
    def __init__(self, smiles_A, smiles_B):
        print("Initializing Applicability Domain Manager...")
        self.fps_A = self._smiles_to_bitvects(smiles_A)
        self.fps_B = self._smiles_to_bitvects(smiles_B)
        print(f"Loaded {len(self.fps_A):,} reference bitvects for D_A and {len(self.fps_B):,} for D_B.")

    def _smiles_to_bitvects(self, smiles_list):
        bvs = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(str(s))
            if m:
                bvs.append(MORGAN_GEN.GetFingerprint(m))
        return bvs

    def compute_distances(self, smiles_or_mol):
        """
        Returns (dist_to_D_A, dist_to_D_B, max_sim_A, max_sim_B)
        Distance = 1.0 - max_Tanimoto
        """
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol
            
        if mol is None:
            return 1.0, 1.0, 0.0, 0.0
            
        fp = MORGAN_GEN.GetFingerprint(mol)
        
        # Max Tanimoto to D_A
        sims_A = DataStructs.BulkTanimotoSimilarity(fp, self.fps_A)
        max_sim_A = max(sims_A) if sims_A else 0.0
        dist_A = 1.0 - max_sim_A
        
        # Max Tanimoto to D_B
        sims_B = DataStructs.BulkTanimotoSimilarity(fp, self.fps_B)
        max_sim_B = max(sims_B) if sims_B else 0.0
        dist_B = 1.0 - max_sim_B
        
        return float(dist_A), float(dist_B), float(max_sim_A), float(max_sim_B)

    def is_in_domain(self, smiles_or_mol, threshold_dist=0.70):
        dist_A, dist_B, _, _ = self.compute_distances(smiles_or_mol)
        in_A = dist_A <= threshold_dist
        in_B = dist_B <= threshold_dist
        return in_A, in_B
