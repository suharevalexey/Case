"""
End-to-End Execution Pipeline for MOST+UV Molecular Design
"""

import os, sys, json, time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    print('=' * 80)
    print('  MOST + UU MOLECULAR DESIGN: ERD-TO-END PIPELINE')
    print('=' * 80)
    from src.strategy_b0 import run_b0_generation
    df, metrics = run_b0_generation(n_candidates=3000, random_seed=42)
    print('Done. Metrics summary;')
    for k, v in metrics.items():
        print(f'  {k:25s}: {v}')

if __name__ == '__main__':
    main()
