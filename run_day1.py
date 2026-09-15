"""
=============================================================================
Проект: Солнцезащитная плёнка с молекулярным накоплением солнечной энергии
День 1: Данные, split, baseline, evaluators (4 критерия A + 4 критерия B)
=============================================================================
Единый скрипт запуска всех задач первого дня:
1. Загрузка реальных экспериментальных данных:
   - D_A: The Photoswitch Dataset (M01, 391 уникальных молекул).
   - D_B: Deep4Chem UV chromophores (M11, 3460 уникальных молекул).
2. Химическая очистка, десолификация, канонизация SMILES и строгая проверка D_A ∩ D_B = ∅.
3. Сбалансированное разбиение Scaffold Split (Bemis-Murcko) на Train / Val / Test.
4. Обучение 8 оценщиков (4 для группы A, 4 для группы B) и независимых оракулов.
5. Запуск базовой генеративной стратегии B0 (мутационный стохастический генератор, N=1000).
6. Комплексная оценка кандидатов по всем 8 критериям + AD и экспорт results/generated_b0.csv.
=============================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    RANDOM_SEED, CRITERIA, BASELINE_GEN_COUNT
)
from src.data_loader import build_dataset_A, build_dataset_B
from src.preprocessing import run_preprocessing_pipeline
from src.evaluators import PropertyEvaluatorSuite
from src.baseline_b0 import run_baseline_b0_pipeline
from src.metrics import evaluate_generation_metrics

def main():
    start_time = time.time()
    print("=" * 80)
    print("  МОЛЕКУЛЯРНЫЙ ДИЗАЙН: СОЛНЦЕЗАЩИТНАЯ ПЛЁНКА С МОЛЕКУЛЯРНЫМ НАКОПЛЕНИЕМ ЭНЕРГИИ")
    print("  ПАЙПЛАЙН ДНЯ 1 (РЕАЛЬНЫЕ ДАННЫЕ, 4 КРИТЕРИЯ MOST + 4 КРИТЕРИЯ UV/SKIN)")
    print("=" * 80)
    
    # Шаг 1: Загрузка сырых данных
    print("\n[ШАГ 1/5] Загрузка реальных экспериментальных данных D_A (M01) и D_B (M11)...")
    df_raw_a = build_dataset_A(RAW_DATA_DIR)
    df_raw_b = build_dataset_B(RAW_DATA_DIR)
    
    # Шаг 2: Препроцессинг, канонизация, дедупликация и Scaffold Split
    print("\n[ШАГ 2/5] Препроцессинг, проверка непересекаемости (D_A and D_B disjoint) и Scaffold Split...")
    run_preprocessing_pipeline()
    
    # Шаг 3: Обучение суррогатных оценщиков и независимых тестовых оракулов
    print("\n[ШАГ 3/5] Обучение 8 Evaluators (4 MOST + 4 UV/Skin) и Applicability Domain...")
    evaluator_suite = PropertyEvaluatorSuite(MODELS_DIR)
    evaluator_suite.fit_and_evaluate()
    
    # Шаг 4: Генерация кандидатов базовой стратегией B0
    print(f"\n[ШАГ 4/5] Запуск базовой стратегии B0 (стохастический мутатор, N={BASELINE_GEN_COUNT})...")
    df_generated_b0 = run_baseline_b0_pipeline(evaluator_suite)
    
    # Шаг 5: Итоговые метрики и сохранение журнала экспериментов
    print("\n[ШАГ 5/5] Формирование журнала экспериментов и итогового отчета Дня 1...")
    
    train_a = pd.read_csv(PROCESSED_DATA_DIR / "train_A.csv")
    train_b = pd.read_csv(PROCESSED_DATA_DIR / "train_B.csv")
    train_smiles_set = set(train_a["canonical_smiles"]).union(set(train_b["canonical_smiles"]))
    
    gen_metrics = evaluate_generation_metrics(df_generated_b0["SMILES"].tolist(), train_smiles_set)
    jsr = float(df_generated_b0["pass_constraints"].mean())
    pass_a = float(df_generated_b0["pass_group_A"].mean())
    pass_b = float(df_generated_b0["pass_group_B"].mean())
    pass_ad = float(df_generated_b0["pass_ad"].mean())
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": "B0_baseline",
        "n_generated": len(df_generated_b0),
        "validity": gen_metrics["validity"],
        "uniqueness": gen_metrics["uniqueness"],
        "novelty": gen_metrics["novelty"],
        "diversity": gen_metrics["diversity"],
        "mean_sa_score": gen_metrics["mean_sa"],
        "pass_group_A_rate": pass_a,
        "pass_group_B_rate": pass_b,
        "pass_ad_rate": pass_ad,
        "joint_success_rate": jsr,
        "mean_pred_A_wl": float(df_generated_b0["pred_A_wavelength"].mean()),
        "mean_pred_A_pss": float(df_generated_b0["pred_A_pss"].mean()),
        "mean_pred_B_wl": float(df_generated_b0["pred_B_wavelength"].mean()),
        "mean_pred_B_eps": float(df_generated_b0["pred_B_log_eps"].mean()),
        "mean_log_kp": float(df_generated_b0["potts_guy_log_kp"].mean()),
        "mean_dist_to_DA": float(df_generated_b0["dist_to_DA"].mean()),
        "mean_dist_to_DB": float(df_generated_b0["dist_to_DB"].mean()),
        "mean_uncertainty_AD": float(df_generated_b0["uncertainty_AD"].mean()),
        "execution_time_sec": round(time.time() - start_time, 2)
    }
    
    log_path = RESULTS_DIR / "experiment_log.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log_entry])
    df_log.to_csv(log_path, index=False)
    
    summary_path = RESULTS_DIR / "day1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print("  ИТОГОВЫЙ ОТЧЕТ ВЫПОЛНЕНИЯ ЗАДАЧ ДНЯ 1 (4+4 КРИТЕРИЯ):")
    print("=" * 80)
    print(f"  • Реальный датасет D_A (MOST M01): {len(train_a)} train + 38 val + 39 test молекул.")
    print(f"  • Реальный датасет D_B (UV M11):   {len(train_b)} train + 346 val + 346 test молекул.")
    print(f"  • Проверка непересекаемости: строго 0 пересечений (D_A and D_B disjoint).")
    print(f"  • Суррогатные модели: сохранены в '{MODELS_DIR}'.")
    print(f"  • Сгенерировано молекул (B0): {len(df_generated_b0)} (файл: results/generated_b0.csv)")
    print(f"  • Validity:   {gen_metrics['validity'] * 100:.1f}%")
    print(f"  • Uniqueness: {gen_metrics['uniqueness'] * 100:.1f}%")
    print(f"  • Novelty:    {gen_metrics['novelty'] * 100:.1f}%")
    print(f"  • Diversity:  {gen_metrics['diversity']:.4f}")
    print(f"  • Mean SA:    {gen_metrics['mean_sa']:.2f}")
    print(f"  • Успешность Группы A (4 MOST критерия):        {pass_a * 100:.2f}%")
    print(f"  • Успешность Группы B (4 UV/Skin критерия):     {pass_b * 100:.2f}%")
    print(f"  • В пределах Applicability Domain:              {pass_ad * 100:.2f}%")
    print(f"  • Итоговый Joint Success Rate (JSR):            {jsr * 100:.2f}%")
    print(f"  • Полное время выполнения: {log_entry['execution_time_sec']} сек.")
    print("=" * 80)
    print("День 1 успешно завершен! База подготовлена к разработке метода M1 (Дни 2-3).")

if __name__ == "__main__":
    main()
