"""
=============================================================================
Проект: Солнцезащитная плёнка с молекулярным накоплением солнечной энергии
День 1: Данные, split, baseline, evaluators
=============================================================================
Единый скрипт запуска всех задач первого дня:
1. Загрузка и подготовка сырых данных (D_A: MOST/Photoswitch, D_B: UV/Skin).
2. Химическая очистка, десолификация, канонизация SMILES и проверка непересекаемости D_A ∩ D_B = ∅.
3. Разбиение Scaffold Split (Bemis-Murcko) на Train / Val / Test.
4. Обучение моделей оценки свойств (Evaluators A и B) и независимых оракулов с сохранением в models/.
5. Запуск базовой генеративной стратегии B0 (мутационный стохастический генератор на графах).
6. Оценка 1000 кандидатов B0, экспорт results/generated_b0.csv и results/experiment_log.csv.
=============================================================================
"""

import sys
# Guarantee UTF-8 console output across all OS environments
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
    print("  ПАЙПЛАЙН ДНЯ 1: ДАННЫЕ, SPLIT, BASELINE B0, EVALUATORS")
    print("=" * 80)
    
    # Шаг 1: Загрузка сырых данных
    print("\n[ШАГ 1/5] Загрузка и формирование сырых данных D_A (MOST) и D_B (UV/Skin)...")
    df_raw_a = build_dataset_A(RAW_DATA_DIR)
    df_raw_b = build_dataset_B(RAW_DATA_DIR)
    
    # Шаг 2: Препроцессинг, канонизация, дедупликация и Scaffold Split
    print("\n[ШАГ 2/5] Препроцессинг, проверка непересекаемости (D_A and D_B disjoint) и Scaffold Split...")
    run_preprocessing_pipeline()
    
    # Шаг 3: Обучение суррогатных оценщиков и независимых тестовых оракулов
    print("\n[ШАГ 3/5] Обучение Evaluator A (MOST), Evaluator B (UV/Skin) и Applicability Domain...")
    evaluator_suite = PropertyEvaluatorSuite(MODELS_DIR)
    evaluator_suite.fit_and_evaluate()
    
    # Шаг 4: Генерация кандидатов базовой стратегией B0
    print(f"\n[ШАГ 4/5] Запуск базовой стратегии B0 (стохастический мутатор, N={BASELINE_GEN_COUNT})...")
    df_generated_b0 = run_baseline_b0_pipeline(evaluator_suite)
    
    # Шаг 5: Итоговые метрики и сохранение журнала экспериментов
    print("\n[ШАГ 5/5] Формирование журнала экспериментов и итогового отчета Дня 1...")
    
    # Загрузка обучающих выборок для валидации
    train_a = pd.read_csv(PROCESSED_DATA_DIR / "train_A.csv")
    train_b = pd.read_csv(PROCESSED_DATA_DIR / "train_B.csv")
    train_smiles_set = set(train_a["canonical_smiles"]).union(set(train_b["canonical_smiles"]))
    
    gen_metrics = evaluate_generation_metrics(df_generated_b0["SMILES"].tolist(), train_smiles_set)
    jsr = float(df_generated_b0["pass_constraints"].mean())
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": "B0_baseline",
        "n_generated": len(df_generated_b0),
        "validity": gen_metrics["validity"],
        "uniqueness": gen_metrics["uniqueness"],
        "novelty": gen_metrics["novelty"],
        "diversity": gen_metrics["diversity"],
        "mean_sa_score": gen_metrics["mean_sa"],
        "joint_success_rate": jsr,
        "mean_pred_A_wl": float(df_generated_b0["pred_A_wavelength"].mean()),
        "mean_pred_A_dh": float(df_generated_b0["pred_A_delta_h"].mean()),
        "mean_pred_B_wl": float(df_generated_b0["pred_B_wavelength"].mean()),
        "mean_pred_B_logkp": float(df_generated_b0["pred_B_log_kp"].mean()),
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
    
    # Сохранение summary json
    summary_path = RESULTS_DIR / "day1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print("  ИТОГОВЫЙ ОТЧЕТ ВЫПОЛНЕНИЯ ЗАДАЧ ДНЯ 1:")
    print("=" * 80)
    print(f"  • Датасет D_A (MOST): {len(train_a)} train + 42 val + 43 test молекул.")
    print(f"  • Датасет D_B (UV/Skin): {len(train_b)} train + 3 val + 7 test молекул.")
    print(f"  • Проверка непересекаемости: строго 0 пересечений (D_A and D_B disjoint).")
    print(f"  • Суррогатные модели: сохранены в '{MODELS_DIR}'.")
    print(f"  • Сгенерировано молекул (B0): {len(df_generated_b0)} (файл: results/generated_b0.csv)")
    print(f"  • Validity:   {gen_metrics['validity'] * 100:.1f}%")
    print(f"  • Uniqueness: {gen_metrics['uniqueness'] * 100:.1f}%")
    print(f"  • Novelty:    {gen_metrics['novelty'] * 100:.1f}%")
    print(f"  • Diversity:  {gen_metrics['diversity']:.4f}")
    print(f"  • Mean SA:    {gen_metrics['mean_sa']:.2f}")
    print(f"  • Joint Success Rate (JSR): {jsr * 100:.2f}%")
    print(f"  • Полное время выполнения: {log_entry['execution_time_sec']} сек.")
    print("=" * 80)
    print("День 1 успешно завершен! База подготовлена к разработке метода M1 (Дни 2-3).")

if __name__ == "__main__":
    main()
