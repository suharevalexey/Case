# Модели-оценщики и генеративные модели (models/)

Данная директория содержит обученные веса моделей-оценщиков свойств, независимых оракулов и конфигураций генеративных моделей в соответствии со структурой Раздела 10 Задания.

## Структура директории

`	ext
models/
├── evaluators/               # Суррогатные модели оценки свойств (с оценкой неопределенности)
│   ├── group_A_absorption_max_nm_evaluator.joblib
│   ├── group_A_log_extinction_evaluator.joblib
│   ├── group_A_quantum_yield_evaluator.joblib (group_A_photochem_efficiency_evaluator.joblib)
│   ├── group_A_log_half_life_evaluator.joblib
│   ├── group_B_log_kp_evaluator.joblib
│   ├── group_B_skin_irritation_evaluator.joblib
│   └── group_B_skin_sensitization_evaluator.joblib
├── oracle/                   # Независимые модели-оракулы (для объективной внешней валидации)
│   ├── group_A_absorption_max_nm_oracle.joblib
│   ├── group_A_log_extinction_oracle.joblib
│   ├── group_A_quantum_yield_oracle.joblib (group_A_photochem_efficiency_oracle.joblib)
│   ├── group_A_log_half_life_oracle.joblib
│   ├── group_B_log_kp_oracle.joblib
│   ├── group_B_skin_irritation_oracle.joblib
│   └── group_B_skin_sensitization_oracle.joblib
└── generative/               # Веса и контрольные точки генеративных моделей (B0 / M1)
    └── README.md
```

## Суррогатные модели (models/evaluators/)
* **group_A_absorption_max_nm_evaluator.joblib**: Random Forest Regressor для длины волны максимума поглощения $\lambda_{max}^{(A)}$ (нм).
* **group_A_log_extinction_evaluator.joblib**: Random Forest Regressor для коэффициента экстинкции $\log_{10} \epsilon$.
* **group_A_quantum_yield_evaluator.joblib**: Random Forest Regressor для квантового выхода фотоизомеризации $\Phi$ ($R^2 = 0.3952$).
* **group_A_log_half_life_evaluator.joblib**: Random Forest Regressor для времени полураспада $\log_{10} t_{1/2}$ ($R^2 = 0.7795$).
* **group_B_log_kp_evaluator.joblib**: Random Forest Regressor для коэффициента кожной проницаемости $\log K_p$ ($R^2 = 0.5658$).
* **group_B_skin_irritation_evaluator.joblib**: Random Forest Classifier для раздражения кожи, ROC-AUC = 0.7106.
* **group_B_skin_sensitization_evaluator.joblib**: Random Forest Classifier для сенсибилизации кожи, ROC-AUC = 0.8590.

## Независимые оракулы (models/oracle/)
Обучены с использованием альтернативных случайных разбиений и архитектур (Gradient Boosting / Extra Trees) для проверки генеративных стратегий без утечки оптимизационного шума (Goodhart\'s law).
