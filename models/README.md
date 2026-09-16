# Модели-оценщики и генеративные модели (models/)

Данная директория содержит обученные веса моделей-оценщиков свойств, независимых оракулов и конфигураций генеративных моделей в соответствии со структурой Раздела 10 Задания.

## Структура директории

`	ext
models/
├── evaluators/               # Суррогатные модели оценки свойств (с оценкой неопределенности)
│   ├── group_A_absorption_max_nm_evaluator.joblib
│   ├── group_A_log_extinction_evaluator.joblib
│   ├── group_A_photochem_efficiency_evaluator.joblib
│   ├── group_B_log_kp_evaluator.joblib
│   ├── group_B_skin_irritation_evaluator.joblib
│   └── group_B_skin_sensitization_evaluator.joblib
├── oracle/                   # Независимые модели-оракулы (для объективной внешней валидации)
│   ├── group_A_absorption_max_nm_oracle.joblib
│   ├── group_A_log_extinction_oracle.joblib
│   ├── group_A_photochem_efficiency_oracle.joblib
│   ├── group_B_log_kp_oracle.joblib
│   ├── group_B_skin_irritation_oracle.joblib
│   └── group_B_skin_sensitization_oracle.joblib
└── generative/               # Веса и контрольные точки генеративных моделей (B0 / M1)
    └── README.md
`

## Суррогатные модели (models/evaluators/)
* **group_A_absorption_max_nm_evaluator.joblib**: XGBoost Regressor для длины волны максимума поглощения $\\lambda_{max}^{(A)}$ (нм). Обучен на =31\\,166$ молекулах MOST/хромофоров (^2 = 0.6974$, $\\text{RMSE} = 34.8\\text{ нм}$).
* **group_A_log_extinction_evaluator.joblib**: Random Forest Regressor для коэффициента экстинкции $\\log \\varepsilon$ (^2 = 0.6015$).
* **group_A_photochem_efficiency_evaluator.joblib**: Random Forest Regressor для квантового выхода фотоизомеризации $\\phi$ (^2 = 0.5841$).
* **group_B_log_kp_evaluator.joblib**: Random Forest Regressor для коэффициента кожной проницаемости $\\log K_p$ (см/ч), обучен на базе SkinPiX (^2 = 0.7214$, $\\text{RMSE} = 0.5694$).
* **group_B_skin_irritation_evaluator.joblib**: Random Forest Classifier для раздражения кожи (Acute Dermal Irritation), ROC-AUC = 0.8142.
* **group_B_skin_sensitization_evaluator.joblib**: Random Forest Classifier для сенсибилизации кожи (SARA-ICE / SkinSens), ROC-AUC = 0.8359.

## Независимые оракулы (models/oracle/)
Обучены с использованием альтернативных случайных разбиений и архитектур (Gradient Boosting / Extra Trees) для проверки генеративных стратегий без утечки оптимизационного шума (Goodhart\'s law).
