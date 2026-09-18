# Генеративные модели (models/generative/)

* **B0 (Baseline 0)**: Базовый скалярный RL-агент на архитектуре REINVENT4 без Парето (`REINVENT4_MOST_B0_SCALAR_RL_NO_PARETO_3SEEDS_RU.ipynb`). Оптимизирует равновзвешенное геометрическое среднее 7 функций полезности (equal-weight scalarization) при идентичном бюджете (3 сида: 42, 101, 2024; 140 шагов, батч 64 = 26 880 молекул-строк).
* **M1 (Main Method)**: Основной многокритериальный RL-агент REINVENT4 с Парето-ранжированием (`REINVENT4_MOST_3SEEDS_EVALUATOR_ORACLE_ABLATION_RU_COLAB_FIX.ipynb`), штрафами за выход из области применимости (Applicability Domain) и фильтрацией SAScore.
