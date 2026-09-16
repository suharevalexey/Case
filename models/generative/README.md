# Генеративные модели (models/generative/)

* **B0 (Baseline 0)**: Стохастическая рекомбинация структурных фрагментов (rule-based stochastic graph mutator). Не требует нейросетевых весов, алгоритм реализован в src/strategy_b0.py.
* **M1 (Main Method)**: Генетический алгоритм с Парето-оптимизацией NSGA-II и штрафом за выход из области применимости (Applicability Domain).
