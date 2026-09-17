import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb_path = os.path.join(BASE_DIR, "MOST_UV_End_to_End_Colab.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell3_src = [
    "# 3. Интерактивная оценка молекулы через PropertyEvaluatorSuite\n",
    "import os, sys\n",
    "from evaluators import PropertyEvaluatorSuite\n",
    "from rdkit import Chem\n",
    "from rdkit.Chem import Draw\n",
    "\n",
    'REPO_DIR = "/content/Case" if os.path.exists("/content") else os.path.abspath(".")\n',
    "suite = PropertyEvaluatorSuite(base_dir=REPO_DIR)\n",
    "\n",
    "# Тестовая молекула: Авобензон (эталонный UV-фильтр)\n",
    'test_smiles = "COc1ccc(C(=O)CC(=O)c2ccc(C(C)(C)C)cc2)cc1"\n',
    "res = suite.evaluate_molecule(test_smiles)\n",
    "\n",
    'print("=" * 85)\n',
    'print(f"  РЕЗУЛЬТАТ ОЦЕНКИ МОЛЕКУЛЫ (7 ЦЕЛЕВЫХ СВОЙСТВ): {test_smiles}")\n',
    'print("=" * 85)\n',
    'print(f"  • Синтетическая доступность (SAScore):  {res[\'synthetic_accessibility\']:.2f} (порог <= 4.5)")\n',
    'print(f"  • Расстояние до домена A (MOST):        {res[\'dist_to_D_A\']:.4f} (AD threshold <= 0.65)")\n',
    'print(f"  • Расстояние до домена B (UV/Skin):     {res[\'dist_to_D_B\']:.4f} (AD threshold <= 0.65)")\n',
    'print(f"  • Группа A - Lambda Max (поглощение):   {res[\'pred_group_A_absorption_max_nm\']:.1f} нм (unc: ±{res[\'unc_group_A_absorption_max_nm\']:.1f}, цель: 290-400 нм)")\n',
    'print(f"  • Группа A - Log Extinction (эпсилон):  {res[\'pred_group_A_log_extinction\']:.2f} (unc: ±{res[\'unc_group_A_log_extinction\']:.2f}, цель: >= 3.5)")\n',
    'print(f"  • Группа A - Quantum Yield (Ф):         {res[\'pred_group_A_photochem_efficiency\']:.3f} (unc: ±{res[\'unc_group_A_photochem_efficiency\']:.3f})\")\n',
    'print(f"  • Группа A - Log t1/2 (время полураспада): {res[\'pred_group_A_log_half_life\']:.2f} log10(c) (unc: ±{res[\'unc_group_A_log_half_life\']:.2f}, цель: >= 3.56 [>=1ч])\")\n',
    'print(f"  • Группа B - Log Kp (проницаемость):    {res[\'pred_group_B_log_kp\']:.2f} см/с (unc: ±{res[\'unc_group_B_log_kp\']:.2f}, цель: <= -6.0)\")\n',
    'print(f"  • Группа B - Сенсибилизация кожи (LLNA): {res[\'pred_group_B_skin_sensitization\']:.4f} (цель: <= 0.50)\")\n',
    'print(f"  • Группа B - Раздражение кожи:          {res[\'pred_group_B_skin_irritation\']:.4f} (цель: <= 0.50)\")\n',
    'print(f"  • Прохождение суррогатных фильтров:     {\'ДА\' if res[\'pass_surrogate_all\'] else \'НЕТ\'}\")\n',
    'print(f"  • Прохождение независимых оракулов:     {\'ДА\' if res[\'pass_oracle_all\'] else \'НЕТ\'}\")\n',
    'print(f"  • Итоговый статус (Overall Pass):       {\'УСПЕХ\' if res[\'pass_constraints\'] else \'ОТСЕВ\'}\")\n',
    'print("=" * 85)\n',
    "\n",
    "mol = Chem.MolFromSmiles(test_smiles)\n",
    "img = Draw.MolToImage(mol, size=(450, 250))\n",
    "display(img)\n"
]

cell5_src = [
    "# 5. Интерактивная генерация новой партии молекул-кандидатов (Стратегия B0)\n",
    "from strategy_b0 import BaselineGeneratorB0\n",
    "\n",
    'REPO_DIR = "/content/Case" if os.path.exists("/content") else os.path.abspath(".")\n',
    'data_dir = os.path.join(REPO_DIR, "data", "processed")\n',
    "\n",
    'print("🔬 Запуск стохастического кроссовера экзоциклических связей фрагментов D_A и D_B...")\n',
    "generator = BaselineGeneratorB0(data_dir=data_dir)\n",
    "new_smiles, attempts, val, uniq = generator.run_generation(seed=42, target_unique_count=50)\n",
    "\n",
    'print("🧪 Оценка сгенерированной выборки через PropertyEvaluatorSuite (7 целевых свойств)...")\n',
    "df_eval = suite.evaluate_batch(new_smiles)\n",
    'pass_n = (df_eval["pass_constraints"] == 1).sum()\n',
    'print(f"Успешно прошли все критерии Оракула и синтетической доступности: {pass_n} из {len(df_eval)} ({pass_n/len(df_eval)*100:.1f}%)\\n")\n',
    "\n",
    'display_cols = ["SMILES", "synthetic_accessibility", "dist_to_D_A", "dist_to_D_B", "pred_group_A_absorption_max_nm", "pred_group_A_log_half_life", "pred_group_B_log_kp", "pass_constraints"]\n',
    "display(df_eval[display_cols].head(10))\n"
]

nb["cells"][3]["source"] = cell3_src
nb["cells"][5]["source"] = cell5_src

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated MOST_UV_End_to_End_Colab.ipynb cleanly!")

