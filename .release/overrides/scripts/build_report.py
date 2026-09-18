"""Generate documentation from the completed experiment; never invent measurements."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from case.common import TARGETS,sha256,save_json

LABELS={'absorption_max_nm':'Максимум поглощения','log_extinction':'Логарифм экстинкции','pss_fraction':'Доля Z в PSS','log_half_life':'Логарифм времени полураспада','log_kp':'Логарифм проницаемости','skin_sensitization':'Сенсибилизация, HPPT','skin_irritation':'Раздражение, RHE'}
SHORT={'absorption_max_nm':'Поглощение','log_extinction':'Экстинкция','pss_fraction':'PSS','log_half_life':'Время хранения','log_kp':'Проницаемость','skin_sensitization':'Сенсибилизация','skin_irritation':'Раздражение'}
REFERENCES=[
('R1','REINVENT4: официальный исходный код; в проекте используется фиксированный commit ee0d56f4a07472bbb622cd0858184d06f11bff5d.','https://github.com/MolecularAI/REINVENT4/tree/ee0d56f4a07472bbb622cd0858184d06f11bff5d'),
('R2','Joung et al. Experimental database of optical properties of organic compounds. Scientific Data (2020).','https://doi.org/10.1038/s41597-020-00634-8'),
('R3','The Photoswitch Dataset: исходная таблица фотопереключателей и условия измерений.','https://github.com/Ryan-Rhys/The-Photoswitch-Dataset'),
('R4','NICE/ICE: исходные наборы токсикологических измерений.','https://ice.ntp.niehs.nih.gov/'),
('R5','HuskinDB: база экспериментальной проницаемости кожи.','https://www.huskindb.drug-design.de/'),
('R6','SkinPiX: исходные данные и описание.','https://doi.org/10.57745/7FHQOY'),
('R7','Ertl, Schuffenhauer. Estimation of synthetic accessibility score. Journal of Cheminformatics (2009).','https://doi.org/10.1186/1758-2946-1-8'),
('R8','RDKit: описание библиотеки и дескрипторов.','https://www.rdkit.org/docs/'),
('R9','scikit-learn: Random Forest и методы оценки.','https://scikit-learn.org/stable/modules/ensemble.html#forest'),
('R10','XGBoost: документация библиотеки.','https://xgboost.readthedocs.io/'),
('R11','uvvisml: спектральные наборы; повторные записи Deep4Chem исключаются при сборке.','https://github.com/learningmatter-mit/uvvisml'),
]


def write(path,text):
    p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text.rstrip()+'\n',encoding='utf-8')


def table_md(rows):
    return '\n'.join(['| '+' | '.join(map(str,rows[0]))+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows[1:]])


def build_report(draft=False):
    data=json.loads((ROOT/'data/processed/manifest.json').read_text())
    experts=json.loads((ROOT/'results/evaluator_metrics.json').read_text())
    marker=ROOT/'results/experiment.json'
    complete=marker.exists() and json.loads(marker.read_text()).get('status')=='completed'
    if not complete and not draft: raise ValueError('Run pipeline.py run before building the final report')
    experiment=json.loads(marker.read_text()) if complete else {'status':'not_run','device':'not_run','elapsed_seconds':0,'experiments':{}}
    main=pd.read_csv(ROOT/'results/main/per_seed.csv') if complete else pd.DataFrame()
    ablation=pd.read_csv(ROOT/'results/ablation_no_AD/per_seed.csv') if complete else pd.DataFrame()
    generated=pd.read_csv(ROOT/'generated.csv') if complete else pd.DataFrame()
    def score(frame,key,method):
        if frame.empty: return 'не измерено'
        s=frame.loc[frame.method_id==method,key]
        multiplier=1 if key=='internal_diversity' else 100
        digits=4 if key=='internal_diversity' else 2
        return f'{s.mean()*multiplier:.{digits}f} ± {s.std(ddof=1)*multiplier:.{digits}f}'
    metrics=[('validity','Валидность исходного sampling, %'),('uniqueness','Уникальность среди валидных, %'),('novelty','Новизна отобранных, %'),('internal_diversity','Разнообразие'),('JSR_oracle','JSR по точечным прогнозам, %'),('JSR_AD_SA','JSR с AD и SA, %'),('inside_both_AD','Внутри обеих областей AD, %'),('SA_pass','SA ≤ 5, %'),('robust_consensus','Консервативное согласие, %')]
    comparison=[['Показатель','B0','M1']]+[[label,score(main,key,'B0'),score(main,key,'M1')] for key,label in metrics]
    abl=[['Показатель','M1','M1: без AD в награде']]+[[label,score(main,key,'M1'),score(ablation,key,'M1')] for key,label in metrics if key in {'JSR_oracle','JSR_AD_SA','internal_diversity','inside_both_AD'}]
    if complete:
        b=main.loc[main.method_id=='B0','JSR_AD_SA'].mean(); m=main.loc[main.method_id=='M1','JSR_AD_SA'].mean()
        if m>b: conclusion=f'В этом запуске средний JSR с AD и SA у M1 выше B0 на {(m-b)*100:.2f} процентного пункта. Три seed и слабые отдельные оценщики не дают основания объявлять универсальное преимущество.'
        elif m<b: conclusion=f'В этом запуске средний JSR с AD и SA у M1 ниже B0 на {(b-m)*100:.2f} процентного пункта. Гипотеза об улучшении совместного прохождения не подтверждена.'
        else: conclusion='JSR с AD и SA у B0 и M1 одинаков. Гипотеза о повышении совместного прохождения методом M1 не подтверждена.'
        if b==0 and m==0: conclusion+=' Обе стратегии не нашли кандидатов, проходящих строгий вычислительный допуск; это не доказательство физического отсутствия подходящих молекул.'
        new_counts={m:int(g.loc[g.novelty,'SMILES'].nunique()) for m,g in generated.groupby('method_id')}
        sampling_text=f'В основной таблице {len(generated):,} строк: по 1000 уникальных структур на каждый из трёх seed для B0 и M1. Новых уникальных структур по всему объединению seed: '+', '.join(f'{m}: {n:,}' for m,n in new_counts.items())+'.'
    else:
        conclusion='Черновик: результаты генерации ещё не измерены.'; sampling_text=conclusion
    counts=pd.read_csv(ROOT/'data/processed/dataset_combined.csv').source_group.value_counts()
    data_text=f'Пересборка из raw основной ветки даёт {int(counts.sum()):,} молекул: A — {int(counts["A"]):,}, B — {int(counts["B"]):,}. Проверяются 20 исходных файлов. Пересечение идентификаторов A/B равно нулю; стереохимия игнорируется только при контроле пересечения и новизны.'
    targets=[['Свойство / группа','Метки','Ограничение']]
    limits=['290–420 нм','log10 ε ≥ 3,8','PSS ≥ 0,25','log10 t½ ≥ log10 3600','log10 Kp ≤ −5','p(active) ≤ 0,5','p(active) ≤ 0,5']
    for (prop,spec),limit in zip(TARGETS.items(),limits): targets.append([SHORT[prop]+' / '+spec['group'],experts[prop]['n_total'],limit])
    quality=[['Оценщик','Train/cal/test','RF','XGB']]
    for prop,entry in experts.items():
        key='ROC_AUC' if entry['classification'] else 'R2'
        quality.append([SHORT[prop]+(' (AUC)' if entry['classification'] else ' (R²)'),f'{entry["n_train"]}/{entry["n_calibration"]}/{entry["n_test"]}',f'{entry["guidance"][key]:.3f}',f'{entry["oracle"][key]:.3f}'])
    pages=[
      ('1. Постановка и исследовательская гипотеза',[
        ('p','Генеративный дизайн молекул при непересекающейся многокритериальной разметке. Вычислительный прототип для исследования сочетания фотохимических, спектральных и кожных свойств; не готовая солнцезащитная плёнка и не заключение о безопасности.'),
        ('h','Объект и формальная цель'),
        ('p','Проектируем отдельную органическую молекулу в формате SMILES. Даны D_A={(x,y_A)} и D_B={(x,y_B)} с непересекающимися идентификаторами. Требуется стохастически строить новые x с одновременно допустимыми y_A и y_B. Отсутствие совместных меток не позволяет непосредственно проверить совместное распределение свойств на одном веществе.'),
        ('p','Гипотеза: Pareto-объединение семи экспертов в M1 увеличит JSR с фильтрами области применимости и синтетической доступности по сравнению со скалярным B0, сохранив разнообразие. Условие возможного неуспеха заранее явно задано: химические области редких фотохимических и кожных меток почти не пересекаются либо оценщики не переносятся на новые каркасы.'),
        ('h','Почему выбрана эта архитектура'),
        ('p','Рассмотрены три семейства: единая условная нейросеть с масками отсутствующих меток; независимо обученные property-эксперты и направленная генерация; эволюционный поиск по структурам с Pareto-отбором. Выбран второй вариант: REINVENT4 даёт предобученную грамматику SMILES, а отдельные эксперты допускают неодинаковую разметку и проверяемые границы применимости [R1].'),
        ('p','В проекте ровно две генеративные стратегии: B0 — scalar RL, M1 — Pareto RL. Случайные леса и XGBoost — оценщики, а не дополнительные генераторы. Абляция меняет настройку M1 и хранится отдельно, сохраняя method_id=M1.'),
        ('h','Приоритет требований'),
        ('p','Основание: предоставленное задание, разделы 1–10. Минимум 1000 новых структур, одинаковый бюджет сравнения, независимая стадия финальной оценки, неопределённость, AD, абляция и воспроизводимость. Контекстная презентация уточняет физические ограничения. Создание презентации исключено по указанию пользователя.'),
      ]),
      ('2. Данные, цели и происхождение меток',[
        ('p',data_text),('table',targets),
        ('p','Группы A/B — принятая в этом вычислительном эксперименте группировка меток, а не классификация баз данных по префиксам. Спектральные наблюдения не превращают все структуры A в доказанные MOST-фотопереключатели.'),
        ('h','Ключевое исправление семантики'),
        ('p','PSS извлекается исключительно из поля Z PhotoStationaryState таблицы M01 и делится на 100. Это стационарная доля Z, не квантовый выход. Из Deep4Chem используются спектральные величины, но не флуоресцентный выход и не fluorescence lifetime как показатели накопления энергии [R2–R3]. Время полураспада вычисляется только из положительной конечной скорости обратной изомеризации.'),
        ('p','Для кожи сохранены раздельные типы наблюдений: проницаемость HuskinDB/SkinPiX; активность HPPT для сенсибилизации; RHE irritation Call для раздражения. Не смешиваются любые доступные токсикологические тесты в один бинарный endpoint [R4–R6].'),
        ('p','В measurements.csv.gz сохранены исходный файл, SHA-256, строка, endpoint, условия, ссылка и решение об использовании. Повторные наблюдения регрессии агрегируются медианой; для классификации any-active — консервативное правило. Это теряет часть информации об условиях, поэтому модель не считается предсказателем конкретной температуры или матрицы.'),
        ('p','Raw сохраняется read-only после проверки хэшей. Восстановление CRLF разрешено только тогда, когда в точности воспроизводит уже записанный SHA-256; ожидаемые хэши не подгоняются к новым байтам. Повторная сборка не усредняет заново существующие processed-значения.'),
      ]),
      ('3. Оценщики, разбиения и неопределённость',[
        ('p','Представление: Morgan radius=2, 1024 бита и 12 RDKit-дескрипторов. Для каждого свойства сохранены scaffold-группы train/calibration/test примерно 60/20/20. Ациклические структуры разделяются по идентичности — это слабее полноценной проверки на новом химическом каркасе. Кэш связан с SHA-256 таблицы, порядком строк и версией RDKit.'),
        ('table',quality),
        ('p','RF используется в награде генератора; XGBoost — на финальной стадии. Они обучены на одинаковых train-метках, поэтому oracle — межмодельная proxy-проверка, а не независимая экспериментальная истина. Низкий или отрицательный R² не скрывается и ограничивает доверие к результатам.'),
        ('h','Границы применимости'),
        ('p','Для каждого эксперта AD рассчитывается только по его размеченным train-структурам: d=1−max Tanimoto. dist_to_D_A и dist_to_D_B — худшее расстояние среди обязательных экспертов соответствующей группы. Допуск требует оба расстояния ≤0,70. Наличие общей спектральной близости не заменяет близость к редким кинетическим меткам.'),
        ('p','unc_* — разброс деревьев RF, эвристическая неопределённость. Отдельная calibration-часть задаёт split-conformal интервалы для регрессии и множества классов для классификации. При недостатке calibration-данных надёжное подтверждение не присваивается. Покрытие маргинально и не гарантируется одновременно по семи свойствам при адаптивной генерации или domain shift.'),
      ]),
      ('4. Генерация и контролируемый эксперимент',[
        ('table', [['Параметр на метод / seed','B0','M1'],['RL-шаги × batch','140 × 64','7 × 20 × 64'],['Максимум строк награды','8960','8960'],['Sampling до фильтров','3500','7 × 500'],['Финальная оценка','1000','1000'],['Seed','42, 101, 2024','42, 101, 2024']]),
        ('p','Одинаковые prior, данные, оценщики, DAP-обучение, learning rate, diversity filter и верхний бюджет. Время wall-clock может различаться из-за семи процессов M1; фактические секунды и число запросов сохраняются в журнале. Проверенный путь использует CPU; GPU-эквивалентность чисел не предполагается.'),
        ('h','Функции объединения'),
        ('p','Нормированный направленный отступ свойства уменьшается на β·σ, β=0,5, и переводится sigmoid в полезность. Для PSS больше лучше, для вероятности неблагоприятного класса меньше лучше: физические границы [0,1] не создают второй конкурирующий оптимум. B0 максимизирует геометрическое среднее полезностей.'),
        ('p','M1 ранжирует по недоминируемым фронтам. Награда R=1/[1+r+0,5(1−t)], где r — Pareto-ранг, t∈[0,1] — приоритет внутри фронта. Диапазоны соседних фронтов не пересекаются. Кэш награды по SMILES отключён: Pareto-ранг зависит от текущего batch, поэтому хранить прежний ранг как постоянное свойство молекулы нельзя.'),
        ('p','В награде используется мягкий штраф AD, в финальном допуске — неизменяемый жёсткий AD. SA≤5 является химическим эвристическим фильтром, не доказательством возможности синтеза [R7]. Выполненная абляция M1 отключает AD только во время обучения; финальная проверка остаётся той же.'),
        ('p','Генерация — sampling обученного авторегрессионного агента, не сортировка исходной базы. Все пулы, включая абляцию, формируются и фиксируются до загрузки oracle. Дубликат выбирается по первому появлению, подвыборка — по SHA256(seed:SMILES), без подбора по итоговому ответу.'),
      ]),
      ('5. Измеренные результаты и вывод',[
        ('p',sampling_text),('table',comparison),
        ('p','Среднее ± выборочное стандартное отклонение по трём seed. Валидность и уникальность вычислены из полного sampling до фильтров; новизна, разнообразие и JSR — по заранее отобранным 1000 кандидатам каждого seed. Разнообразие использует все неупорядоченные пары fingerprints. Знаменатели не смешиваются.'),
        ('table',abl),('p',conclusion),
        ('p','Все числа получены из results/main/per_seed.csv и results/ablation_no_AD/per_seed.csv этого запуска. Изменение одной настройки на уже готовом пуле не считается обучающей абляцией. Результаты короткого smoke-теста не включаются в сравнительную таблицу.'),
      ]),
      ('6. Ограничения, ошибки и воспроизводимость',[
        ('h','Что исправлено и что доказано'),
        ('p','Единая реализация B0/M1 заменяет дублирующиеся маршруты расчёта. Проверяются raw, processed, split, модели и замороженные кандидаты. Исправлены семантика PSS, направления полезностей, сохранение порядка Pareto-фронтов, batch-зависимый кэш, источники AD и выбор дубликатов. Пустые и невалидные входы получают явный статус, а не успешный результат.'),
        ('p','Выявленная несовместимость внутреннего checkpoint-хэша устранена переносимым digest сети, словаря и метаданных. Официальный prior принимается только после проверки полного SHA-256; тест проверяет сохранение/загрузку и обнаружение изменения веса. Полная контрольная сумма файла хранится дополнительно. Повторные неуспешные конфигурации описаны в журнале, но не подмешиваются к итоговой статистике.'),
        ('h','Пределы научного вывода'),
        ('p','Без независимых совместных экспериментальных меток нельзя оценить реальную вероятность совместной пригодности. Редкие PSS, кинетические и RHE-наборы дают малые holdout и слабые переносимые модели. Большая предсказанная величина вне AD не становится подтверждением. Новизна проверена относительно курированного корпуса проекта, а не относительно всего предобучающего корпуса prior.'),
        ('p','Один SMILES и максимум поглощения не предсказывают SPF плёнки. Нет измеренной ΔH_storage, циклирования, фотопродуктов, совместимости с матрицей, реального тепловыделения и продуктовой безопасности. Для реальной химической проверки нужны синтез, спектры обоих состояний, фотокинетика/PSS, калориметрия, контроль фотопродуктов и испытания плёнки. Здесь отбираются гипотезы для следующего исследования, не вещества для нанесения на кожу.'),
        ('h','Повторение и сохранение версии'),
        ('p','Python 3.13; установить requirements.txt; запуск: python pipeline.py run --device cpu. Быстрая проверка опубликованного результата: python pipeline.py verify --complete. REINVENT устанавливается в отдельное зафиксированное окружение; полный повторный запуск требует доступа к официальному prior. Данные и веса оценщиков включены в репозиторий, checkpoints генераторов и подробные логи приложены к релизу.'),
        ('p','Предыдущий снимок main сохранён в archive/main-before-v3-20260918. Текущая версия основана только на том снимке main. Полные ответы на 14 вопросов — DEFENSE_GUIDE.md; схема полей — docs/SCHEMA.md; первичные ссылки — docs/SOURCES.md; манифесты и журнал входят в комплект.'),
      ]),
    ]
    md='# B0 и M1: исследовательский отчёт\n\n'
    for title,blocks in pages:
        md+='## '+title+'\n\n'
        for kind,value in blocks:
            md+=(table_md(value) if kind=='table' else ('### '+value if kind=='h' else value))+'\n\n'
    write('docs/REPORT.md',md)
    sources='# Источники и атрибуция\n\nТочная байтовая база текущего эксперимента — data/dataset_manifest.json и data/processed/manifest.json. Внешние страницы приводятся для происхождения и определений; динамические сайты не заменяют сохранённый snapshot.\n\n'
    sources+='\n\n'.join(f'**[{key}]** {title}\n\n{url}' for key,title,url in REFERENCES)
    sources+='\n\nПрикладные ограничения основаны также на предоставленных документах: «Задание», разделы 1–13; контекстная презентация, страницы 6, 9, 11–13, 16, 19–25. Их оригиналы не перепубликуются. Лицензии сторонних данных и моделей сохраняют силу; общая лицензия на все raw-файлы не заявляется.'
    write('docs/SOURCES.md',sources)
    qa=[
('Как формально определена задача?','Даны два непересекающихся по идентичности множества молекул с разными наблюдаемыми метками. Требуется генератор новых SMILES, для которых прогнозы всех заданных свойств одновременно удовлетворяют ограничениям. Формальная цель условной генерации задаёт исследовательскую задачу, а не доказанную факторизацию неизвестного совместного распределения.'),
('Почему отсутствие совместных меток создаёт проблему?','Нельзя напрямую обучить и проверить совместную зависимость y_A и y_B на одной молекуле. Независимые ошибки экспертов не предполагаются доказанными. Перенос между химическими областями проверяется ограниченно: scaffold-holdout, разброс моделей, calibration и отдельный AD для каждой цели.'),
('Какие минимум три семейства решений рассматривались?','Единая условная модель с masked loss; раздельные эксперты и управляемая генерация; эволюционный поиск по химическим структурам. Первое семейство требовало бы больше совместной или хорошо переносимой разметки; третье требует химически корректных операторов изменения структуры. Раздельные эксперты с предобученным SMILES-агентом дали проверяемую реализацию в доступном бюджете.'),
('Почему выбрана конкретная архитектура?','REINVENT4 предоставляет генеративный prior, а RF-эксперты работают с неодинаково размеченными таблицами без заполнения неизвестных меток фиктивными значениями. B0 и M1 используют один движок, что позволяет проверять прежде всего способ объединения критериев. Выбор не основан на сложности или новизне нейросети.'),
('Где происходит генерация, а где только оценка и отбор?','Стохастическая генерация происходит в обученном агенте REINVENT при построении последовательности токенов SMILES. case.models вычисляет прогнозы; case.reward объединяет guidance-полезности; case.metrics фиксирует и оценивает выборку. Ранжирование готовой таблицы само по себе генерацией не называется.'),
('Каким образом объединяются свойства A и B?','B0 использует геометрическое среднее семи направленных полезностей. M1 строит недоминируемые фронты тех же полезностей и применяет функцию с непересекающимися диапазонами рангов; приоритет свойства действует внутри фронта. Штрафы неопределённости и AD заранее фиксированы. Final pass требует все oracle-пороги, AD обоих блоков и SA.'),
('Какие статистические предположения делает объединение?','Предполагается ограниченная переносимость отдельных экспертов в их обучающих областях. Произведение полезностей не объявляется произведением независимых вероятностей. Calibration-интервалы опираются на допущение обменности и дают маргинальные, а не совместные гарантии; при адаптивном поиске это допущение может нарушаться.'),
('Что происходит, если пространства почти не пересекаются?','Жёсткий совместный AD может отклонить все кандидаты. Не ослабляем пороги после просмотра результатов, чтобы искусственно повысить JSR. Указываем нулевое или малое пересечение как отрицательный результат и планируем новые совместные измерения в пограничной области.'),
('Как контролируется выход за область применимости?','Каждый эксперт имеет только собственные размеченные train-reference. Расстояние равно 1−max Tanimoto; по группе берётся худшее расстояние. Во время обучения используется мягкий штраф, на финальной стадии неизменный порог 0,70. Это эвристическое ограничение, не гарантия корректности прогноза.'),
('Как предотвращаются одинаковые и почти одинаковые кандидаты?','В RL работает один и тот же Murcko-scaffold diversity filter. M1 использует разные детерминированные seed по профилям и стадиям. Sampling сохраняет дубликаты для честного знаменателя uniqueness; затем canonical-дедупликация и отбор по заранее заданному хэшу формируют финальный набор. Почти одинаковые структуры дополнительно отражаются в парном Tanimoto-разнообразии.'),
('Какие компоненты действительно дали улучшение?',conclusion+' Причинный вклад AD проверяется отдельным переобучением M1 без AD в награде. Тесты корректности функций не являются измерением прироста качества генерации; само исправление кода не доказывает улучшение химии.'),
('Какие отрицательные результаты получены?','Данные реального scaffold-holdout показывают слабый перенос части редких регрессионных целей, включая PSS и кинетику. Генеративные отрицательные результаты и отличия абляции представлены в отчёте и CSV без ручного выбора удачных seed. '+conclusion),
('В каких случаях метод не должен использоваться?','Не использовать для доказательства безопасности вещества, выбора готового средства для кожи, расчёта SPF или запаса энергии плёнки. Не доверять отдельному высокому score при выходе за AD, отсутствии provenance, слабом evaluator или изменённой цели. Не запускать произвольные непроверенные pickle/joblib checkpoints.'),
('Что требуется для реальной химической проверки?','Нужны совместные измерения обеих групп свойств на одних веществах, независимый внешний evaluator и новый holdout. Затем синтезируемость/ретросинтез, спектры исходного и заряженного состояний, измерение фотоизомеризационного выхода отдельно от PSS, скорости при заданных температурах, ΔH_storage, циклы, фотопродукты, совместимость с матрицей и профильные испытания кожи/плёнки.')]
    write('DEFENSE_GUIDE.md','# Ответы на 14 вопросов задания\n\n'+ '\n\n'.join(f'## {i}. {q}\n\n{a}' for i,(q,a) in enumerate(qa,1)))
    readme='# B0 и M1: генеративный молекулярный дизайн\n\nВерсия **3.0.0**. Ровно две стратегии: **B0 — scalar REINVENT4 RL**, **M1 — Pareto REINVENT4 RL**. Это вычислительное исследование, не подтверждение MOST-функции или безопасности готовой плёнки.\n\n'
    readme+='## Запуск\n\nПроверенная платформа: Linux x86-64, Python 3.13, CPU. После `python -m pip install -r requirements.txt` полный цикл данных, обучения, генерации, абляции и отчёта запускается одной командой:\n\n```bash\npython pipeline.py run --device cpu\n```\n\nПроверка сохранённой версии без нового обучения:\n\n```bash\npython pipeline.py verify --complete\npython -m pytest -q\n```\n\nПервый полный запуск требует интернета для официального REINVENT4/prior и установки отдельного окружения. GPU-путь требует совместимого PyTorch; опубликованные числа получены на CPU. `python pipeline.py plan` выводит бюджет. `python pipeline.py reproduce` заново строит данные и экспертов и переоценивает сохранённые SMILES: это **не** повторная генерация.\n\n'
    readme+='## Результаты текущего контролируемого запуска\n\n'+sampling_text+'\n\n'+table_md(comparison)+'\n\n'+conclusion+'\n\nВсе проценты, кроме разнообразия, указаны как среднее ± SD по трём seed. Полные знаменатели: `results/main/per_seed.csv`. Абляция M1: `results/ablation_no_AD/`. Прогнозы final oracle не являются экспериментально установленными свойствами.\n\n'
    readme+='## Данные и оценщики\n\n'+data_text+'\n\n'+table_md(targets)+'\n\nPSS не называется квантовым выходом. Условия неизвестной температуры не заменяются комнатной температурой. Для каждого свойства train/calibration/test и AD отдельные. Низкие R² сохраняются в `results/evaluator_metrics.json`, а не скрываются.\n\n'
    readme+='## Структура\n\n```text\ncase/                 единые данные, эксперты, награды, генерация и метрики\nscripts/              установка движка, проверенный эксперимент, отчёт\ndata/raw/             20 исходных файлов с SHA-256\ndata/processed/       две группы, combined, measurement ledger, manifest\nmodels/               RF guidance, XGB oracle, splits и train-reference\ninputs/               полные новые sampling-пулы, включая невалидные строки\nresults/main/         B0/M1, по 3 seed и 1000 оценённых структур\nresults/ablation_no_AD/ M1 без AD в награде; final AD неизменен\ngenerated.csv         основной финальный набор\ndocs/REPORT.pdf       отчёт на 6 страницах\nDEFENSE_GUIDE.md      ответы на все 14 вопросов\n```\n\n'
    readme+='## Документация и артефакты\n\n[Отчёт](docs/REPORT.pdf) · [Текст отчёта](docs/REPORT.md) · [Ответы](DEFENSE_GUIDE.md) · [Данные](docs/DATA_CARD.md) · [Схема CSV](docs/SCHEMA.md) · [Журнал](docs/EXPERIMENT_LOG.md) · [Соответствие заданию](docs/COMPLIANCE.md) · [Источники](docs/SOURCES.md).\n\nВеса обученных генераторов и подробные логи публикуются как assets релиза `v3.0.0`: `B0_M1_checkpoints.zip` и `run-evidence.zip`. Манифест внутри архива содержит SHA-256 каждого checkpoint. Исходный снимок `main` сохранён в ветке `archive/main-before-v3-20260918`; источником новой версии служил только этот снимок.\n'
    write('README.md',readme)
    write('docs/DATA_CARD.md','# Карточка данных\n\n'+data_text+'\n\n'+table_md(targets)+'\n\n'+ '\n\n'.join(text for kind,text in pages[1][1] if kind=='p')+'\n\n'+sources)
    write('docs/MODEL_CARD.md','# Карточка оценщиков\n\n'+table_md(quality)+'\n\n'+ '\n\n'.join(text for kind,text in pages[2][1] if kind=='p')+'\n\nПоложительный класс — active в конкретном endpoint. Эти вероятности не являются вероятностями общего вреда для человека. Внешний научный evaluator может заменить межмодельную proxy-проверку без изменения замороженных кандидатов.')
    schema='# Схема generated.csv\n\nОдна строка — кандидат для метода и seed, а не доказанная экспериментальная молекула. Ключ основной таблицы: `(method_id, seed, SMILES)`. Между seed совпадения допустимы и отдельно учитываются в числе новых уникальных структур.\n\n'
    fields=[['Поле / семейство','Смысл'],['SMILES','Канонический крупнейший органический фрагмент; стереохимия сохраняется'],['method_id / seed / variant','Только B0/M1; основной random seed; none либо no_AD'],['valid / novelty','Валидный кандидат / отсутствие stereo-insensitive identity в обучающем корпусе проекта'],['pred_* / unc_*','Прогноз RF / разброс деревьев RF, не калиброванный доверительный интервал'],['oracle_*','Финальный XGBoost proxy; не использован в награде'],['guidance_*_lower/upper, oracle_*_lower/upper','Калиброванные маргинальные интервалы регрессионных целей'],['*_set0 / *_set1','Принадлежность класса 0/1 к conformal-множеству'],['dist_*','Расстояние до размеченного train-reference конкретного эксперта'],['dist_to_D_A / dist_to_D_B','Максимум расстояний обязательных экспертов группы'],['synthetic_accessibility','SA-score, меньше — более простая эвристическая структура'],['pass_guidance / pass_oracle','Одновременное выполнение точечных ограничений соответствующей моделью'],['pass_AD_SA','Обе групповые AD ≤0,70 и SA ≤5'],['pass_constraints','pass_oracle и pass_AD_SA'],['pass_robust_consensus','Обе модели проходят калиброванные границы/классовые множества, точечные ограничения и AD/SA'],['status','invalid / guidance_only / proxy_pass / proxy_fail'],['final_evaluator_used','Была ли загружена финальная модель'],['frozen_input_order','Позиция в зафиксированном входном пуле']]
    write('docs/SCHEMA.md',schema+table_md(fields)+'\n\nЕдиницы и пороги определены только в `case/common.py`. Поля, не определённые для невалидной строки, пустые; флаги успешности для неё False. `ΔH_storage`, SPF и безопасность плёнки не выдаются как прогнозы.')
    tests=ROOT/'results/test_results.xml'; tested='не зафиксировано'
    if tests.exists():
        tree=ET.parse(tests).getroot(); suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
        tested=str(sum(int(s.get('tests',0)) for s in suites))
    log='# Журнал экспериментов\n\n## До фиксации финальной конфигурации\n\nПроверена целостность raw основной ветки; разрешено только доказуемое восстановление исходных переводов строк. Реальные оценщики обучены на пересобранной таблице. При подключении движка выявлена несовместимость внутреннего checkpoint-хэша; запуск останавливался, пока не добавлены аутентификация байтов prior и переносимый digest с round-trip/tamper-тестом.\n\nПри проверке кода выявлены недопустимый кэш batch-зависимой Pareto-награды и смешение физических границ вероятностей с направлением оптимизации. Кэш отключён для обеих стратегий, функции полезности исправлены до итогового запуска. Результаты промежуточных конфигураций не используются как итоговые.\n\n## Зафиксированный опыт\n\n'
    log+=f'Статус: `{experiment["status"]}`. Устройство: `{experiment["device"]}`. Run ID: `{experiment.get("run_id","not_run")}`. Измеренное время контролируемой стадии: {experiment["elapsed_seconds"]/3600:.3f} ч; установка и отдельный smoke могут не входить в этот таймер. Число тестов в сохранённом JUnit: {tested}.\n\n'
    log+=table_md(comparison)+'\n\n'+table_md(abl)+'\n\n'+conclusion+'\n\nПодробные TOML, seed по профилям, query-count, wall-clock, checkpoint SHA и stdout/stderr находятся в `run-evidence.zip` релиза. `results/experiment.json` содержит source hashes и модельный manifest hash. Smoke — тест работоспособности, не научная выборка. Повторный выбор удачных seed и перенастройка порогов по финальному oracle не выполнялись.'
    write('docs/EXPERIMENT_LOG.md',log)
    compliance=[['Требование задания','Реализация / граница'],['§1–4: постановка, гипотеза, ≥3 семейства','REPORT, страницы 1–2; DEFENSE_GUIDE, вопросы 1–7'],['§2, §10: ≥1000 новых структур','generated.csv и фактические числа новых уникальных структур; не означает 1000 успешных кандидатов'],['§5: B0/M1, одинаковый бюджет','По 8960 максимальных train-строк и 3500 sampling; три seed; wall-clock сохранён отдельно'],['§5: обучающая абляция','M1/no_AD переобучен с тем же бюджетом; final AD неизменен'],['§6: метрики и надёжность','per_seed.csv, summary.csv, calibration и отдельный train-AD'],['§7: независимая стадия','Oracle не загружается до фиксации всех входов; остаётся ограничение общих train-меток'],['§8: вычислительные ограничения','CPU, <30000 curated molecules, фиксированный небольшой prior; фактическое время в журнале'],['§9: ответы','Все 14 ответов в DEFENSE_GUIDE.md'],['§10: воспроизводимость','pipeline.py, два environment-файла, исходные данные, модели, manifest'],['§10: отчёт','docs/REPORT.pdf, 6 страниц; ссылки и журнал отдельно'],['§10: презентация','Не создаётся по прямому указанию пользователя'],['§12: интерпретация результата','Нулевой или отрицательный эффект не подменяется заявлением успеха'],['Химическое подтверждение','Не получено: ΔH, плёночный SPF, фотопродукты и безопасность вне текущих меток']]
    write('docs/COMPLIANCE.md','# Матрица соответствия\n\n'+table_md(compliance))
    write('models/README.md','# Веса и назначение моделей\n\nЗдесь находятся семь RF guidance и семь XGBoost oracle, их train-reference, зафиксированные train/calibration/test и holdout-прогнозы. Это оценщики свойств, не 14 генеративных методов. B0 и M1 используют официальный REINVENT prior, происхождение которого зафиксировано в scripts/setup_engine.py. Обученные generative checkpoints с манифестом SHA-256 находятся в assets релиза v3.0.0. Загружайте pickle/joblib только из доверенного релиза и после проверки контрольной суммы.\n\n'+table_md(quality))
    write('CHANGELOG.md','# Журнал версии\n\n## 3.0.0\n\nЕдиные B0/M1; детерминированная сборка raw→measurements→processed; исправленная семантика PSS и токсикологических endpoints; отдельные scaffold-разбиения и AD; калибровка; строгий порядок Pareto-награды и отключение batch-кэша; новые контролируемые sampling-пулы и обучающая абляция; отчёт, тесты и манифесты. Исходный main сохранён как archive/main-before-v3-20260918. Полноценная экспериментальная валидация молекул не заявляется.')
    # Programmatic six-page report. System-installed fonts are embedded, not distributed as files.
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    paths=[Path('/usr/share/fonts/truetype/dejavu'),Path('/usr/share/fonts/truetype/liberation2')]
    regular=next((p/'DejaVuSans.ttf' for p in paths if (p/'DejaVuSans.ttf').exists()),None)
    bold=regular.with_name('DejaVuSans-Bold.ttf') if regular else None
    if not regular: raise RuntimeError('Install fonts-dejavu-core to render Russian PDF')
    pdfmetrics.registerFont(TTFont('CaseText',str(regular))); pdfmetrics.registerFont(TTFont('CaseBold',str(bold)))
    body=ParagraphStyle('body',fontName='CaseText',fontSize=9.25,leading=13.3,spaceAfter=8)
    heading=ParagraphStyle('heading',fontName='CaseBold',fontSize=16,leading=21,spaceAfter=17)
    sub=ParagraphStyle('sub',parent=body,fontName='CaseBold',fontSize=10.2,leading=14,spaceBefore=6,spaceAfter=6)
    cell=ParagraphStyle('cell',parent=body,fontSize=8.1,leading=10.6,spaceAfter=0)
    def para(text,style=body): return Paragraph(html.escape(str(text)),style)
    story=[]
    for i,(title,blocks) in enumerate(pages):
        if i: story.append(PageBreak())
        story.append(para(title,heading))
        for kind,value in blocks:
            if kind=='table':
                n=len(value[0]); widths=[245,110,110] if n==3 else [185,120,80,80]
                t=Table([[para(x,cell) for x in row] for row in value],colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef3')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#698195')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d4dde4'))]))
                story.extend([t,Spacer(1,11)])
            else: story.append(para(value,sub if kind=='h' else body))
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont('CaseText',8); canvas.setFillColor(colors.HexColor('#52616d'))
        canvas.drawString(65,30,'Case 3.0.0  |  B0 / M1  |  computational proof of concept')
        canvas.drawRightString(530,30,str(doc.page)); canvas.restoreState()
    pdf=ROOT/'docs/REPORT.pdf'
    SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),leftMargin=65,rightMargin=65,topMargin=48,bottomMargin=50,title='B0 and M1 - computational molecular design',author='Case project').build(story,onFirstPage=footer,onLaterPages=footer)
    import fitz
    with fitz.open(pdf) as document:
        if len(document)!=6: raise ValueError(f'Report must be 6 pages; got {len(document)}')
    save_json(ROOT/'results/report_manifest.json',dict(report_sha256=sha256(pdf),pages=6,completed_experiment=complete,experiment_sha256=sha256(marker) if complete else None))
    print('Six-page report and synchronized B0/M1 documentation built')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--draft',action='store_true'); args=p.parse_args(); build_report(args.draft)
