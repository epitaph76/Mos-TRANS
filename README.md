# Mos-TRANS — предиктор задержек транспорта

Цель проекта — заранее предупреждать диспетчера о возможной задержке наземного транспорта: прогнозировать событие за **10–15 минут** по телеметрии NDTP и отклонению от расписания.

## Задача

- Принимать потоковую телеметрию и исторические данные; сопоставлять положение, скорость и состояние дверей ТС с маршрутом и расписанием.
- Вычислять признаки движения, вероятность задержки, ожидаемое отклонение по времени и вероятную причину сбоя.
- Показывать на дашборде положение ТС, риск по маршрутам и карточки инцидентов с рекомендациями диспетчеру.
- Разделить решение на Backend (API и обработка потока) и ML-модуль (обучение и инференс); предусмотреть устойчивую работу при обрывах телеметрии.

## Данные

[Исходный датасет на Яндекс Диске](https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw). Локальная копия находится в `data/dataset.zip` и исключена из Git.

## Сдача

Требуются две версии: CSV-файл в разделе **Data Science** и заполненная форма в разделе **«Загрузка решения»**. Дедлайн обоих разделов — **27 сентября, 23:59 МСК**. В Data Science учитывается лучший результат, в форме — последняя сохранённая версия. Лимит: 36 попыток в день, из них не более 24 успешных.

Рекомендуемый стек: Python 3.12+, PyTorch, CatBoost, Docker; документация API — OpenAPI/Swagger.

## Общее ядро предобработки

На Windows из корня репозитория:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-preprocessing.txt
.\.venv\Scripts\python.exe -m mos_trans.preprocessing --input data/dataset.zip --output data/processed
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Для работы в CatBoost-ветке установите модельный стек в то же окружение:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-catboost.txt
```

## CatBoostClassifier: риск задержки

Классификатор предсказывает, будет ли фактическое отклонение `target_delay_s`
**строго больше 150 секунд (2,5 минуты)**. Значение ровно 150 секунд относится
к классу 0. Модель выдаёт `probability_delay_over_150s` и бинарный класс
с порогом вероятности, выбранным по групповым holdout. Это не прогноз задержки в секундах и его нельзя
сравнивать с MAE регрессора.

В [ноутбуке для Colab](https://colab.research.google.com/github/epitaph76/Mos-TRANS/blob/ya-dolbayob/notebooks/train_catboost_classifier_colab.ipynb)
укажите путь к `dataset.zip` и папке результатов на Drive и запустите ячейки сверху вниз.
Архив копируется в `/content`, затем выполняется общая предобработка. Три
групповых holdout отделяют исходные ТС вместе с их синтетическими копиями;
число деревьев выбирается по реальным точкам holdout. Публикуемый test нужен
только для итоговых метрик. Сохраняются модель, вероятности для test и validate,
метрики, таблица holdout-прогнозов и важности признаков.
Результат проверенного локального запуска также находится в
`artifacts/catboost_classifier/`.

Локальный запуск после установки зависимостей:

```powershell
.\.venv\Scripts\python.exe -m mos_trans.modeling.classifier --input data/processed --dataset data/dataset.zip --output data/catboost-classifier
```

Для инференса из Python используйте
`mos_trans.modeling.classifier.predict(model_path, metrics_path, features)`.
На вход подаются признаки общей предобработки без фактических времён расписания.
Опубликованный test содержит те же ТС и день, поэтому главным показателем
переноса на другие ТС служит групповой holdout.

Метрики бинарной модели: F1, precision, recall, ROC AUC, average precision,
log loss и матрица ошибок. Положительный класс составляет около 17% train;
поэтому одной accuracy для оценки недостаточно. Разница между групповым
holdout и опубликованным test показывает ограниченность test для новых ТС.
В проверенном локальном запуске по групповым holdout выбраны 93 дерева и
порог вероятности 0,30. Средний F1 этих holdout — 0,563 (он несколько
оптимистичен, поскольку на них же выбирался порог). На опубликованном test:
F1 0,619, precision 0,632, recall 0,606, ROC AUC 0,879 и AP 0,729.

## Сравнение CatBoost, LightGBM и движения по маршруту

[Ноутбук Colab](https://colab.research.google.com/github/epitaph76/Mos-TRANS/blob/ya-dolbayob/notebooks/compare_risk_colab.ipynb)
сравнивает CatBoost, LightGBM, квантильный прогноз, Optuna-вариант и ансамбль для вероятности задержки строго больше
150 секунд. Для каждого из 13 исходных ТС он обучает модели без этого ТС и
его синтетических копий, а метрики считает только на реальных точках. Главная
метрика выбора — average precision (AP); порог для F1 выбирается на тех же
out-of-fold прогнозах, поэтому такой F1 несколько оптимистичен.
Один из вариантов LightGBM не использует `tr_id`, чтобы снизить зависимость
от идентификаторов обучающих ТС при проверке на новых ТС.

Дополнительные признаки включают продвижение за 2, 5 и 10 минут, длительность
стоянки и расстояние до ближайшей плановой остановки. Расстояние «по маршруту»
приближено отрезками между плановыми остановками: дорожной геометрии в данных
нет. Пакет учитывается лишь после его `event_time` **и** `receive_time`;
фактическое время прибытия не используется. Вероятностный вариант обучает
квантили 10/50/90% остаточной задержки и интерполирует вероятность превышения
150 секунд. Эта вероятность приблизительная, её калибровка не гарантирована.

Локально:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-risk.txt
.\.venv\Scripts\python.exe -m mos_trans.modeling.compare_risk --dataset data/dataset.zip --processed data/processed --cache data/progress-cache --output data/risk-comparison
```

В папке результата сохраняются `metrics.json`, прогнозы для каждого отложенного
ТС, test и validate, а также `selected_model.joblib`. После подготовки признаков
вызов `mos_trans.modeling.compare_risk.predict(model_path, features)` возвращает
вероятности. В Colab архив читается локально из `/content`, результаты и кэш
признаков лежат на Drive. Test содержит тот же день и ТС, поэтому не заменяет
проверку переноса на другие дни и маршруты.
Файлы проверенного локального запуска находятся в `artifacts/risk_comparison/`.
В обновлённом запуске LightGBM без `tr_id` получил AP 0,571 на 13 отложенных
ТС против 0,557 у LightGBM с `tr_id`. На опубликованном test: F1 0,691,
AP 0,759. Разница AP на отложенных ТС мала относительно разброса между ТС;
при повторной выборке групп 95%-й интервал разницы составляет примерно
−0,008…0,040.

### История отклонения, Optuna и Ordered CatBoost

Дополнительный эксперимент проверяет только прошлые значения `cur_dev_s` того
же ТС: изменение за предыдущие 5 и 10 минут. На 13 группах LightGBM без
`tr_id` остаётся лучше: AP 0,571; с полной историей AP 0,564, только с
изменением за 5 минут 0,558, за 10 минут 0,562. Для применения истории в
потоке нужно хранить предыдущие наблюдения `cur_dev_s`. Будущие точки и
фактическое время прибытия не используются.

Optuna выполняет 25 проб на 10 семействах ТС с внутренними групповыми фолдами;
три семейства не участвуют в подборе. На них AP вырос с 0,445 до 0,476, но
после проверки всех 13 семейств подобранный вариант получил AP 0,546 против
0,571 у текущего LightGBM. Порог 0,20, выбранный на внутренних фолдах,
повысил F1 на этих трёх семействах с 0,361 до 0,451 для базовой модели;
на всех 13 семействах порог 0,30 лучше (F1 0,555 против 0,510). CatBoost
`Ordered` получил AP 0,544 против 0,528 у CatBoost `Plain`, но LightGBM
остался лидером. Итоговая модель и test-прогноз не заменены.

Запустить эти эксперименты локально:

```powershell
.\.venv\Scripts\python.exe -m mos_trans.modeling.history_risk --dataset data/dataset.zip --processed data/processed --output data/risk-history
.\.venv\Scripts\python.exe -m mos_trans.modeling.tune_risk --dataset data/dataset.zip --processed data/processed --output data/risk-tuning --trials 25
.\.venv\Scripts\python.exe -m mos_trans.modeling.compare_risk --dataset data/dataset.zip --processed data/processed --cache data/progress-cache --output data/risk-comparison --tuning-report data/risk-tuning/metrics.json
```

Проверенные таблицы и параметры сохранены в `artifacts/risk_history/`,
`artifacts/risk_tuning/` и `artifacts/risk_comparison/`. Ноутбук Colab запускает
их в том же порядке.

Вместо `data/dataset.zip` можно указать распакованную папку с `train/`, `test/`, `validate/` и `labels/`. Ядро создаёт очищенный `traffic_clean.parquet` и таблицы `train/test/validate_samples.parquet` и `train/test/validate_features.parquet` в папке `data/processed/`, исключённой из Git. Формат и определения всех полей описаны в [FEATURES.md](FEATURES.md). Модельные ветки используют эти выходы и не чистят исходные CSV независимо друг от друга.
