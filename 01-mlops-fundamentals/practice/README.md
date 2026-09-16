# Практика 01. От taxi-эксперимента к проверяемому пакету

Задача — прогноз длительности поездки по зонам
отправления и назначения. Предполагаем, что пользователь сообщил назначение
заранее. Фактические расстояние, стоимость и время окончания не являются признаками.

## Что в комплекте

- [01-experiment.ipynb](01-experiment.ipynb) — готовый эксперимент: запускаем и разбираем ключевые решения.
- `data/` — локальные train/validation, запросы без target и манифест SHA-256.
- `starter/` — ваш устанавливаемый пакет с тремя небольшими TODO.
- `tests/` — выданные проверки; свои тесты добавляйте в `starter/tests/test_student.py`.
- `requirements-lock.txt` — проверенный снимок env; `starter/pyproject.toml` — метаданные и сборка пакета.
- `prepare_data.py` — повторная подготовка данных; во время занятия не требуется.

Валидация входов, CLI, сохранение bundle и паспорт запуска уже реализованы.
Полного решения пакета в репозитории нет. Готовый notebook служит подсказкой,
но его упрощённые преобразования не заменяют строгую валидацию пакета.

## Подготовка до занятия

Из `01-mlops-fundamentals/practice`, POSIX:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
cd starter
python -m pip install --no-build-isolation --no-deps -e .
taxi-duration --help
python -m pytest -c pyproject.toml ../tests/test_scaffold.py -q
cd ..
```

Два теста готовой инфраструктуры должны проходить до заполнения TODO.
Полный набор пока ожидаемо падает на `NotImplementedError`; после реализации
заданий все выданные проверки должны стать зелёными. Не меняйте тесты ради статуса.

Комплектный env проверен на Python 3.13.5/macOS. Условие `>=3.11` в metadata
описывает код, а не совместимость этого снимка с Python 3.11. Другие ОС требуют
предварительной проверки. Интернет нужен для установки, но не для заданий.
`--no-deps` используем после установки снимка; `--no-build-isolation` — с уже
установленными setuptools/wheel, а не как универсальное правило сборки.

В редакторе с поддержкой Jupyter выберите ядро `.venv` и откройте notebook из
папки `practice/`. JupyterLab отдельно в снимок не включён; можно использовать
уже настроенный VS Code. В PowerShell команда активации — `.venv\Scripts\Activate.ps1`.

## Маршрут занятия

| Этап | Что делаем |
|---|---|
| Эксперимент | Restart → Run All; разбираем split, признаки и baseline |
| Реализация | Заполняем три TODO в заготовке |
| Проверки | Пишем два своих теста, ловим ошибку, запускаем общий pytest |
| Запуск | Выполняем CLI train/predict, смотрим метрики, паспорт и CSV |
| Передача | Вместе проверяем wheel в отдельной среде; объясняем результат |

Полную самостоятельную реализацию CLI, настройку CI и stress-исследование
не нужно успевать на этой практике.

## Три задания

### 1. Duration и обучающая когорта

В `starter/src/taxi_duration/data.py`, функция `prepare_training_data`:
вход уже скопирован в `result`, timestamps приведены к datetime. Вычислите
`duration` в минутах и оставьте диапазон `[1, 60]` включительно. Невалидные
timestamps должны исключаться, исходный `frame` не должен изменяться.
Подсказка — `labeled_cohort` в notebook.

```bash
# Все следующие команды — из starter/, с активным env.
cd starter
python -m pytest -c pyproject.toml ../tests -q -k training_filter
```

### 2. Fit только на train

В `starter/src/taxi_duration/model.py`, функция `train_model`: определите
`x_train` и `x_validation`, используя созданный `vectorizer` и `prepare_features`.
Словарь признаков обучается только на train; validation использует тот же словарь.

### 3. Предсказание

В `predict` определите `features` из готовых `records` и обученного vectorizer
в bundle. Используйте только transform. Не требуйте target, не фильтруйте строки
и не меняйте порядок. Обработка пустого входа уже реализована.

```bash
python -m pytest -c pyproject.toml ../tests -q -k 'validation_does_not_refit or empty_training'
```

### Два своих теста

Создайте `starter/tests/test_student.py`:

1. Проверьте границы duration на ручных данных: 59 секунд не входят, 60 и 3600 входят, 3601 не входит.
2. Проверьте неизменность исходного dataframe либо предсказание без target с сохранением количества строк.

Ожидаемые значения задайте вручную. Временно внесите ошибку, например исключите
границу 60 минут, убедитесь в падении теста, затем верните корректное условие.

```bash
python -m pytest -c pyproject.toml tests ../tests -q
```

## Запуск результата

Из `starter/` после заполнения TODO:

```bash
taxi-duration train --train ../data/train.parquet \
  --validation ../data/validation.parquet --output artifacts
taxi-duration predict --model artifacts/model.pkl \
  --input ../data/inference.csv --output predictions.csv
python -m build --wheel --no-isolation
```

В `artifacts/` находятся `model.pkl`, `metrics.json`, `run.json`. Откройте метрики:
сравните baseline и модель на одной validation. В паспорте найдите хеши данных,
версии, конфигурацию и команду. При повторном запуске выбирайте новые пути,
например `artifacts-run2` и `predictions-run2.csv`: результаты не перезаписываются.

```bash
python -c 'import pandas as pd, numpy as np; src=pd.read_csv("../data/inference.csv", dtype={"ride_id": str}); out=pd.read_csv("predictions.csv", dtype={"ride_id": str}); assert out.ride_id.tolist()==src.ride_id.tolist(); assert np.isfinite(out.predicted_duration).all(); print(len(out), "строк; ID и ответы проверены")'
```

Wheel содержит код, bundle — обученное состояние. Загружайте pickle только из
собственного доверенного запуска: он может выполнять код. Проверка версии после
загрузки не защищает от вредоносного файла. Используем ту же версию sklearn.

### Совместная проверка wheel и самостоятельное повторение

Этот этап на занятии выполняем вместе. Для самостоятельного повторения нужны
новое окружение и установка зависимостей; сетевое ожидание не входит в задания пары.
Команды для POSIX, начало — в `starter/`, после train/predict/build:

```bash
course_package="$PWD"
course_practice="$(cd .. && pwd)"
course_receiver=$(mktemp -d)
python3.13 -m venv "$course_receiver/venv"
"$course_receiver/venv/bin/python" -m pip install -r "$course_practice/requirements-lock.txt"
env -u PYTHONPATH "$course_receiver/venv/bin/python" -m pip install --no-deps \
  "$course_package/dist/taxi_duration_course-0.1.0-py3-none-any.whl"
(
  cd "$course_receiver"
  env -u PYTHONPATH "$course_receiver/venv/bin/python" -c 'import taxi_duration; print(taxi_duration.__file__)'
  env -u PYTHONPATH "$course_receiver/venv/bin/taxi-duration" predict \
    --model "$course_package/artifacts/model.pkl" \
    --input "$course_practice/data/inference.csv" --output predictions.csv
  env -u PYTHONPATH "$course_receiver/venv/bin/python" -c 'import sys, pandas as pd, numpy as np; a=pd.read_csv(sys.argv[1], dtype={"ride_id": str}); b=pd.read_csv("predictions.csv", dtype={"ride_id": str}); assert a.ride_id.tolist()==b.ride_id.tolist(); np.testing.assert_allclose(a.predicted_duration, b.predicted_duration, rtol=1e-10, atol=1e-10); print("Wheel: ID и предсказания совпадают")' "$course_package/predictions.csv"
)
```

Импорт должен указывать на `site-packages` нового env, не на `src`. Интеграционный
тест CLI из выданного набора подставляет исходники через PYTHONPATH: он проверяет
код, но не заменяет реальное предсказание из установленного wheel.

## Данные, контракты и ограничения

Источник — NYC TLC Green Taxi, январь и февраль 2021 года. В каждом месяце по
10 000 строк до фильтра длительности, seed 2026. URL источников и SHA-256
зафиксированы в `data/manifest.json`. Историческая таблица сама по себе не
доказывает доступность назначения до поездки — это допущение нашей задачи.

- `prepare_features`: две зоны, пропуск → `-1`, выходные ID — строки. Готовая валидация допускает целые 1–265 и `-1`, отклоняет дроби, boolean, бесконечности и некорректные значения.
- `prepare_training_data`: копия размеченных строк с duration 1–60 минут включительно.
- `train_model`: непустые train/validation; validation строго позже train; fit преобразователя только на train.
- Неизвестная допустимая категория игнорируется DictVectorizer. Это техническое поведение, не гарантия качества.
- `predict`: не требует target, сохраняет число и порядок строк; пустой вход даёт пустой массив.
- CLI сохраняет непустые уникальные `ride_id`; при их отсутствии создаёт `row-0`, …, уникальные только внутри файла.
- `inference.csv` — 20 строк февраля без target, пример интерфейса, не независимый test.
- RMSE описывает только историческую когорту 1–60 минут. Отдельного test нет. Отрицательные ответы линейной модели автоматически не обрезаются.

Повторная подготовка — необязательна и выполняется в новый каталог, не во время пары:

```bash
# Из practice/
python prepare_data.py --download --output data-rebuilt
# Или с полными исходными Parquet без сети:
python prepare_data.py --raw-dir /path/to/raw --output data-rebuilt-local
```
