#!/usr/bin/env bash
# Проверка собранного wheel вне исходников, в новом Python env.
# -e: остановиться при ошибке; -u: ошибка при обращении к незаданной переменной;
# pipefail: ошибка любой команды в конвейере считается ошибкой всего конвейера.
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo 'Использование: bash check-wheel.sh [папка_пакета [папка_практики]]' >&2
  exit 2
fi

# По умолчанию проверяем студенческий starter рядом со скриптом.
# BASH_SOURCE[0] — путь к скрипту, поэтому пути не зависят от папки запуска.
# ${1:-...} и ${2:-...} берут аргументы запуска или значения по умолчанию.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
course_package="${1:-$script_dir/starter}"
course_practice="${2:-$script_dir}"

for directory in "$course_package" "$course_practice"; do
  if [[ ! -d "$directory" ]]; then
    echo "Не найдена папка: $directory" >&2
    exit 1
  fi
done
# Сохраняем абсолютные пути: ниже перейдём в другой каталог.
course_package=$(cd -- "$course_package" && pwd)
course_practice=$(cd -- "$course_practice" && pwd)

# Проверяем входы до создания env и загрузки зависимостей.
# Wheel содержит код; model.pkl — обученную модель и vectorizer;
# predictions.csv — исходные прогнозы, с которыми сравним результат установки.
for file in \
  "$course_practice/requirements-lock.txt" \
  "$course_practice/data/inference.csv" \
  "$course_package/dist/taxi_duration_course-0.1.0-py3-none-any.whl" \
  "$course_package/artifacts/model.pkl" \
  "$course_package/predictions.csv"; do
  if [[ ! -f "$file" ]]; then
    echo "Не найден файл: $file" >&2
    echo 'Сначала выполните обучение, предсказание и сборку из README.' >&2
    exit 1
  fi
done

if ! command -v python3.13 >/dev/null 2>&1; then
  echo 'Не найден python3.13. Установите Python 3.13 и повторите запуск.' >&2
  exit 1
fi

# Новый уникальный каталог имитирует передачу пакета другому пользователю.
# Каталог оставляем после проверки, чтобы можно было изучить env и результат.
course_receiver=$(mktemp -d)
echo "Среда получателя: $course_receiver"
# При ошибке покажем строку скрипта и путь к env для разбора причины.
trap 'echo "Проверка прервана на строке $LINENO. Env сохранён: $course_receiver" >&2' ERR
python3.13 -m venv "$course_receiver/venv"
# Используем Python по полному пути — активировать env через source не нужно.
receiver_python="$course_receiver/venv/bin/python"

# Сначала устанавливаем зафиксированные версии библиотек, затем наш wheel.
# --no-deps не меняет зависимости, уже установленные из requirements-lock.txt.
"$receiver_python" -m pip install -r "$course_practice/requirements-lock.txt"
env -u PYTHONPATH "$receiver_python" -m pip install --no-deps \
  "$course_package/dist/taxi_duration_course-0.1.0-py3-none-any.whl"
# Проверяем, что установленные зависимости удовлетворяют требованиям пакетов.
"$receiver_python" -m pip check

# Проверяем установленный пакет, а не импорт из каталога проекта.
# Уходим из проекта и убираем PYTHONPATH, который мог бы подставить исходники.
# <<'PY' передаёт Python многострочный код до строки PY; Bash не подставляет в нём переменные.
cd -- "$course_receiver"
env -u PYTHONPATH "$receiver_python" - <<'PY'
from pathlib import Path
import sysconfig
import taxi_duration

# __file__ показывает, откуда импортирован пакет; ожидаем site-packages нового env.
location = Path(taxi_duration.__file__).resolve()
assert location.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()), location
print("Импорт:", location)
PY

# Запускаем CLI из нового env с прежней моделью и теми же запросами.
# Новый predictions.csv запишется во временный каталог, не поверх исходного.
env -u PYTHONPATH "$course_receiver/venv/bin/taxi-duration" predict \
  --model "$course_package/artifacts/model.pkl" \
  --input "$course_practice/data/inference.csv" \
  --output predictions.csv

# «-» означает чтение Python-кода из stdin; следующий аргумент попадёт в sys.argv[1].
env -u PYTHONPATH "$receiver_python" - "$course_package/predictions.csv" <<'PY'
import sys
import pandas as pd
import numpy as np

# ID читаем как строки: это идентификаторы, а не числовые признаки.
original = pd.read_csv(sys.argv[1], dtype={"ride_id": str})
received = pd.read_csv("predictions.csv", dtype={"ride_id": str})

# Сравнение списков проверяет одновременно ID, количество строк и их порядок.
assert original.ride_id.tolist() == received.ride_id.tolist()
# Числа сравниваем с небольшим допуском: |новое - исходное| <= atol + rtol * |новое|.
# Здесь received передан вторым аргументом и служит опорой для относительного допуска.
np.testing.assert_allclose(
    original.predicted_duration,
    received.predicted_duration,
    rtol=1e-10,
    atol=1e-10,
)
print("Wheel: ID и предсказания совпадают")
PY
