# Last Token · финальное решение кейса Rascar

Мы помогаем оператору кафе выбрать, какие рамки автоматического детектора проверить первыми. На каждый исходный бокс выдаём оценку от 0 до 1. Чем она выше, тем подозрительнее бокс.

## Что команда выбрала для финала

| Отправленный файл | AP 25 октября | AP 26 октября | Kaggle Public AP |
|---|---:|---:|---:|
| [Базовая проверка рамок](submissions/standard_box_check.csv) | 0.92579 | 0.92565 | **0.92465** |
| [Усиленная проверка рамок людей](submissions/enhanced_person_check.csv) | **0.93343** | **0.93435** | 0.92144 |

Оба варианта проверяют все классы объектов. Второй при проверке человека вдали от окна придаёт больше веса тому, где модель видит тело человека.

В папке [submissions](submissions) лежат ровно две выбранные версии. В [коротком отчёте](REPORT.md) объясняем, чем они различаются и как проверяли качество.

## Как повторить результат на CPU

Скачайте репозиторий и откройте его каталог.

~~~sh
git clone https://github.com/reet130/last-token-final.git
cd last-token-final
~~~

1. Распакуйте [датасет организаторов](https://www.kaggle.com/competitions/rascar-hack-case-1/data) в `data/rascar-hack-case-1/`. Для быстрой проверки достаточно пяти файлов рядом друг с другом: `scoring_policy.json`, `train.csv`, `train_frames.csv`, `test.csv`, `sample_submission.csv`. Изображения понадобятся только при повторном запуске детекторов.
2. Создайте отдельное окружение Python 3.13. В Windows PowerShell выполните команды ниже. В Linux и macOS используйте `python3.13 -m venv .venv` и замените `.\.venv\Scripts\python.exe` на `./.venv/bin/python`.

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r finalist/requirements.txt
.\.venv\Scripts\python.exe finalist/reproduce.py
~~~

Если датасет уже скачан, добавьте к последней команде `--data "/path/to/rascar-hack-case-1"`.

Скрипт сопоставит исходные рамки с сохранёнными предсказаниями моделей, рассчитает AP каждого отложенного дня и запишет два CSV в `reproduced/`. Для каждого файла он проверяет порядок и состав всех 362 ID, диапазон оценок и совпадение с отправленной версией с точностью до 10⁻¹². Успех обозначается двумя строками `Verified 362 scores`. Сохранённые ответы в `submissions/` используются только для проверки.

### Как запустить ноутбук

[Ноутбук проверки](operator_review.ipynb) воспроизводит оценки на двух отложенных днях и сверяет обе финальные отправки по ID.

В окружении из предыдущего раздела установите интерфейс Jupyter и библиотеку графиков. Команды для PowerShell из корня проекта:

~~~powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab matplotlib
.\.venv\Scripts\python.exe -m jupyterlab operator_review.ipynb
~~~

В первой ячейке кода укажите папку данных, затем выберите **Run → Run All Cells**. В Linux и macOS используйте `./.venv/bin/python`. Ноутбук обращается к тем же функциям в `finalist/`, что и командная сборка.

## Как повторить прогноз по изображениям без обучения

В отдельной копии репозитория распакуйте полный датасет в `data/rascar-hack-case-1/` и подготовьте GPU-окружение по [TRAINING.md](TRAINING.md). В следующих командах PowerShell `python` должен указывать на это окружение. Из [GitHub Release](https://github.com/reet130/last-token-final/releases/tag/final-models-v1) скачайте три файла в `weights/release/`. Каждый весит 51 301 529 байт. Оба финальных варианта используют одни и те же три детектора.

| Файл релиза | Каталог назначения внутри outputs/redetect_strong/runs/ | SHA256 |
|---|---|---|
| cafe-detector-base.pt | all/weights/last.pt | `4c502f60be1835a51c502b97bcc1355c289274933c6b83ae59b815396885739e` |
| cafe-detector-s1.pt | all_s1/weights/last.pt | `a2fcc02ec57fbd1277eb2d8e121fdc6839f52b9d4ee57700da6474f128443fa0` |
| cafe-detector-s2.pt | all_s2/weights/last.pt | `72a8a252a61ccbfbf55824f195e6a1b806690c96c407889fffbbc59b39da5589` |

Сверьте контрольные суммы с таблицей. Затем разместите веса и запустите три детектора на обычных и зеркальных кадрах.

~~~powershell
Get-FileHash weights/release/*.pt -Algorithm SHA256 | Format-List Path, Hash
foreach ($member in 'base', 's1', 's2') {
    $run = if ($member -eq 'base') { 'all' } else { "all_$member" }
    $dir = "outputs/redetect_strong/runs/$run/weights"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "weights/release/cafe-detector-$member.pt" "$dir/last.pt"
    python finalist/redetect.py predict --sets all --member $member --device 0
    if ($LASTEXITCODE) { throw "predict $member failed" }
    python finalist/redetect.py predict --sets all --member $member --device 0 --flip
    if ($LASTEXITCODE) { throw "flip $member failed" }
}
python finalist/reproduce.py
~~~

Эта команда пересобирает обе отправки с новыми шестью прогнозами и опубликованными признаками экспертов. Дневные AP считаются по сохранённым дневным прогнозам.

Чтобы заново вычислить **также экспертные признаки test** по исходным JPG, скачайте публичные `yolo11x.pt` и `yolo11x-pose.pt` по ссылкам из [отчёта](REPORT.md) в `weights/` и выполните:

~~~powershell
python finalist/expert_veto.py test --rerun --det-device 0 --pose-device 0
if ($LASTEXITCODE) { throw "test expert predictions failed" }
python finalist/reproduce.py
~~~

Файл `submissions/expert_veto.csv`, созданный при этом запуске, содержит промежуточную оценку одних экспертов. Итоговые файлы записываются в `reproduced/`. При смене устройства, версий библиотек или численной точности сравнение с конкурсными оценками может обнаружить расхождения.

## Как устроен репозиторий

- [finalist/reproduce.py](finalist/reproduce.py) собирает оба ответа и считает метрики.
- [outputs/redetect_strong](outputs/redetect_strong) содержит 18 файлов. Шесть прогнозов для каждого из двух отложенных дней и шесть для test. В имени дневного файла указана дата **обучения** детектора.
- [outputs/expert_veto](outputs/expert_veto) содержит предсказания позы и результаты дополнительных проверок на train и test.
- [Инструкция по обучению и инференсу на GPU](TRAINING.md) показывает, как получить эти промежуточные файлы из изображений и публичных весов.
- [История экспериментов команды](https://github.com/lapislasulii/last_token) хранится отдельно.
