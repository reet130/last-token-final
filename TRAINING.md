# Повторное обучение и расчёт признаков по изображениям

Для проверки уже отправленных файлов достаточно [команды на CPU](README.md). Ниже приведён полный путь, если нужно повторить обучение. Используйте отдельную копию репозитория: инференс записывает результаты в те же пути, что и опубликованные промежуточные прогнозы.

1. Распакуйте весь пакет соревнования в `data/rascar-hack-case-1/`. CSV и JPG должны лежать так, как указано в `train_frames.csv`.
2. Для NVIDIA подготовьте Python со сборкой PyTorch для CUDA. Версии в рабочем окружении команды: Python 3.13.17, torch 2.11.0+cu128, ultralytics 8.3.40. Установите библиотеки из `finalist/requirements.txt` и `python -m pip install ultralytics==8.3.40` в GPU-окружении.
3. Создайте в корне каталог `weights` и положите туда публичные веса под исходными именами. Для обучения использован [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt). Для проверок без дообучения использованы [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) и [YOLO11x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt).

Выполните из корня проекта в PowerShell:

~~~powershell
python finalist/redetect.py build
foreach ($member in 'base', 's1', 's2') {
    python finalist/redetect.py train --member $member --device 0
    if ($LASTEXITCODE) { throw "train $member failed" }
    python finalist/redetect.py predict --member $member --device 0
    if ($LASTEXITCODE) { throw "predict $member failed" }
    python finalist/redetect.py predict --member $member --device 0 --flip
    if ($LASTEXITCODE) { throw "flip $member failed" }
}
python finalist/expert_veto.py train --rerun --det-device 0 --pose-device 0
python finalist/expert_veto.py test --rerun --det-device 0 --pose-device 0
python finalist/reproduce.py
~~~

Каждый запуск train обучает две дневные модели и одну на обоих днях. Итого девять обучений. Для одного только test в команды train и predict можно передать `--sets all`. Использованы seed 0, 1 и 2, разрешение 1280, batch 4, 250 эпох, финальные веса последней эпохи. При обучении учитываем 12 классов, в итоговом файле оцениваем исходные рамки четырёх классов. Псевдоразметка в эти два решения не входит.

Воспроизводимость на CPU проверена на сохранённых прогнозах. При новом обучении версия CUDA, видеокарта и библиотеки могут изменить рамки около порогов. Проверка сравнения с отправленными CSV покажет эти расхождения.
