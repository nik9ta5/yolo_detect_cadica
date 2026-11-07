1. Собрать фрейм 

Указать путь до CADICA на машине (до директории `datasets\CADICA`)
```bash
python metaframe_create.py ..\cv_projects\datasets\CADICA
```

2. Сделать разбиение на train/val, перенести данные
```bash
python split_dataset.py .\data\meta_frame.csv
```

3. Запустить обучение
```bash
python yolo_train_script.py
```

4. Оценка
```bash
python yolo_val_script.py
```