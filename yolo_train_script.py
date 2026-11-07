from ultralytics  import YOLO

SEED = 42
EPOCHS = 150
PATIENCE = 40 # Можно отключить ранюю остановку
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

if __name__ == "__main__":

    # === init model ===
    model = YOLO("yolov8n.pt")  # или yolov8s.pt и т.д.

    results = model.train(
        data="./data.yaml", 
        epochs=EPOCHS,
        patience=PATIENCE, # early stopping
        imgsz=512,
        batch=BATCH_SIZE,
        workers=2,
        cache="ram",

        project="runs/baseline_v1_0",   # корневая папка для экспериментов
        name="yolov8n_mini_aug_all_bg",          # имя конкретного запуска
        exist_ok=False,                # не перезаписывать, если папка уже есть
        plots=True,

        optimizer="AdamW",
        lr0=LEARNING_RATE, # add lr
        # --- seed ---
        seed=SEED,
        deterministic=True,

        # --- полное отключение аугментаций ---
        # auto_augment=None,
        # hsv_h=0.0,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # degrees=0.0,
        # translate=0.0,
        # scale=0.0,
        # shear=0.0,
        # perspective=0.0,
        # flipud=0.0,
        # fliplr=0.0,
        # mosaic=0.0,
        # mixup=0.0,
        # copy_paste=0.0,
        # erasing=0.0

        # --- аугментации ---
        # цвет: у изображений один канал, оттенок и насыщенность бессмысленны
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, # яркость/контраст: разные аппараты и экспозиции
        
        # геометрия: умеренная
        degrees=10, 
        translate=0.1, 
        scale=0.3, 
        shear=0.0, 
        perspective=0.0,
        fliplr=0.5, 
        flipud=0.0,
        
        # смешивание
        mosaic=0, # склеивает 4 картинки в одну
        close_mosaic=10, 
        mixup=0.0, 
        copy_paste=0.0,
        erasing=0 # затирает кусок изображения
    )