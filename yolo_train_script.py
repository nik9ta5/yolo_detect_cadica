from ultralytics  import YOLO
import argparse


RANDOM_STATE = 42
# EPOCHS = 150
# PATIENCE = 40 # Можно отключить ранюю остановку
# BATCH_SIZE = 32
# LEARNING_RATE = 3e-4


def get_parameters_argparse():
    parser = argparse.ArgumentParser(description="train script")
    parser.add_argument("model_weigths", type=str, help="path 2 model weigths")
    parser.add_argument("path2data_file", type=str, help="path to data.yaml file")
    parser.add_argument("imgsize", type=int, help="image size")
    parser.add_argument("project", type=str, help="project name")
    parser.add_argument("exp_name", type=str, help="experiment name")
    
    parser.add_argument("epochs", type=int, help="num epoch")
    parser.add_argument("patience", type=int, help="num patience")
    parser.add_argument("batch_size", type=int, help="batch_size for train")
    parser.add_argument("lr", type=float, help="learning rate") 
    
    return parser.parse_args()



if __name__ == "__main__":
    
    # --- parameters ---    
    args = get_parameters_argparse()
    model_weigths = args.model_weigths
    path2data_file = args.path2data_file 
    imgsize = args.imgsize
    project = args.project
    exp_name = args.exp_name
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    lr = args.lr

    
    # === init model ===
    model = YOLO(model_weigths)  # или yolov8s.pt и т.д.

    results = model.train(
        data=path2data_file, 
        epochs=epochs,
        patience=patience, # early stopping
        imgsz=imgsize,
        batch=batch_size,
        workers=2,
        cache="ram",

        project=project,   # корневая папка для экспериментов
        name=exp_name,          # имя конкретного запуска
        exist_ok=True,                # не перезаписывать, если папка уже есть
        plots=True,

        optimizer="AdamW",
        lr0=lr, # add lr
        # --- seed ---
        seed=RANDOM_STATE,
        deterministic=True,

        # --- полное отключение аугментаций ---
        auto_augment=None,
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