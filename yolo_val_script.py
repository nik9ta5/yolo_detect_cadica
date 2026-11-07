import argparse
from ultralytics  import YOLO

# === parameters ===
SEED = 2


if __name__ == "__main__":

    # === argparse ====
    parser = argparse.ArgumentParser(description="val script for yolo model")

    parser = argparse.ArgumentParser(description="Приветствие пользователя")
    parser.add_argument("weights",type=str, help="Путь до весов модели")
    parser.add_argument("-bs", "--batch_size", type=str, default=1, help="Размер батча")

    
    args = parser.parse_args()

    path2model = args.weights 
    batch_size = int(args.batch_size)


    # === model init ===
    model = YOLO(path2model)

    metrics = model.val(
        data="./data.yaml",
        split="val",            # "val" или "test"
        imgsz=512,
        batch=batch_size,
        workers=1,
        seed=SEED,
        deterministic=True,

        project="runs/baseline_v1_0_valid",
        name="yolov8n_val_only_bg",
        exist_ok=True,

        save_json=True,         # метрики в формате COCO -> predictions.json
        plots=True,             # PR-кривые, confusion matrix и т.д.
        conf=0.25, # 0.0001             # низкий порог для честного расчёта mAP
        iou=0.6,                # IoU для NMS при валидации
    )

    print('========== METRICS ==========')
    print(metrics.results_dict)


    # python yolo_val_script.py .\runs\detect\runs\baseline_v1_0\yolov8n_mini_aug\weights\best.pt --batch_size=16