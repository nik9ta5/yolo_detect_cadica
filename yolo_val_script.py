import argparse
from ultralytics  import YOLO

# === parameters ===
RANDOM_STATE = 42


if __name__ == "__main__":

    # === argparse ====
    parser = argparse.ArgumentParser(description="script for val model")
    parser.add_argument("path2data_file", type=str, help="path to data.yaml file")
    parser.add_argument("split", type=str, help="split - val or train")
    parser.add_argument("weights",type=str, help="path to weigths model")
    parser.add_argument("imgsize", type=int, help="image size")
    parser.add_argument("project",type=str, help="project name")
    parser.add_argument("exp_name", type=str, help="experiment name")
    parser.add_argument("conf", type=float, help="confidence")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Размер батча")

    args = parser.parse_args()
    path2data_file = args.path2data_file
    split = args.split
    path2model = args.weights 
    imgsize = args.imgsize
    project = args.project
    exp_name = args.exp_name
    conf = args.conf
    batch_size = int(args.batch_size)


    # === model init ===
    model = YOLO(path2model)

    metrics = model.val(
        data=path2data_file,
        split=split,            # "val" или "test"
        imgsz=imgsize,
        batch=batch_size,
        workers=2,
        seed=RANDOM_STATE,
        deterministic=True,

        project=project,
        name=exp_name,
        exist_ok=True,

        save_json=True,         # метрики в формате COCO -> predictions.json
        plots=True,             # PR-кривые, confusion matrix и т.д.
        conf=conf, # 0.0001             # низкий порог для честного расчёта mAP
        iou=0.6,                # IoU для NMS при валидации
    )

    print('========== METRICS ==========')
    print(metrics.results_dict)