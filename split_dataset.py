import os 
from tqdm import tqdm
import pandas as pd
import numpy as np 
import argparse
import shutil

from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def write_files(
    path2file: str,
    frame_list: list[str]
):
    with open(path2file, "w", encoding="utf-8") as f:
        for item in frame_list:
            f.write(f"{item}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument("path2metaframe",type=str, help="path2metaframe")
    
    args = parser.parse_args()
    path2data = args.path2metaframe 

    # === mapping for new labels ===
    map_to_new_cls = {
        "p0_20": "p0_50",
        "p20_50": "p0_50",

        "p50_70": "p50_90",
        "p70_90": "p50_90",

        "p90_98": "p_90_100",
        "p99": "p_90_100",
        "p100": "p_90_100",

        "-": '-'
    }
    new_cls_to_label = {
        "p0_50": 0,
        "p50_90": 1,
        "p_90_100": 2,
        "-": 3
    }

    # === read frame ===
    frame = pd.read_csv(path2data)

    # === Так как файлы в текущей директории - формируем новые пути ===
    frame['new_path2frame'] = frame['path2frame'].apply(lambda x: f"./dataset_yolo/images/{x.split("/")[-1]}")

    frame['new_cls'] = frame['cls'].apply(lambda x: map_to_new_cls[x])
    frame['label'] = frame['new_cls'].apply(lambda x: new_cls_to_label[x])
    
    # === split ===
    # === Со всем фоном ===

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    splits = []

    for k, (tr, va) in enumerate(sgkf.split(frame, y=frame["new_cls"], groups=frame["patient"])):
        splits.append((tr, va))

    # === Второе разбиение лучше ===
    tr, va = splits[1]

    tr_frame = frame.iloc[tr]
    val_frame = frame.iloc[va]

    train_frames = tr_frame['new_path2frame'].unique().tolist()
    val_frames = val_frame['new_path2frame'].unique().tolist()

    print(len(train_frames), len(val_frames))

    os.makedirs("./data", exist_ok=True)

    write_files(f"./train_split_2.txt", train_frames)
    write_files(f"./val_split_2.txt", val_frames)


    # === Сохраняем лейблы ===
    base_imgs_dir = "./dataset_yolo/images"
    base_labels_dir = "./dataset_yolo/labels"

    os.makedirs(base_labels_dir, exist_ok=True)
    os.makedirs(base_imgs_dir, exist_ok=True)


    for frame_item, group in tqdm(frame.groupby('frame'), desc="label files create"):
        label_file = os.path.join(base_labels_dir, f"{frame_item}.txt")
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                if pd.notna(row['x_center_norm']):
                    line = f"{int(row['label'])} {row['x_center_norm']:.6f} {row['y_center_norm']:.6f} {row['w_norm']:.6f} {row['h_norm']:.6f}\n"
                    f.write(line)

    # === Сохраняем картинки ===
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.tiff')
    for item_frame in tqdm(frame['path2frame'].unique().tolist(), desc="images load"):
        filename = os.path.basename(item_frame)
        if item_frame.lower().endswith(valid_extensions):
            target_path = os.path.join(base_imgs_dir, filename)

            shutil.copy2(item_frame, target_path)