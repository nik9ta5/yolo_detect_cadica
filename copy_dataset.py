import os 
from tqdm import tqdm
import pandas as pd
import numpy as np 
import argparse
import shutil


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

    parser = argparse.ArgumentParser(description="copy dataset files")
    parser.add_argument("path2metaframe",type=str, help="path to meta_frame.csv")
    parser.add_argument("folder_name", type=str, help="name for new folder dataset")
    
    args = parser.parse_args()
    path2data = args.path2metaframe 
    folder_name = args.folder_name

    # --- создаем новую папку для датасета ---
    os.makedirs(folder_name, exist_ok=True)

    folder_name_images = f"{folder_name}/images"
    folder_name_labels = f"{folder_name}/labels"
    
    os.makedirs(folder_name_images, exist_ok=True)
    os.makedirs(folder_name_labels, exist_ok=True)

    # === read frame ===
    frame = pd.read_csv(path2data)

    # === сохраняем лейблы ===
    for frame_item, group in tqdm(frame.groupby('frame'), desc="create lable files"):
        label_file = os.path.join(folder_name_labels, f"{frame_item}.txt")
        with open(label_file, 'w') as f:
            for _, row in group.iterrows():
                if pd.notna(row['x_center_norm']) and (row['new_cls'] != "-"):
                    line = f"{int(row['label'])} {row['x_center_norm']:.6f} {row['y_center_norm']:.6f} {row['w_norm']:.6f} {row['h_norm']:.6f}\n"
                    f.write(line)

    # === Сохраняем картинки ===
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.tiff')
    for item_frame in tqdm(frame['path2frame'].unique().tolist(), desc="copy images"):
        filename = os.path.basename(item_frame)
        if item_frame.lower().endswith(valid_extensions):
            target_path = os.path.join(folder_name_images, filename)
            shutil.copy2(item_frame, target_path)

    print("done")