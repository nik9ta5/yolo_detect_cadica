import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

# =======================================
# My modules
# =======================================
from data_utils.utils import (
    patient_lesion_nonlesion_videos,
    load_select_frames,
    load_frames_with_bboxes,
)


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def create_metaframe_fn(
    path_2_cadica_dataset_folder_selected_video: str, # Путь до директории с отобранными видео
    filenames_patirnts: list['str'] # Список пациентов
):
    all_rows = []

    for patient_item in tqdm(filenames_patirnts):
        
        path2_patient = f"{path_2_cadica_dataset_folder_selected_video}/{patient_item}"
        
        lesionVideos, nonlesionVideos = patient_lesion_nonlesion_videos(path2_patient)

        # === load lesion frames ===
        for video in lesionVideos:
            if video == "":
                continue

            frames_img_np, selectedFrames_filenames_without_ext, boxes = load_frames_with_bboxes(
                path2_patient,
                video
            )

            for key in list(boxes.keys()):
                for item_box in boxes[key]:
                    all_rows.append({
                        "path2frame": f"{path_2_cadica_dataset_folder_selected_video}/{patient_item}/{video}/input/{key}.png",
                        "patient": patient_item,
                        "video": video,
                        "frame": key,
                        "x": item_box['x'],
                        "y": item_box['y'],
                        "w": item_box['w'],
                        "h": item_box['h'],
                        "cls": item_box['cls'],
                    })

        # === load nonlesion frames ===
        for video in nonlesionVideos:
            if video == "":
                continue
            
            frames_img_np, selectedFrames_filenames = load_select_frames(
                path2_patient,
                video 
            )
            # === delete extension ===
            selectedFrames_filenames_without_ext = [iframe.split(".")[0] for iframe in selectedFrames_filenames]
            
            for item in selectedFrames_filenames_without_ext:
                all_rows.append({
                    "path2frame": f"{path_2_cadica_dataset_folder_selected_video}/{patient_item}/{video}/input/{item}.png",
                    "patient": patient_item,
                    "video": video,
                    "frame": item,
                    "x": None,
                    "y": None,
                    "w": None,
                    "h": None,
                    "cls": "-",
                })

    # === Собираем frame ===
    frame = pd.DataFrame(all_rows)
    return frame


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument("path2dataset",type=str, help="path2CADICA dataset 'CADICA'")
    parser.add_argument("filename",type=str, help="filename to save 'metaframe.csv'")
    parser.add_argument("imgsize",type=int, help="images size dataset")
    
    args = parser.parse_args()
    path2data = args.path2dataset 
    filename = args.filename
    imgsize = args.imgsize

    # === CADICA ===
    path_2_cadica_dataset_folder = f"{path2data}" # "../cv_projects/datasets/CADICA"
    path_2_cadica_dataset_folder_selected_video = f"{path_2_cadica_dataset_folder}/selectedVideos"

    # === Директории всех пациентов ===
    # Папки со всеми видео пациентов
    filenames_patirnts = sorted(os.listdir(path_2_cadica_dataset_folder_selected_video))
    
    frame = create_metaframe_fn(
    path_2_cadica_dataset_folder_selected_video,
        filenames_patirnts
    )
    
    # === Нормализуем координаты (yolo формат) ===
    frame['x_center_norm'] = (frame['x'] + frame['w'] / 2) / imgsize
    frame['y_center_norm'] = (frame['y'] + frame['h'] / 2) / imgsize

    frame['w_norm'] = frame['w'] / imgsize
    frame['h_norm'] = frame['h'] / imgsize

    os.makedirs("./data", exist_ok=True)
    frame.to_csv(f"./data/{filename}", index=False)