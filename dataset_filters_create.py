import os
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import shutil


def get_parameters_argparse():
    parser = argparse.ArgumentParser(description="dataset filter create")
    parser.add_argument("path2dataset", type=str, help="path to dataset")
    parser.add_argument("folder_name", type=str, help="new folder name")

    return parser.parse_args()


def get_filenames(path2images: str, path2labels):
    image_filenames = os.listdir(path2images)
    labels_filenames = os.listdir(path2labels)
    image_filenames.sort()
    labels_filenames.sort()
    return image_filenames, labels_filenames


# ==========================================
# Filters
# ==========================================
def create_clahe_img(img: np.ndarray):
    """Функция для получения CLAHE изображения"""
    img_copy = img.copy()
    tmp_img_item_gray = img_copy
    if len(img_copy.shape) > 2:
        tmp_img_item_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    enhanced = clahe.apply(tmp_img_item_gray)

    return enhanced

def multi_scale_tophat(img_gray: np.ndarray, kernel_sizes=(10, 15, 20, 25, 30), mode="black") -> np.ndarray:
    """
    Multi-scale Top-Hat (формула 4 из статьи):
    берём максимум по всем масштабам. 
    mode='white' — светлые сосуды на тёмном фоне;
    mode='black' — тёмные сосуды на светлом (если у тебя инвертированный рентген).
    """
    morph_type = cv2.MORPH_TOPHAT if mode == "white" else cv2.MORPH_BLACKHAT
    responses = []
    for k in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        resp = cv2.morphologyEx(img_gray, morph_type, kernel)
        responses.append(resp)
    stack = np.stack(responses, axis=0)
    multi = stack.max(axis=0)
    return cv2.normalize(multi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


if __name__ == "__main__":
    # --- parameters ---    
    args = get_parameters_argparse()
    path2dataset = args.path2dataset
    folder_name = args.folder_name

    path2dataset_images = f"{path2dataset}/images"
    path2dataset_labels = f"{path2dataset}/labels"
    
    os.makedirs(folder_name, exist_ok=True)
    images_path = f"{folder_name}/images"
    labels_path = f"{folder_name}/labels"
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)


    # --- копируем все файлы с разбиениями ---
    for item in os.listdir(path2dataset):
        if Path(item).suffix == '.txt':
            shutil.copy2(f"{path2dataset}/{item}", f"{folder_name}/{item}")

    # 1. Получить файлы с изображениями и лейблами
    image_filenames, labels_filenames = get_filenames(path2dataset_images, path2dataset_labels)

    # Копируем лейблы
    for item in tqdm(labels_filenames, desc="labels copy"):
        item_path = f"{path2dataset_labels}/{item}"
        path_to = f"{labels_path}/{item}"
        shutil.copy2(item_path, path_to)

    # 2. Перебрать в цикле - открыть - применить фильтр - сконкатенировать - сохранить + сохранить лейбл    
    for item_img in tqdm(image_filenames, desc="images preprocessing"):
        item_path = f"{path2dataset_images}/{item_img}"
        img = cv2.imread(item_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR -> GRAY
        
        # --- clahe ---
        clahe = create_clahe_img(img_gray)
        # --- top hat ---
        img_tophat = multi_scale_tophat(img_gray, mode="black")
        
        # --- merge ---
        # [Orig_GRAY, CLAHE, Top-Hat]
        combined = cv2.merge([img_gray, clahe, img_tophat])

        # --- save image ---
        cv2.imwrite(f"{images_path}/{item_img}", combined)