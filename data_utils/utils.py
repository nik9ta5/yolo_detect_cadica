import os
import cv2
import numpy as np


# ==============================
# Функция для считывания видео с поражениями и без
# ==============================
def patient_lesion_nonlesion_videos(
    path2patient_folder: str # Путь до директории пациента
):  
    # Видео на которых присутствуют поражения
    lesionVideos = []
    with open(f'{path2patient_folder}/lesionVideos.txt', "r", encoding="utf-8") as file:
        lesionVideos = file.read().strip().split("\n")

    # Видео без поражения
    nonlesionVideos = []
    with open(f'{path2patient_folder}/nonlesionVideos.txt', "r", encoding="utf-8") as file:
        nonlesionVideos = file.read().strip().split("\n")
    
    return lesionVideos, nonlesionVideos


# ==============================
# Функция для загрузки видео
# ==============================
def load_video(
    path2patient_folder: str, # Путь до директории пациента
    video: str, # Видео 
    filenames: list[str], # Список с названием файлов   
    mode_img: str = "GRAY" # Режим загрузки
):
    """Функция для загрузки видео
    
    (Args)
        path2patient_folder: str - Путь до директории пациента
        video: str - Видео 
        filenames: list[str] - Список с названием файлов   
        mode_img: str = "GRAY" -  Режим загрузки
    
    Returns: frames_img_np: np.ndarray - тензор с фреймами, в порядке загрузке
    """
    frames = []
    for item in filenames:
        path2item_file = f"{path2patient_folder}/{video}/input/{item}"
        
        img_item = cv2.imread(path2item_file)
        if img_item is None:
            raise RuntimeError(f"file: {path2item_file} is None")

        if mode_img == "GRAY":
            tmp_img_item = cv2.cvtColor(img_item, cv2.COLOR_BGR2GRAY) # BGR -> Gray
        elif mode_img == "RGB":
            tmp_img_item = cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"not correct mode_img is {mode_img}")

        frames.append(tmp_img_item)

    frames_img_np = np.array(frames)
    
    return frames_img_np


# ==============================
# Функция для загрузки отдельных фреймов
# ==============================
def load_frames_paths(
    path2_frames: list[str],
    mode_img: str = "GRAY" # Режим загрузки
):
    """Функция для загрузки видео

    (Args)
        path2patient_folder: str - Путь до директории пациента
        video: str - Видео 
        filenames: list[str] - Список с названием файлов   
        mode_img: str = "GRAY" -  Режим загрузки

    Returns: frames_img_np: np.ndarray - тензор с фреймами, в порядке загрузке
    """
    frames = []
    for item in path2_frames:
        
        img_item = cv2.imread(item)
        if img_item is None:
            raise RuntimeError(f"file: {item} is None")

        if mode_img == "GRAY":
            tmp_img_item = cv2.cvtColor(img_item, cv2.COLOR_BGR2GRAY) # BGR -> Gray
        elif mode_img == "RGB":
            tmp_img_item = cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"not correct mode_img is {mode_img}")

        frames.append(tmp_img_item)

    frames_img_np = np.array(frames)

    return frames_img_np

# ==============================
# Функция для загрузки всего видео
# ==============================
def load_all_video(
    path2patient_folder: str, # Путь до директории пациента
    video: str, # Видео
    mode_img: str = "GRAY" # Режим загрузки
):
    """Функция для загрузки всего видео пациента
    
    (Args)
        path2patient_folder: str - путь до директории пациента
        video: str
    
    Returns: frames_img_np: np.ndarray, frame_filenames: list[str]
    """
    path2folder = f"{path2patient_folder}/{video}/input"
    frame_filenames = sorted(os.listdir(path2folder))

    # === load frames ===
    frames_img_np = load_video(
        path2patient_folder,
        video,
        frame_filenames,
        mode_img
    )

    return frames_img_np, frame_filenames


# ==============================
# Функция для загрузки выбранных фреймов
# ==============================
def load_select_frames(
    path2patient_folder: str, # Путь до директории пациента
    video: str, # Видео
    mode_img: str = "GRAY" # Режим загрузки    
):
    """Функция для загрузки выбранных фреймов
        
    (Args)
        path2patient_folder: str - Путь до директории пациента
        video: str - Видео   
        mode_img: str = "GRAY" -  Режим загрузки
    
    Returns: frames_img_np: np.ndarray - тензор с фреймами, в порядке загрузке
    """
    patient = path2patient_folder.split('/')[-1]
    path2video = f"{path2patient_folder}/{video}"
    
    path2selected_frames_file = f"{path2video}/{patient}_{video}_selectedFrames.txt"
    if not os.path.isfile(path2selected_frames_file):
        raise ValueError(f"patient is {patient} and video is {video} not excist")

    selectedFrames = []
    with open(path2selected_frames_file, "r", encoding="utf-8") as file:
        selectedFrames = file.read().strip().split("\n")

    selectedFrames_filenames = [f"{item}.png" for item in selectedFrames]
    selectedFrames_filenames = sorted(selectedFrames_filenames)

    frames_img_np = load_video(path2patient_folder, video, selectedFrames_filenames, mode_img)
    
    return frames_img_np, selectedFrames_filenames


# ==============================
# Загрузка боксов
# ==============================
def load_bboxs(
    path2patient_folder: str, # Путь до директории пациента
    video: str, # Видео
    filenames: list[str]
):
    """Функция для загрузке боксов"""
    path2gt_folder = f"{path2patient_folder}/{video}/groundtruth"
    if not os.path.isdir(path2gt_folder):
        raise ValueError(f'folder: "{path2gt_folder}" is not excist GT folder')

    boxes = dict()
    for item in filenames:
        boxes[item] = []

        path2item = f"{path2gt_folder}/{item}.txt"
        # [?] Если файла нет
        
        with open(path2item, "r", encoding="utf-8") as file:
            item_masks = file.read().strip().split("\n")
            # === Разделим все строки ===
            for i_str_mask in item_masks:
                x, y, w, h, cls = tuple(i_str_mask.split(" "))
                boxes[item].append({
                    "x": int(x), 
                    "y": int(y), 
                    "w": int(w), 
                    "h": int(h), 
                    "cls": cls
                })

    return boxes


# ==============================
# Загрузка фреймов с боксами
# ==============================
def load_frames_with_bboxes(
    path2patient_folder: str, # Путь до директории пациента
    video: str, # Видео
    mode_img: str = "GRAY"
):
    """Загрузка фреймов с боксами"""
    
    # === Проверяме, что для данного пациента и видео есть GT ===
    path2gt_folder = f"{path2patient_folder}/{video}/groundtruth"
    if not os.path.isdir(path2gt_folder):
        raise ValueError(f'folder: "{path2gt_folder}" is not excist GT folder')

    # === Загружаем выбранные фреймы ===
    frames_img_np, selectedFrames_filenames = load_select_frames(
        path2patient_folder,
        video,
        mode_img
    )

    # 1. Все выбранные фреймы для видео с поражениями содержат боксы
    # === without extension - filenames (c GT) (БЕЗ РАСШИРЕНИЯ: "px_vx_0000X" ) ===
    selectedFrames_filenames_without_ext = [item.split(".")[0] for item in selectedFrames_filenames]
    
    # === Загружаем боксы ===
    boxes = load_bboxs(path2patient_folder, video, selectedFrames_filenames_without_ext)

    return frames_img_np, selectedFrames_filenames_without_ext, boxes