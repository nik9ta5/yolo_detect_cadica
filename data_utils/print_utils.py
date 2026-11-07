import cv2
import numpy as np


def frame_bboxes_print(
    frame: np.ndarray, # Фрейм
    bboxes: list[dict], # Список боксов
    color: tuple[int], # Цвет для отрисовки
    pantins: int # Толщина
):
    """Функция для отрисовки бокса на фрейме
    
    (Args)
        frame: np.ndarray - фрейм, на котором нужно нарисовать боксы
        bboxes: list[dict] - боксы
        color: tuple[int] - цвет для отрисовки 
        pantins: int - толщина линии

    Return frame with bboxes
    """
    frame_copy = frame.copy()
    
    for item in bboxes:
        x = item['x']
        y = item['y'] 
        w = item['w'] 
        h = item['h'] 
        cls = item['cls']

        pnt1 = (x, y)
        pnt2 = (x + w, y + h)

        cv2.rectangle(frame_copy, pnt1, pnt2, color, pantins)

    return frame_copy

def video_bboxes_print(
    frames: np.ndarray,
    filenames: list[str],
    bboxes: list[dict],
    color: tuple[int],
    pantins: int
):
    """Функция для отрисовки боксов на видео"""
    frames_for_paint = []
    for idx, item in enumerate(filenames):
        item_frame = frame_bboxes_print(
            frames[idx],
            bboxes[item],
            color,
            pantins
        )
        frames_for_paint.append(item_frame)

    return frames_for_paint