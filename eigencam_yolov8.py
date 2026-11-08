"""
EigenCAM для YOLOv8 — визуализация того, на какие части изображения
смотрит детектор при принятии решения.

Запуск:
    python eigencam_yolov8.py --image photo.jpg --output result.jpg
    python eigencam_yolov8.py --image photo.jpg --weights yolov8s.pt --conf 0.4
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from ultralytics import YOLO


# ----------------------------------------------------------------------
# 1. EigenCAM — реализация «с нуля» (без pytorch-grad-cam)
# ----------------------------------------------------------------------

# Хранилище активаций на уровне модуля.
# Важно: хук — это просто функция (не замыкание/не bound method объекта,
# который держит модель), иначе deepcopy модели внутри ultralytics падает
# с ошибкой "cannot pickle '_thread.lock' object".
_CAPTURED_ACTIVATION: list = []


def _forward_hook(module, inputs, output):
    """Сохраняет выход слоя в глобальный список."""
    if isinstance(output, (tuple, list)):
        _CAPTURED_ACTIVATION.append(output[0])
    else:
        _CAPTURED_ACTIVATION.append(output)


class EigenCAM:
    """
    Class-agnostic визуализация карты признаков через PCA.

    Идея: берём feature map последнего свёрточного слоя (H x W x K),
    разворачиваем в матрицу (N x K), N = H*W, берём 1-ю главную компоненту
    и используем её как веса для линейной комбинации каналов.
    Получаем 2D-карту значимости (H x W).
    """

    def __init__(self, model, target_layer_idx: int):
        """
        Args:
            model: модель Ultralytics YOLO (model = YOLO('yolov8n.pt'))
            target_layer_idx: индекс слоя в model.model.model,
                              на который ставим hook.
                              Для YOLOv8 обычно 9 (SPPF) или 10.
        """
        # ВНИМАНИЕ: не храним ссылку на model у себя, чтобы избежать
        # циклической ссылки (model -> hooks -> self -> model), которая
        # ломает deepcopy внутри ultralytics.predict().
        self._target_layer_idx = target_layer_idx
        self._hook_handle = None
        self.feature_map = None

        # Регистрируем forward hook на нужном слое
        layer = model.model.model[target_layer_idx]
        self._hook_handle = layer.register_forward_hook(_forward_hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @staticmethod
    def _compute_eigencam(feature_map: torch.Tensor) -> np.ndarray:
        """
        Считает EigenCAM-карту из feature map.

        Args:
            feature_map: тензор формы (1, K, H, W) или (K, H, W)

        Returns:
            карта значимости формы (H, W), значения в [0, 1]
        """
        # Приводим к виду (K, H, W)
        if feature_map.dim() == 4:
            fmap = feature_map.squeeze(0)
        elif feature_map.dim() == 3:
            fmap = feature_map
        else:
            raise ValueError(f"Ожидался тензор 3D/4D, получен {feature_map.dim()}D")

        K, H, W = fmap.shape
        # Разворачиваем в матрицу (N x K), N = H*W
        X = fmap.reshape(K, H * W).T.detach().cpu().numpy()  # (N, K)

        # PCA: первая главная компонента
        # n_components=1 → fit_transform возвращает (N, 1) — проекцию на 1-ю ГК
        pca = PCA(n_components=1, svd_solver="randomized")
        try:
            L = pca.fit_transform(X)  # (N, 1)
        except Exception:
            # Запасной вариант: SVD вручную
            X_centered = X - X.mean(axis=0, keepdims=True)
            # Первая правая сингулярная компонента
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            L = X_centered @ Vt[0]  # (N,)
            L = L.reshape(-1, 1)

        # Разворачиваем обратно в (H, W)
        cam = L.reshape(H, W).astype(np.float32)

        # Берём модуль (карта может быть знаковой — нас интересует «сила»)
        cam = np.abs(cam)

        # Нормируем в [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            cam = np.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def __call__(self, model, image_rgb: np.ndarray,
                 conf: float = 0.25, iou: float = 0.45):
        """
        Полный пайплайн: детекция + EigenCAM + визуализация.

        Args:
            model: модель Ultralytics YOLO (передаётся снаружи, чтобы не
                   хранить ссылку на неё в self — это ломает deepcopy
                   внутри ultralytics.predict()).
            image_rgb: исходное изображение в RGB (H x W x 3), uint8
            conf: порог уверенности детекции
            iou: порог IoU для NMS

        Returns:
            dict:
              'image_with_cam': изображение с тепловой картой (RGB uint8)
              'image_with_cam_bbox': то же + bounding box'ы
              'detections': список (x1,y1,x2,y2,cls,conf)
              'cam_raw': сырая карта (H x W, float [0,1])
        """
        H_img, W_img = image_rgb.shape[:2]

        # 1) Очищаем хранилище и прогоняем YOLOv8 (хук сохранит feature map)
        _CAPTURED_ACTIVATION.clear()
        results = model.predict(
            source=image_rgb,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        result = results[0]

        if not _CAPTURED_ACTIVATION:
            raise RuntimeError("Feature map не была захвачена. "
                               "Проверьте target_layer_idx.")
        self.feature_map = _CAPTURED_ACTIVATION[-1]

        # 2) Достаём детекции
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # (N,4)
            confs = result.boxes.conf.cpu().numpy()              # (N,)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int) # (N,)
            names = result.names
            for (x1, y1, x2, y2), c, cid in zip(boxes, confs, cls_ids):
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(c),
                    "cls_id": int(cid),
                    "cls_name": names.get(cid, str(cid)),
                })

        # 3) Считаем EigenCAM из сохранённой feature map
        cam = self._compute_eigencam(self.feature_map)

        # 4) Resize карты до размера исходного изображения
        cam_resized = cv2.resize(cam, (W_img, H_img),
                                 interpolation=cv2.INTER_LINEAR)

        # 5) Переводим в цветную тепловую карту
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR

        # 6) Накладываем тепловую карту на изображение
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # 7.1) Версия без bbox
        image_with_cam = overlay_rgb.copy()

        # 7.2) Версия с bbox
        image_with_cam_bbox = overlay_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['cls_name']} {det['conf']:.2f}"

            # Прямоугольник
            cv2.rectangle(image_with_cam_bbox, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            # Фон под подпись
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, 2)
            cv2.rectangle(image_with_cam_bbox,
                          (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          (0, 255, 0), -1)
            cv2.putText(image_with_cam_bbox, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)

        return {
            "image_with_cam": image_with_cam,
            "image_with_cam_bbox": image_with_cam_bbox,
            "detections": detections,
            "cam_raw": cam_resized,
        }


# ----------------------------------------------------------------------
# 2. Утилиты для определения подходящего слоя
# ----------------------------------------------------------------------

def suggest_target_layer(model) -> int:
    """
    Подбирает индекс последнего свёрточного слоя перед головой детектора.

    В YOLOv8: model.model.model — это nn.ModuleList из ~23 слоёв,
    последний слой (index -1) — это Detect (голова).
    Мы хотим слой перед ним. Но у Ultralytics в новой версии может быть
    разное число слоёв, поэтому идём с конца и берём первый nn.Conv2d
    перед Detect.
    """
    layers = model.model.model
    for i in range(len(layers) - 2, -1, -1):
        # Ищем слой с conv (обычно C2f или Conv)
        if hasattr(layers[i], "cv1") or hasattr(layers[i], "cv2") \
           or "Conv" in type(layers[i]).__name__:
            return i
    return len(layers) - 2


# ----------------------------------------------------------------------
# 3. CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="EigenCAM для YOLOv8 — тепловая карта значимости"
    )
    p.add_argument("--image", type=str, required=True, help="Путь к входному изображению")
    p.add_argument("--weights", default="yolov8n.pt", help="Веса YOLOv8 (yolov8n.pt / yolov8s.pt / yolov8m.pt ...)")
    p.add_argument("--output", default=None, help="Куда сохранить результат (по умолчанию: рядом с image)")
    p.add_argument("--conf", type=float, default=0.25, help="Порог уверенности детекции")
    p.add_argument("--iou", type=float, default=0.45, help="Порог IoU для NMS")
    p.add_argument("--layer", type=int, default=None, help="Индекс слоя для хука (по умолчанию — подбирается автоматически)")
    p.add_argument("--save-cam-only", action="store_true", help="Также сохранить только тепловую карту (без bbox)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Изображение не найдено: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Загружаем модель {args.weights} ...")
    model = YOLO(args.weights)

    # Определяем целевой слой
    layer_idx = args.layer if args.layer is not None else suggest_target_layer(model)
    print(f"Целевой слой для EigenCAM: index={layer_idx} "
          f"({type(model.model.model[layer_idx]).__name__})")

    # Читаем изображение
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Не удалось прочитать изображение: {args.image}", file=sys.stderr)
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"shape image: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # Запускаем EigenCAM
    print("Запускаем детекцию и EigenCAM ...")
    cam = EigenCAM(model, target_layer_idx=layer_idx)
    try:
        out = cam(model, image_rgb, conf=args.conf, iou=args.iou)
    finally:
        cam.remove_hook()

    # Печать детекций
    print(f"\nНайдено объектов: {len(out['detections'])}")
    for d in out["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        print(f"   - {d['cls_name']:12s} conf={d['conf']:.3f}  "
              f"bbox=({x1},{y1})-({x2},{y2})")

    # Сохраняем результаты
    if args.output is None:
        stem = Path(args.image).stem
        out_dir = Path(args.image).parent
        out_path = str(out_dir / f"{stem}_eigencam.jpg")
    else:
        out_path = args.output

    out_bgr = cv2.cvtColor(out["image_with_cam_bbox"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)
    print(f"\nРезультат сохранён: {out_path}")

    if args.save_cam_only:
        stem = Path(out_path).stem
        ext = Path(out_path).suffix
        cam_path = str(Path(out_path).parent / f"{stem}_heatmap{ext}")
        overlay_only = cv2.cvtColor(out["image_with_cam"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(cam_path, overlay_only)
        print(f"Только heatmap: {cam_path}")

    print("done")


if __name__ == "__main__":
    main()
