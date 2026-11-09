"""
EigenCAM: class-agnostic saliency map для ultralytics YOLO.

Идея (Raj et al., 2021, arXiv:2110.03077):
    Берём feature map A размерности (C, H, W), разворачиваем в (C, H*W),
    считаем SVD: A = U @ diag(S) @ Vh.
    Первая строка Vh (правый сингулярный вектор с наибольшим σ) — это
    доминирующий пространственный паттерн активации по всем каналам.
    Ресайзим её в (H, W), нормируем в [0, 1] — это и есть EigenCAM.

Особенности:
    * Без градиентов → работает на inference, не требует backward.
    * Class-agnostic → для детекции: показывает «куда вообще смотрит сеть»,
      без привязки к конкретному классу.
    * Не зависит от confidence/NMS — отражает внутреннюю активацию.

Usage:
    from analyze_errors.eigen_cam import EigenCAM, visualize_prediction_with_cam

    cam = EigenCAM(model, target_layer_idx=-2)        # P5 — самый семантичный слой
    result, heatmap, feat_shape = cam(result_item.path, imgsz=imgsize)

    matches, un_gt, un_pred, gt = match_gt_with_results(
        result, path2label, iou_thresh=match_iou, conf_thresh=conf, return_gt=True
    )
    visualize_prediction_with_cam(result, matches, un_gt, un_pred, gt, heatmap)

    # В самом конце:
    cam.remove_hooks()
"""
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO


# =====================================================================================
# EigenCAM
# =====================================================================================
class EigenCAM:
    """
    SVD-based class-agnostic saliency для ultralytics YOLO.

    Хукирует один слой в model.model.model, при каждом вызове __call__
    сохраняет его forward-output, по нему строит EigenCAM-тепловую карту.
    """

    def __init__(self, model: YOLO, target_layer_idx: int = -2):
        """
        Args:
            model: обученная ultralytics YOLO (та, что передаёшь в model.predict(...)).
            target_layer_idx: индекс слоя в model.model.model, с которого снимаем feature map.

                Для YOLOv8/v11 (стандартный backbone+head из ~23 модулей):
                    15  — P3 (высокое разрешение, ~imgsz/8)   — лучше для мелких объектов
                    18  — P4 (среднее,       ~imgsz/16)
                    21  — P5 (низкое,        ~imgsz/32)       — самый семантичный
                По умолчанию -2 (=21 для 23-модульной модели) — рекомендуется,
                если хочешь общую картину «куда смотрит сеть».
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.activations: torch.Tensor | None = None
        self._hook_handles: list = []
        self._register_hook()

    # ----- hook -----
    def _get_seq(self):
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "model"):
            raise RuntimeError(
                "Не нашёл model.model.model — это точно ultralytics YOLO? "
                "Ожидалась структура YOLO -> DetectionModel -> nn.Sequential."
            )
        return self.model.model.model

    def _register_hook(self):
        seq = self._get_seq()
        layer = seq[self.target_layer_idx]

        def forward_hook(module, inp, out):
            # В YOLO-бэкбоне/голове выход почти всегда тензор,
            # но на всякий случай берем первый элемент tuple/list.
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.activations = out.detach()

        self._hook_handles.append(layer.register_forward_hook(forward_hook))

    def remove_hooks(self):
        """Снять все хуки. Вызывать в конце работы, чтобы не течь память."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        self.activations = None

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

    # ----- EigenCAM math -----
    @staticmethod
    def _compute_eigencam(feature_map: torch.Tensor) -> np.ndarray:
        """
        SVD на (C, H*W), берём Vh[0] (первый правый сингулярный вектор) и
        разворачиваем в (H, W). Корректируем знак по корреляции со средней
        активацией по каналам, нормируем в [0, 1].

        Зачем коррекция знака:
            У SVD знак сингулярного вектора произволен — Vh[0] и -Vh[0]
            дают одинаковый reconstruction error. На ReLU-сетях знак обычно
            выходит «правильным» (там, где активация высокая, CAM > 0),
            но на SiLU/GELU-сетях (YOLOv8/v11, YOLO-NAS) активации могут
            быть отрицательными, и CAM может инвертироваться: покажет
            «куда сеть НЕ смотрит» вместо «куда смотрит».
            Поэтому принудительно ориентируем Vh[0] так, чтобы он был
            положительно скоррелирован со средней по каналам активацией.
        """
        if feature_map.dim() == 4:
            feat = feature_map.squeeze(0)
        elif feature_map.dim() == 3:
            feat = feature_map
        else:
            raise ValueError(f"Ожидался 3D/4D feature map, получил {feature_map.dim()}D")

        C, H, W = feat.shape
        A = feat.cpu().numpy().reshape(C, H * W).astype(np.float32)

        # A = U @ diag(S) @ Vh
        # Vh[0] — первый правый сингулярный вектор (соответствует наибольшему σ).
        # Это доминирующий пространственный паттерн активации по всем каналам.
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        cam_flat = Vh[0]                            # (H*W,)
        mean_act_flat = A.mean(axis=0)              # (H*W,) — средняя активация по каналам

        # Знак: Vh[0] должен положительно скоррелировать со средней активацией.
        # Если корреляция отрицательная — переворачиваем.
        if np.dot(cam_flat, mean_act_flat) < 0:
            cam_flat = -cam_flat

        cam = cam_flat.reshape(H, W)

        cam_min, cam_max = float(cam.min()), float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.astype(np.float32)

    # ----- inference -----
    def __call__(self, img, imgsz: int = 512, **predict_kwargs):
        """
        Прогоняет img через model.predict (с активным хуком) и возвращает
        результат + EigenCAM-карту.

        Args:
            img: путь | BGR numpy | list — что угодно, что принимает model.predict.
                 Внимание: для батча активации перезапишутся последней картинкой —
                 для CAM лучше передавать ОДНУ картинку.
            imgsz: размер инференса (важно совпадать с обучением для честной CAM).
            **predict_kwargs: пробрасываются в model.predict (conf, iou, device, ...).

        Returns:
            result:     ultralytics Results (первая/единственная картинка).
            heatmap:    (H_fm, W_fm) float32 в [0, 1].
            feat_shape: (C, H_fm, W_fm).
        """
        self.activations = None

        defaults = dict(imgsz=imgsz, verbose=False, conf=0.001)
        defaults.update(predict_kwargs)
        results = self.model.predict(source=img, **defaults)
        result = results[0]

        if self.activations is None:
            raise RuntimeError(
                f"Активации со слоя idx={self.target_layer_idx} не пойманы. "
                f"Вероятно, индекс не подходит для этой версии модели — "
                f"попробуй 15 / 18 / 21 (для YOLOv8/v11)."
            )

        heatmap = self._compute_eigencam(self.activations)
        feat_shape = tuple(self.activations.shape[-3:])
        return result, heatmap, feat_shape


# =====================================================================================
# Overlay helpers
# =====================================================================================
def overlay_heatmap(
    img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Наложить [0,1] heatmap на BGR-картинку.

    Args:
        img_bgr:   HxWx3 BGR uint8.
        heatmap:   HxWx float в [0, 1] (ресайзится под img).
        alpha:     вес тепловой карты (1-alpha — вес изображения).
        colormap:  cv2 colormap id (COLORMAP_JET / INFERNO / TURBO ...).

    Returns:
        HxWx3 BGR uint8.
    """
    H, W = img_bgr.shape[:2]
    hm = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, colormap)
    blended = cv2.addWeighted(img_bgr.astype(np.uint8), 1 - alpha, hm_color, alpha, 0)
    return blended


# =====================================================================================
# Visualizations
# =====================================================================================
def visualize_eigencam(
    img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    figsize=(10, 5),
    colormap: int = cv2.COLORMAP_JET,
):
    """
    Только EigenCAM: 2 панели — сырая карта | оверлей на картинку.
    Боксов нет, чисто «куда смотрит сеть».
    """
    overlay = overlay_heatmap(img_bgr, heatmap, alpha=alpha, colormap=colormap)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(heatmap, cmap="jet")
    axs[0].set_title(f"EigenCAM  ({heatmap.shape[0]}x{heatmap.shape[1]})")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Overlay  (alpha={alpha})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_prediction_with_cam(
    result,
    matches,
    un_gt,
    un_pred,
    gt,
    heatmap: np.ndarray,
    dup_iou: float = 0.3,
    loc_iou: float = 0.1,
    alpha: float = 0.45,
    figsize=(20, 7),
):
    """
    Расширяет analyze_errors.visualization.visualize_prediction четвёртой панелью —
    EigenCAM-оверлей поверх боксов. Так видно, куда смотрит сеть относительно
    GT / TP / FN / FP-боксов.

    Args — те же, что у visualize_prediction, плюс:
        heatmap:  (H_fm, W_fm) float в [0, 1] из EigenCAM.__call__
        alpha:    вес тепловой карты в оверлее (1-alpha — вес изображения)

    Панели:
        1. GT                          (как в visualize_prediction)
        2. TP / FN                     (как в visualize_prediction)
        3. FP по категориям            (как в visualize_prediction)
        4. EigenCAM + все боксы        (новая)
    """
    from analyze_errors.matching import classify_unmatched_pred

    font = cv2.FONT_HERSHEY_SIMPLEX

    img = result.orig_img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if result.boxes is not None and len(result.boxes) > 0:
        pred_xyxy = result.boxes.xyxy
    else:
        pred_xyxy = torch.zeros((0, 4))
    gt_xyxy = gt[:, 1:5] if gt.numel() > 0 else torch.zeros((0, 4))

    fp_info = classify_unmatched_pred(
        un_pred, pred_xyxy, gt_xyxy, matches,
        dup_iou=dup_iou, loc_iou=loc_iou,
    )

    color_map = {
        "duplicate":    (255, 0, 255),   # magenta — дубликат
        "localization": (0, 255, 255),   # cyan    — почти попал
        "background":   (255, 0, 0),     # red     — галлюцинация
    }

    # --- 1. GT ---
    img_gt = img.copy()
    for item in gt:
        pt1 = (int(item[1]), int(item[2])); pt2 = (int(item[3]), int(item[4]))
        cv2.rectangle(img_gt, pt1, pt2, (0, 255, 0), 1)

    # --- 2. TP / FN ---
    img_tp = img.copy()
    for g, p, iou in matches:
        b = pred_xyxy[p]
        pt1 = (int(b[0]), int(b[1])); pt2 = (int(b[2]), int(b[3]))
        cv2.rectangle(img_tp, pt1, pt2, (241, 196, 15), 1)
        cv2.putText(img_tp, f"{iou:.3f}", (pt1[0], pt1[1] - 5), font, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    for g in un_gt:
        b = gt[g]
        pt1 = (int(b[1]), int(b[2])); pt2 = (int(b[3]), int(b[4]))
        cv2.rectangle(img_tp, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(img_tp, "FN", (pt1[0], pt1[1] - 5), font, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)

    # --- 3. FP категории ---
    img_fp = img.copy()
    for item in gt:
        pt1 = (int(item[1]), int(item[2])); pt2 = (int(item[3]), int(item[4]))
        cv2.rectangle(img_fp, pt1, pt2, (0, 225, 0), 1)
    for p, meta in fp_info["per_fp"].items():
        b = pred_xyxy[p]
        pt1 = (int(b[0]), int(b[1])); pt2 = (int(b[2]), int(b[3]))
        c = color_map[meta["category"]]
        cv2.rectangle(img_fp, pt1, pt2, c, 1)
        label = f"{meta['category'][:4]} iou={meta['best_iou']:.2f}"
        cv2.putText(img_fp, label, (pt1[0], pt1[1] - 5), font, 0.4, c, 1, cv2.LINE_AA)

    # --- 4. EigenCAM + все боксы ---
    img_cam = overlay_heatmap(img, heatmap, alpha=alpha)

    # GT тонким зелёным (контекст)
    for item in gt:
        pt1 = (int(item[1]), int(item[2])); pt2 = (int(item[3]), int(item[4]))
        cv2.rectangle(img_cam, pt1, pt2, (0, 255, 0), 1)
    # TP — жёлтый, потолще
    for g, p, iou in matches:
        b = pred_xyxy[p]
        pt1 = (int(b[0]), int(b[1])); pt2 = (int(b[2]), int(b[3]))
        cv2.rectangle(img_cam, pt1, pt2, (241, 196, 15), 2)
    # FN — красный
    for g in un_gt:
        b = gt[g]
        pt1 = (int(b[1]), int(b[2])); pt2 = (int(b[3]), int(b[4]))
        cv2.rectangle(img_cam, pt1, pt2, (255, 0, 0), 2)
    # FP — по категориям
    for p, meta in fp_info["per_fp"].items():
        b = pred_xyxy[p]
        pt1 = (int(b[0]), int(b[1])); pt2 = (int(b[2]), int(b[3]))
        c = color_map[meta["category"]]
        cv2.rectangle(img_cam, pt1, pt2, c, 2)

    # --- plot ---
    fig, axs = plt.subplots(1, 4, figsize=figsize)

    axs[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"GT  (N={len(gt)})")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img_tp, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"TP={len(matches)}   FN={len(un_gt)}")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(img_fp, cv2.COLOR_BGR2RGB))
    axs[2].set_title(
        f"FP:  dup={len(fp_info['duplicate'])}  "
        f"loc={len(fp_info['localization'])}  "
        f"bg={len(fp_info['background'])}"
    )
    axs[2].axis("off")

    axs[3].imshow(cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB))
    axs[3].set_title(
        f"EigenCAM (alpha={alpha})  TP={len(matches)} FN={len(un_gt)} FP={len(un_pred)}"
    )
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()
