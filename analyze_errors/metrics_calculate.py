import os
import numpy as np
import torch

from analyze_errors.boxes_utils import gt_bboxes_read, yolo_format_xcycwh_to_xyxy
from analyze_errors.matching import match_gt_pred_greedy, match_gt_with_results
from analyze_errors.matching import box_iou_xyxy


def _trapz(y, x):
    """numpy >= 2 имеет trapezoid, старый — trapz."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def compute_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    101-point interpolated AP, COCO/VOC-like.
    """
    if len(recall) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # monotone decreasing precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = _trapz(np.interp(x, mrec, mpre), x)
    return float(ap)

# ===============================
# Compute metrics
# ===============================
def compute_map(
    results,
    labels_dir: str,
    iou_thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05),
    num_classes: int = 1,
):
    """
    Считает AP/mAP для всех IoU-порогов.
    ВАЖНО: results должны быть получены с conf≈0.001 (или ниже),
    чтобы содержать максимум предсказаний. Никакой фильтрации по conf не делаем.

    Returns dict:
        ap_table:   (num_classes, num_iou_thresholds) — AP[class, iou]
        map_05:     mAP@0.5
        map_50_95:  mAP@0.5–0.95
        per_iou_map: dict {iou_t: mAP}
        per_class_ap_05: dict {class_id: AP@0.5}
    """
    all_conf, all_cls, all_correct = [], [], []
    gt_count = np.zeros(num_classes, dtype=int)

    for r in results:
        stem = os.path.splitext(os.path.basename(r.path))[0]
        p2l = os.path.join(labels_dir, f"{stem}.txt")

        # --- GT ---
        h, w = r.orig_shape[:2]
        gt = gt_bboxes_read(p2l, h, w)
        if gt.numel() == 0:
            gt_xyxy_cls = gt.new_zeros((0, 5))
        else:
            coords = yolo_format_xcycwh_to_xyxy(gt[:, 1:].clone())
            gt_xyxy_cls = torch.cat([gt[:, 0:1], coords], dim=1)
        for g_cls in gt_xyxy_cls[:, 0].long().tolist():
            if 0 <= g_cls < num_classes:
                gt_count[g_cls] += 1

        # --- PRED (без порога conf) ---
        n_pred = len(r.boxes) if r.boxes is not None else 0
        if n_pred == 0:
            continue

        pred_xyxy = r.boxes.xyxy
        pred_cls = r.boxes.cls
        pred_conf = r.boxes.conf

        pred_xyxy_cls = torch.cat([pred_cls.unsqueeze(1), pred_xyxy], dim=1)
        dev = pred_xyxy_cls.device

        # --- correct matrix (N_pred, N_iou) ---
        correct = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
        for t_idx, iou_t in enumerate(iou_thresholds):
            matches, _, _ = match_gt_pred_greedy(
                gt_xyxy_cls.to(dev), pred_xyxy_cls, float(iou_t)
            )
            for g, p, _iou in matches:
                correct[p, t_idx] = True

        all_conf.append(pred_conf.cpu().numpy())
        all_cls.append(pred_cls.cpu().numpy().astype(int))
        all_correct.append(correct)

    if not all_conf:
        return {
            'ap_table': np.zeros((num_classes, len(iou_thresholds))),
            'map_05': 0.0, 'map_50_95': 0.0,
            'per_iou_map': {}, 'per_class_ap_05': {},
        }

    all_conf = np.concatenate(all_conf)
    all_cls = np.concatenate(all_cls)
    all_correct = np.concatenate(all_correct, axis=0)

    # Глобальная сортировка по conf (по убыванию) — ключ к mAP
    order = np.argsort(-all_conf)
    all_conf = all_conf[order]
    all_cls = all_cls[order]
    all_correct = all_correct[order]

    ap_table = np.full((num_classes, len(iou_thresholds)), np.nan)
    for c in range(num_classes):
        n_gt = gt_count[c]
        if n_gt == 0:
            continue  # класса нет в GT → AP не определён
        mask = all_cls == c
        for t_idx in range(len(iou_thresholds)):
            tp = all_correct[mask, t_idx]
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(~tp)
            recall = tp_cum / n_gt
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            ap_table[c, t_idx] = compute_ap_101(recall, precision)

    iou_05_idx = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    return {
        'ap_table':      ap_table,
        'map_05':        float(np.nanmean(ap_table[:, iou_05_idx])),
        'map_50_95':     float(np.nanmean(ap_table)),
        'per_iou_map':   {float(t): float(np.nanmean(ap_table[:, i])) for i, t in enumerate(iou_thresholds)},
        'per_class_ap_05': {c: float(ap_table[c, iou_05_idx]) for c in range(num_classes)},
    }


# ===============================
# Compute box metrics
# ===============================
def box_metrics_calculate(results, path2label_dir, iou, conf):
    tp, fp, fn = 0, 0, 0
    for i_res in results:
        stem = os.path.splitext(os.path.basename(i_res.path))[0]
        path2label = os.path.join(path2label_dir, f"{stem}.txt")

        matches, un_gt, un_pred = match_gt_with_results(
            i_res, path2label, iou_thresh=iou, conf_thresh=conf, return_gt=False
        )
        tp += len(matches)
        fp += len(un_pred)
        fn += len(un_gt)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"TP": tp, "FP": fp, "FN": fn, "P": precision, "R": recall, "F1": f1}


# ===============================
# Compute frame level metrics
# ===============================
def frame_level_metrics(results, labels_dir, conf_thresh=0.25, iou_thresh=0.5):
    """
    Метрики на уровне кадров для бинарной задачи:
    "есть ли стеноз в кадре?"
    
    Возвращает:
        sensitivity, specificity, precision, recall, f1, accuracy
        + confusion matrix
    """
    tp_frames = fp_frames = fn_frames = tn_frames = 0
    
    for r in results:
        stem = os.path.splitext(os.path.basename(r.path))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        
        # GT: есть ли хотя бы один стеноз в кадре?
        h, w = r.orig_shape[:2]
        gt = gt_bboxes_read(label_path, h, w)
        gt_positive = len(gt) > 0
        
        # Pred: есть ли хотя бы один предсказанный стеноз с достаточной уверенностью?
        pred_positive = False
        if r.boxes is not None and len(r.boxes) > 0:
            pred_positive = (r.boxes.conf >= conf_thresh).any().item()
        
        # Confusion matrix на уровне кадров
        if gt_positive and pred_positive:
            tp_frames += 1
        elif gt_positive and not pred_positive:
            fn_frames += 1
        elif not gt_positive and pred_positive:
            fp_frames += 1
        else:
            tn_frames += 1
    
    total = tp_frames + fp_frames + fn_frames + tn_frames
    sensitivity = tp_frames / (tp_frames + fn_frames + 1e-8)  # Recall для стенозов
    specificity = tn_frames / (tn_frames + fp_frames + 1e-8)
    precision = tp_frames / (tp_frames + fp_frames + 1e-8)
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp_frames + tn_frames) / (total + 1e-8)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp_frames, 'fp': fp_frames,
        'fn': fn_frames, 'tn': tn_frames
    }


# ===============================
# Compute metrics COCO style
# ===============================
def compute_map_coco_style(
    results,
    labels_dir: str,
    iou_thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05),
    num_classes: int = 1,
):
    """
    COCO-style mAP.

    Основные отличия от твоей прошлой версии:
    1. Для каждого изображения читаем GT и предсказания.
    2. Для каждого класса:
       - собираем все предсказания этого класса;
       - сортируем их глобально по убыванию confidence;
       - для каждого порога IoU идем по предсказаниям в этом порядке;
       - каждое предсказание матчится с лучшим свободным GT того же класса,
         если IoU >= threshold.

    Args:
        results: список ultralytics Results.
            ВАЖНО: лучше получать с conf=0.001 или ниже.
        labels_dir: папка с YOLO label .txt.
        iou_thresholds: пороги IoU, обычно [0.50, 0.55, ..., 0.95].
        num_classes: число классов.

    Returns:
        dict:
            ap_table: np.ndarray shape (num_classes, len(iou_thresholds))
            map_05: float
            map_50_95: float
            per_iou_map: dict
            per_class_ap_05: dict
    """
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float32)

    gt_count = np.zeros(num_classes, dtype=int)

    # Будем хранить всё на CPU, чтобы не зависеть от device
    image_gt = []       # list[torch.Tensor] shape (N, 5): [cls, x1, y1, x2, y2]
    image_preds = []    # list[torch.Tensor] shape (M, 6): [cls, conf, x1, y1, x2, y2]

    # ------------------------------------------------------------------
    # 1. Чтение GT и предсказаний
    # ------------------------------------------------------------------
    for r in results:
        stem = os.path.splitext(os.path.basename(r.path))[0]
        p2l = os.path.join(labels_dir, f"{stem}.txt")

        h, w = r.orig_shape[:2]

        gt = gt_bboxes_read(p2l, h, w, is_torch_tensor=True)
        if gt.numel() == 0:
            gt_xyxy_cls = torch.zeros((0, 5), dtype=torch.float32)
        else:
            coords = yolo_format_xcycwh_to_xyxy(gt[:, 1:].clone())
            gt_xyxy_cls = torch.cat([gt[:, 0:1], coords], dim=1)

        image_gt.append(gt_xyxy_cls.cpu())

        if gt_xyxy_cls.shape[0] > 0:
            cls_arr = gt_xyxy_cls[:, 0].long().cpu().numpy()
            cls_arr = cls_arr[(cls_arr >= 0) & (cls_arr < num_classes)]
            np.add.at(gt_count, cls_arr, 1)

        if r.boxes is not None and len(r.boxes) > 0:
            pred = torch.cat(
                [
                    r.boxes.cls.float().cpu().unsqueeze(1),
                    r.boxes.conf.float().cpu().unsqueeze(1),
                    r.boxes.xyxy.float().cpu(),
                ],
                dim=1,
            )
        else:
            pred = torch.zeros((0, 6), dtype=torch.float32)

        image_preds.append(pred)

    # ------------------------------------------------------------------
    # 2. AP для каждого класса и каждого IoU threshold
    # ------------------------------------------------------------------
    ap_table = np.full((num_classes, len(iou_thresholds)), np.nan, dtype=np.float32)

    for c in range(num_classes):
        n_gt = int(gt_count[c])
        if n_gt == 0:
            # класс отсутствует в валидации
            continue

        # GT boxes только этого класса для каждого изображения
        gt_by_image = []
        for gt_xyxy_cls in image_gt:
            if gt_xyxy_cls.shape[0] == 0:
                gt_by_image.append(torch.zeros((0, 4), dtype=torch.float32))
            else:
                cls_mask = gt_xyxy_cls[:, 0].long() == c
                if cls_mask.any():
                    gt_by_image.append(gt_xyxy_cls[cls_mask, 1:].clone())
                else:
                    gt_by_image.append(torch.zeros((0, 4), dtype=torch.float32))

        # Собираем все предсказания класса c
        pred_records = []  # (img_id, conf, box_tensor)
        for img_id, preds in enumerate(image_preds):
            if preds.shape[0] == 0:
                continue

            cls_mask = preds[:, 0].long() == c
            if not cls_mask.any():
                continue

            class_preds = preds[cls_mask]
            for row in class_preds:
                conf = float(row[1].item())
                box = row[2:6].clone().float()
                pred_records.append((img_id, conf, box))

        # Если предсказаний нет, а GT есть => AP = 0
        if len(pred_records) == 0:
            ap_table[c, :] = 0.0
            continue

        # Глобальная сортировка по confidence — ключевое отличие
        pred_records.sort(key=lambda x: x[1], reverse=True)

        # Заранее посчитаем IoU каждого предсказания с GT того же класса
        pred_ious = []
        for img_id, conf, box in pred_records:
            gt_boxes = gt_by_image[img_id]
            if gt_boxes.shape[0] == 0:
                ious = torch.zeros((0,), dtype=torch.float32)
            else:
                ious = box_iou_xyxy(box.view(1, 4), gt_boxes)[0]
            pred_ious.append((img_id, ious))

        # Для каждого порога IoU свой независимый матчинг
        for t_idx, iou_t in enumerate(iou_thresholds):
            matched_gt_set = set()
            tp = np.zeros(len(pred_ious), dtype=bool)

            for p_idx, (img_id, ious) in enumerate(pred_ious):
                if ious.numel() == 0:
                    continue

                valid_idx = torch.nonzero(ious >= float(iou_t), as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    continue

                # среди валидных GT берем лучший по IoU, но только если он еще свободен
                sorted_valid_idx = valid_idx[
                    torch.argsort(ious[valid_idx], descending=True)
                ]

                for gt_local_idx in sorted_valid_idx.tolist():
                    key = (img_id, int(gt_local_idx))
                    if key not in matched_gt_set:
                        matched_gt_set.add(key)
                        tp[p_idx] = True
                        break

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(~tp)

            recall = tp_cum / float(n_gt)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)

            ap_table[c, t_idx] = compute_ap_101(recall, precision)

    # ------------------------------------------------------------------
    # 3. Итоговые метрики
    # ------------------------------------------------------------------
    iou_05_idx = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    valid_class_mask = gt_count > 0

    if valid_class_mask.any():
        map_05 = float(np.nanmean(ap_table[valid_class_mask, iou_05_idx]))
        map_50_95 = float(np.nanmean(ap_table[valid_class_mask]))
    else:
        map_05 = 0.0
        map_50_95 = 0.0

    per_iou_map = {}
    for i, t in enumerate(iou_thresholds):
        if valid_class_mask.any():
            per_iou_map[float(t)] = float(np.nanmean(ap_table[valid_class_mask, i]))
        else:
            per_iou_map[float(t)] = 0.0

    per_class_ap_05 = {}
    for c in range(num_classes):
        if gt_count[c] > 0:
            per_class_ap_05[c] = float(ap_table[c, iou_05_idx])
        else:
            per_class_ap_05[c] = None

    return {
        "ap_table": ap_table,
        "map_05": map_05,
        "map_50_95": map_50_95,
        "per_iou_map": per_iou_map,
        "per_class_ap_05": per_class_ap_05,
        "gt_count": gt_count,
    }