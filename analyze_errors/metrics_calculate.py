import os
import numpy as np
import torch

from analyze_errors.boxes_utils import gt_bboxes_read, yolo_format_xcycwh_to_xyxy
from analyze_errors.matching import match_gt_pred_greedy, match_gt_with_results


def compute_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO-стиль)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Делаем precision монотонно убывающим (интерполяция справа налево)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    # trapz (в новых numpy — trapezoid)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    return float(ap)


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

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"TP": tp, "FP": fp, "FN": fn, "P": precision, "R": recall, "F1": f1}