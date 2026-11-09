import cv2 
import matplotlib.pyplot as plt
import torch

from analyze_errors.matching import classify_unmatched_pred


def visualize_prediction(
    result, matches, un_gt, un_pred, gt, dup_iou: float = 0.3, loc_iou: float = 0.1
):
    """Функция для визуализации GT и предсказаний"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # --- img ---
    img = result.orig_img
    if result.boxes is not None and len(result.boxes) > 0:
        pred_xyxy = result.boxes.xyxy
    else:
        pred_xyxy = torch.zeros((0, 4))
    gt_xyxy = gt[:, 1:5]
    
    # orig with gt, img with pred TP, img with FP
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    
    fp_info = classify_unmatched_pred(
        un_pred, pred_xyxy, gt_xyxy, matches,
        dup_iou=dup_iou, loc_iou=loc_iou,
    )

    # --- gt visualization ---
    img_gt = img.copy()
    for item in gt:
        pt1 = (int(item[1]), int(item[2]))
        pt2 = (int(item[3]), int(item[4]))
        cv2.rectangle(img_gt, pt1, pt2, (0, 255, 0), 1)
    axs[0].imshow(img_gt)
    axs[0].set_title(f"GT  (N={len(gt)})")
    axs[0].axis("off")

    # --- pred visualization ---
    img_tp = img.copy()
    # TP — жёлтый
    for g, p, iou in matches:
        b = pred_xyxy[p]
        pt1 = (int(b[0]), int(b[1])); pt2 = (int(b[2]), int(b[3]))
        cv2.rectangle(img_tp, pt1, pt2, (241, 196, 15), 1)
        cv2.putText(img_tp, f"{iou:.3f}", (pt1[0], pt1[1] - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # FN — красный
    for g in un_gt:
        b = gt[g]
        pt1 = (int(b[1]), int(b[2])); pt2 = (int(b[3]), int(b[4]))
        cv2.rectangle(img_tp, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(img_tp, "FN", (pt1[0], pt1[1] - 5), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    axs[1].imshow(img_tp)
    axs[1].set_title(f"TP={len(matches)}   FN={len(un_gt)}")
    axs[1].axis("off")

    # FP по категориям
    img_fp = img.copy()
    # для контекста — GT тонким серым
    for item in gt:
        pt1 = (int(item[1]), int(item[2])); pt2 = (int(item[3]), int(item[4]))
        cv2.rectangle(img_fp, pt1, pt2, (0, 225, 0), 1)

    color_map = {
        'duplicate':    (255, 0, 255),   # magenta  — дубликат (близко к занятому GT)
        'localization': (0, 255, 255),   # cyan     — почти попал, но мимо
        'background':   (255, 0, 0),     # красный  — галлюцинация
    }

    for p, meta in fp_info['per_fp'].items():
        b = pred_xyxy[p]
        pt1 = (int(b[0]), int(b[1])); pt2 = (int(b[2]), int(b[3]))
        c = color_map[meta['category']]
        
        cv2.rectangle(img_fp, pt1, pt2, c, 1)
        
        label = f"{meta['category'][:4]} iou={meta['best_iou']:.2f}"
        cv2.putText(img_fp, label, (pt1[0], pt1[1] - 5), font, 0.4, c, 1, cv2.LINE_AA)
    
    axs[2].imshow(img_fp)
    axs[2].set_title(
        f"FP:  dup={len(fp_info['duplicate'])}  "
        f"loc={len(fp_info['localization'])}  "
        f"bg={len(fp_info['background'])}"
    )
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()