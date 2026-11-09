import torch

from analyze_errors.boxes_utils import gt_bboxes_read, yolo_format_xcycwh_to_xyxy


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N, 4), b: (M, 4) -> (N, M) IoU"""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))

    a_ = a[:, None, :]
    b_ = b[None, :, :]
    inter_x1 = torch.max(a_[..., 0], b_[..., 0])
    inter_y1 = torch.max(a_[..., 1], b_[..., 1])
    inter_x2 = torch.min(a_[..., 2], b_[..., 2])
    inter_y2 = torch.min(a_[..., 3], b_[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_a = (a_[..., 2] - a_[..., 0]) * (a_[..., 3] - a_[..., 1])
    area_b = (b_[..., 2] - b_[..., 0]) * (b_[..., 3] - b_[..., 1])
    union = (area_a + area_b - inter).clamp(min=1e-7)
    return inter / union


def match_gt_pred_greedy(
    gt_xyxy_cls: torch.Tensor,    # (N, 5) [cls, x1, y1, x2, y2]
    pred_xyxy_cls: torch.Tensor,  # (M, 5) [cls, x1, y1, x2, y2]
    iou_thresh: float = 0.5,
):
    """
    Жадный one-to-one матчинг GT <-> предсказаний (аналог ultralytics.match_predictions
    для одного порога IoU):
      1. IoU, с обнулением пар с разными классами
      2. Оставляем пары с IoU >= iou_thresh
      3. Сортируем по убыванию IoU
      4. Жадно принимаем пару, если gt и pred ещё не сопоставлены

    Возвращает: (matches, unmatched_gt, unmatched_pred),
    где matches = [(gt_idx, pred_idx, iou), ...]
    """
    N, M = len(gt_xyxy_cls), len(pred_xyxy_cls)
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    gt_cls = gt_xyxy_cls[:, 0].long()
    pred_cls = pred_xyxy_cls[:, 0].long()

    ious = box_iou_xyxy(gt_xyxy_cls[:, 1:], pred_xyxy_cls[:, 1:])
    ious = ious * (gt_cls[:, None] == pred_cls[None, :])

    gt_idx, pred_idx = torch.nonzero(ious >= iou_thresh, as_tuple=True)
    if gt_idx.numel() == 0:
        return [], list(range(N)), list(range(M))

    pair_ious = ious[gt_idx, pred_idx]
    # === SORT IoU, not CONF ===    
    order = torch.argsort(pair_ious, descending=True)
    gt_idx = gt_idx[order].tolist()
    pred_idx = pred_idx[order].tolist()
    pair_ious = pair_ious[order].tolist()

    used_gt, used_pred = set(), set()
    matches = []
    for g, p, iou in zip(gt_idx, pred_idx, pair_ious):
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        matches.append((g, p, iou))

    unmatched_gt = [i for i in range(N) if i not in used_gt]
    unmatched_pred = [i for i in range(M) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def match_gt_with_results(
    result,
    path2labels: str,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.0,
    return_gt: bool = False
):
    """
    Матчинг GT (YOLO-файл) vs result.boxes (жадный, как в ultralytics).
    conf_thresh: дополнительный фильтр предиктов по уверенности (0 — не фильтровать).
    Возвращает: (matches, unmatched_gt, unmatched_pred)
    """
        
    # --- read gt ---
    h, w = result.orig_shape[:2]
    gt = gt_bboxes_read(path2labels, h, w, is_torch_tensor=True)  # (N, 5)

    if gt.numel() == 0:
        gt_xyxy_cls = gt.new_zeros((0, 5))
    else:
        coords_xyxy = yolo_format_xcycwh_to_xyxy(gt[:, 1:])
        gt_xyxy_cls = torch.cat([gt[:, 0:1], coords_xyxy], dim=1)  # (N, 5)

    # --- boxes ---
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        device = gt_xyxy_cls.device if gt_xyxy_cls.numel() > 0 else torch.device('cpu')
        pred_xyxy_cls = torch.zeros((0, 5), device=device)
    else:
        if conf_thresh > 0:
            keep = boxes.conf >= conf_thresh
            pred_xyxy = boxes.xyxy[keep]
            pred_cls = boxes.cls[keep].unsqueeze(1)
        else:
            pred_xyxy = boxes.xyxy
            pred_cls = boxes.cls.unsqueeze(1)

        pred_xyxy_cls = torch.cat([pred_cls, pred_xyxy], dim=1)  # (M, 5)
    
    # --- device ---
    device = pred_xyxy_cls.device
    matches, unmatched_gt, unmatched_pred = match_gt_pred_greedy(gt_xyxy_cls.to(device), pred_xyxy_cls, iou_thresh)
    
    if return_gt:
        return matches, unmatched_gt, unmatched_pred, gt_xyxy_cls
    return matches, unmatched_gt, unmatched_pred



def classify_unmatched_pred(
    un_pred, 
    pred_xyxy, 
    gt_xyxy, 
    matches,
    dup_iou: float = 0.3, 
    loc_iou: float = 0.1,
):
    """
    un_pred:    [p, ...]                 индексы несопоставленных предиктов
    pred_xyxy:  (M, 4) tensor            все предикты (xyxy)
    gt_xyxy:    (N, 4) tensor            все GT (xyxy)
    matches:    [(g, p, iou), ...]       для знания уже занятых GT

    Категории:
      duplicate     — IoU >= dup_iou с уже заматченным GT (NMS/дубль)
      localization  — loc_iou <= IoU < dup_iou (объект найден, бокс кривой)
      background    — IoU < loc_iou со всеми GT (модель "придумала")

    Возвращает dict:
      {'duplicate': [p...], 'localization': [p...], 'background': [p...],
       'per_fp':    {p: {'category': str, 'best_iou': float, 'best_gt': int}}}
    """
    if not un_pred or gt_xyxy.numel() == 0:
        return {
            'duplicate': [], 'localization': [], 'background': [],
            'per_fp': {p: {'category': 'background', 'best_iou': 0.0, 'best_gt': -1}
                       for p in un_pred},
        }

    matched_gt = {m[0] for m in matches}
    device = pred_xyxy.device

    fp_idx = torch.tensor(un_pred, dtype=torch.long, device=device)
    fp_boxes = pred_xyxy[fp_idx]

    ious = box_iou_xyxy(gt_xyxy.to(device), fp_boxes)          # (N, K)
    best_iou, best_gt = ious.max(dim=0)                        # (K,), (K,)

    duplicate, localization, background = [], [], []
    per_fp = {}
    for k, p in enumerate(un_pred):
        bi = float(best_iou[k].item())
        bg = int(best_gt[k].item())
        if bi >= dup_iou and bg in matched_gt:
            cat = 'duplicate';    duplicate.append(p)
        elif bi >= loc_iou:
            cat = 'localization'; localization.append(p)
        else:
            cat = 'background';   background.append(p)
        per_fp[p] = {'category': cat, 'best_iou': bi, 'best_gt': bg}

    return {
        'duplicate': duplicate, 'localization': localization,
        'background': background, 'per_fp': per_fp,
    }