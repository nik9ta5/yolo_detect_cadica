import os
import numpy as np 
import torch 


def gt_bboxes_read(path2labels, img_h, img_w, is_torch_tensor=True):
    rows = []
    if os.path.exists(path2labels):
        with open(path2labels, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                c, x, y, w, h = map(float, p[:5])
                rows.append([c, x * img_w, y * img_h, w * img_w, h * img_h])
    arr = np.array(rows, dtype=np.float32).reshape(-1, 5)
    return torch.from_numpy(arr) if is_torch_tensor else arr

def yolo_format_xcycwh_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """Функция для конвертации [x_center_norm, y_center_norm, w_norm, h_norm] to [x,y,x,y]"""
    result = bboxes.clone()
    x1 = bboxes[:, 0] - (bboxes[:, 2] / 2)
    y1 = bboxes[:, 1] - (bboxes[:, 3] / 2)
    x2 = bboxes[:, 0] + (bboxes[:, 2] / 2)
    y2 = bboxes[:, 1] + (bboxes[:, 3] / 2)
    
    result[:, 0] = x1
    result[:, 1] = y1
    result[:, 2] = x2
    result[:, 3] = y2
    return result