"""
Анализ ошибок детектора YOLO (Ultralytics) на выбранном сплите: val / train / test.

Что делает
  1. Берёт список изображений так же, как Ultralytics (data.yaml -> txt -> images/labels).
  2. Прогоняет модель с очень низким порогом confidence и сохраняет ВСЕ предсказания.
  3. Сопоставляет предсказания с GT жадно, по убыванию confidence (как в COCO).
  4. Считает TP / FP / FN, Precision, Recall, F1, AP (IoU 0.1 / 0.3 / 0.5 / 0.75 и 0.5:0.95),
     mAP, IoU, FROC, разбивки по исходным классам тяжести, пациентам, размеру бокса, типы ошибок.
  5. Сохраняет пары картинок (GT | предсказания) для случайных кадров и для худших FN / FP.

Пример
  python analyze_errors.py data.yaml val runs/.../weights/best.pt 512 analysis exp1 \
      --conf 0.25 --meta data/meta_frame_mapped.csv --n-vis 50

Важно: AP здесь считается по собственной реализации (жадное сопоставление по каждому кадру, 101-точечная
интерполяция). Цифры близки к model.val() из Ultralytics, но могут слегка отличаться.
"""
import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

LOOSE_IOU = 0.1  # ниже этого IoU считаем, что рядом с боксом ничего нет
IOU_COCO = [round(float(x), 2) for x in np.arange(0.5, 0.96, 0.05)]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CADICA_ORDER = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]
CONF_GRID = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
COL = {"gt": (255, 160, 0), "tp": (0, 200, 0), "fp": (0, 0, 255), "fn": (0, 255, 255)}  # BGR
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ======================================================================
# Ввод / вывод
# ======================================================================
def imread_color(path):
    """cv2.imread, устойчивый к не-ASCII путям (Windows)."""
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read image: {path}")
    return img


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def collect_images(src):
    """Список изображений из директории / txt-файла / списка. Логика пути './' как в Ultralytics."""
    items = src if isinstance(src, (list, tuple)) else [src]
    files = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            parent = str(p.parent) + os.sep
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("./"):
                    line = parent + line[2:]
                files.append(str(Path(line)))
        else:
            raise FileNotFoundError(f"{item} not found")
    return files


def label_path(img_path):
    """images/ -> labels/, расширение -> .txt (так же, как img2label_paths в Ultralytics)."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return sb.join(img_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def load_gt(img_path, shape):
    """GT в пикселях xyxy, классы, нормализованные (cx, cy, w, h), найден ли label-файл."""
    h, w = shape
    lp = label_path(img_path)
    empty = (np.zeros((0, 4)), np.zeros(0, int), np.zeros((0, 4)))
    if not os.path.isfile(lp):
        return (*empty, False)
    rows = [ln.split() for ln in Path(lp).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        return (*empty, True)
    a = np.array([r[:5] for r in rows], dtype=float)
    cls, norm = a[:, 0].astype(int), a[:, 1:5]
    cx, cy, bw, bh = norm.T
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    return xyxy, cls, norm, True


def load_meta(path):
    """meta-CSV (frame, x_center_norm, ..., cls) -> {frame: (norm_boxes, orig_cls_names)}."""
    df = pd.read_csv(path)
    need = ["frame", "x_center_norm", "y_center_norm", "w_norm", "h_norm", "cls"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"meta csv has no columns: {miss}")
    df = df.dropna(subset=["x_center_norm"])
    return {
        str(k): (g[need[1:5]].to_numpy(float), g["cls"].astype(str).tolist())
        for k, g in df.groupby("frame")
    }


def assign_orig(stem, g_norm, meta, tol=1e-4):
    """Исходный класс тяжести (p50_70, ...) для каждого GT-бокса через совпадение координат с meta-CSV."""
    n = len(g_norm)
    if meta is None:
        return ["n/a"] * n
    if stem not in meta:
        return ["?"] * n
    arr, names = meta[stem]
    used, res = set(), []
    for row in g_norm:
        d = np.abs(arr - row).max(1)
        for u in used:
            d[u] = np.inf
        j = int(d.argmin())
        if d[j] < tol:
            res.append(names[j])
            used.add(j)
        else:
            res.append("?")
    return res


# ======================================================================
# Сопоставление предсказаний и GT
# ======================================================================
def box_iou_np(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Rec:
    path: str
    stem: str
    patient: str
    video: str
    shape: tuple
    p_xyxy: np.ndarray   # предсказания, отсортированы по убыванию conf
    p_conf: np.ndarray
    p_cls: np.ndarray
    g_xyxy: np.ndarray
    g_cls: np.ndarray
    g_orig: list         # исходный класс тяжести каждого GT
    tp: dict             # iou_thr -> bool[n_pred]
    miou: dict           # iou_thr -> IoU с сопоставленным GT (0, если не сопоставлен)
    g_mconf: dict        # iou_thr -> conf сопоставленного предсказания (-1, если GT пропущен)
    p_best: np.ndarray   # лучший IoU предсказания с любым GT того же класса
    g_best: np.ndarray   # лучший IoU GT с любым предсказанием (conf >= conf_eval) того же класса
    label_found: bool


def build_record(path, shape, p_xyxy, p_conf, p_cls, thrs, conf_eval, meta, class_agnostic):
    stem = Path(path).stem
    m = re.match(r"(p\d+)_(v\d+)_", stem)
    patient, video = (m.group(1), m.group(2)) if m else ("?", "?")

    order = np.argsort(-p_conf, kind="stable")
    p_xyxy, p_conf, p_cls = p_xyxy[order], p_conf[order], p_cls[order]
    g_xyxy, g_cls, g_norm, found = load_gt(path, shape)
    g_orig = assign_orig(stem, g_norm, meta)
    if class_agnostic:
        p_cls, g_cls = np.zeros_like(p_cls), np.zeros_like(g_cls)

    n_p, n_g = len(p_conf), len(g_cls)
    tp = {t: np.zeros(n_p, bool) for t in thrs}
    miou = {t: np.zeros(n_p) for t in thrs}
    gmc = {t: np.full(n_g, -1.0) for t in thrs}
    p_best, g_best = np.zeros(n_p), np.zeros(n_g)

    if n_p and n_g:
        iou = box_iou_np(p_xyxy, g_xyxy) * (p_cls[:, None] == g_cls[None, :])
        p_best = iou.max(1)
        sel = p_conf >= conf_eval
        if sel.any():
            g_best = iou[sel].max(0)
        for t in thrs:
            used = np.zeros(n_g, bool)
            for i in range(n_p):
                if p_best[i] < t:
                    continue
                cand = np.where(used, -1.0, iou[i])
                j = int(cand.argmax())
                if cand[j] >= t:
                    tp[t][i], miou[t][i], gmc[t][j], used[j] = True, cand[j], p_conf[i], True
                    if used.all():
                        break
    return Rec(path, stem, patient, video, tuple(shape), p_xyxy, p_conf, p_cls, g_xyxy, g_cls, g_orig,
               tp, miou, g_mconf=gmc, p_best=p_best, g_best=g_best, label_found=found)


# ======================================================================
# Агрегация и метрики
# ======================================================================
def flatten(recs, thrs):
    def cat(f, dtype):
        arrs = [f(r) for r in recs]
        return np.concatenate(arrs) if arrs else np.zeros(0, dtype)

    P = {
        "conf": cat(lambda r: r.p_conf, float),
        "cls": cat(lambda r: r.p_cls, int),
        "best": cat(lambda r: r.p_best, float),
        "img": np.concatenate([np.full(len(r.p_conf), i, int) for i, r in enumerate(recs)]),
        "tp": {t: cat(lambda r, t=t: r.tp[t], bool) for t in thrs},
        "miou": {t: cat(lambda r, t=t: r.miou[t], float) for t in thrs},
    }
    G = {
        "cls": cat(lambda r: r.g_cls, int),
        "xyxy": cat(lambda r: r.g_xyxy.reshape(-1, 4), float),
        "best": cat(lambda r: r.g_best, float),
        "orig": np.array([o for r in recs for o in r.g_orig], dtype=object),
        "img": np.concatenate([np.full(len(r.g_cls), i, int) for i, r in enumerate(recs)]),
        "mconf": {t: cat(lambda r, t=t: r.g_mconf[t], float) for t in thrs},
    }
    return P, G


def compute_ap(recall, precision):
    """101-точечная интерполяция, как в Ultralytics / COCO."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def ap_table(P, G, classes, thrs, names):
    rows = []
    for c in classes:
        mp = P["cls"] == c
        n_gt = int((G["cls"] == c).sum())
        if n_gt == 0:
            continue
        order = np.argsort(-P["conf"][mp], kind="stable")
        row = {"class": names.get(c, str(c)), "n_gt": n_gt}
        for t in thrs:
            tp = P["tp"][t][mp][order]
            if len(tp) == 0:
                row[t] = 0.0
                continue
            ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
            row[t] = compute_ap(ctp / n_gt, ctp / (ctp + cfp))
        rows.append(row)
    return rows


def op_row(P, G, n_img, t, c):
    m = P["conf"] >= c
    tp = int(P["tp"][t][m].sum())
    fp = int(m.sum()) - tp
    n_gt = len(G["cls"])
    fn = n_gt - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mi = float(P["miou"][t][m & P["tp"][t]].mean()) if tp else float("nan")
    return {"iou_thr": t, "conf": c, "TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec,
            "F1": f1, "FP_per_img": fp / n_img, "mean_IoU_of_TP": mi}


def fmt(df):
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def all_thrs(args):
    return sorted(set(IOU_COCO + [0.1, 0.3, 0.5, args.iou]))


def report(recs, names, args, out):
    conf, thr = args.conf, args.iou
    rep_thrs = sorted({0.1, 0.3, 0.5, thr})
    thrs = all_thrs(args)
    n_img = len(recs)
    P, G = flatten(recs, thrs)
    n_gt = len(G["cls"])
    patient_of_img = np.array([r.patient for r in recs])
    summary = {"n_images": n_img, "n_gt": n_gt, "n_pred_total": len(P["conf"]),
               "conf_eval": conf, "iou_main": thr, "args": vars(args)}

    # ---------- AP ----------
    classes = sorted(set(G["cls"].tolist()) | set(P["cls"].tolist()))
    ap_df = pd.DataFrame(ap_table(P, G, classes, thrs, names))
    ap_out = pd.DataFrame({
        "class": ap_df["class"], "n_gt": ap_df["n_gt"],
        "AP@0.10": ap_df[0.1], "AP@0.30": ap_df[0.3], "AP@0.50": ap_df[0.5], "AP@0.75": ap_df[0.75],
        "AP@0.50:0.95": ap_df[IOU_COCO].mean(axis=1),
    })
    mean_row = {"class": "ALL (mean over classes)", "n_gt": n_gt,
                **ap_out.drop(columns=["class", "n_gt"]).mean().to_dict()}
    ap_out = pd.concat([ap_out, pd.DataFrame([mean_row])], ignore_index=True)
    ap_out.to_csv(out / "metrics_ap.csv", index=False)
    summary["ap"] = ap_out.to_dict("records")

    # ---------- рабочая точка ----------
    op = pd.DataFrame([op_row(P, G, n_img, t, conf) for t in rep_thrs])
    op.to_csv(out / "operating_point.csv", index=False)
    summary["operating_point"] = op.to_dict("records")

    # ---------- свип по порогам confidence ----------
    grid = [c for c in CONF_GRID if c >= args.pred_conf] or [conf]
    sweep = pd.DataFrame([op_row(P, G, n_img, t, c) for t in rep_thrs for c in grid])
    sweep.to_csv(out / "sweep_conf.csv", index=False)
    best = {}
    for t in rep_thrs:
        s = sweep[sweep.iou_thr == t]
        b = s.loc[s["F1"].idxmax()]
        best[str(t)] = {"conf": float(b.conf), "F1": float(b.F1), "precision": float(b.precision),
                        "recall": float(b.recall), "FP_per_img": float(b.FP_per_img)}
    summary["best_F1_over_grid"] = best

    # ---------- по кадрам ----------
    rows = []
    for i, r in enumerate(recs):
        m = r.p_conf >= conf
        tp = int((r.tp[thr] & m).sum())
        fp = int(m.sum()) - tp
        fn = int((r.g_mconf[thr] < conf).sum())
        fpc = r.p_conf[m & ~r.tp[thr]]
        gg = box_iou_np(r.g_xyxy, r.g_xyxy)
        pairs = int((np.triu(gg, 1) > 0.5).sum()) if len(gg) > 1 else 0
        rows.append({"image": r.path, "stem": r.stem, "patient": r.patient, "video": r.video,
                     "n_gt": len(r.g_cls), "n_pred": int(m.sum()), "TP": tp, "FP": fp, "FN": fn,
                     "max_fp_conf": float(fpc.max()) if len(fpc) else 0.0, "gt_pairs_iou>0.5": pairs})
    pi = pd.DataFrame(rows)
    pi.to_csv(out / "per_image.csv", index=False)

    pos, hit = pi.n_gt > 0, pi.n_pred > 0
    summary["frame_level"] = {
        "frames_with_gt": int(pos.sum()), "frames_without_gt": int((~pos).sum()),
        "frames_gt_and_pred": int((pos & hit).sum()), "frames_gt_no_pred(FN)": int((pos & ~hit).sum()),
        "frames_no_gt_but_pred(FP)": int((~pos & hit).sum()), "frames_no_gt_no_pred(TN)": int((~pos & ~hit).sum()),
        "frame_sensitivity": float((pos & hit).sum() / max(pos.sum(), 1)),
        "false_alarm_rate_on_negatives": float((~pos & hit).sum() / max((~pos).sum(), 1)),
    }

    # ---------- типы ошибок ----------
    m = P["conf"] >= conf
    fp_mask = m & ~P["tp"][thr]
    fn_mask = G["mconf"][thr] < conf
    errors = {
        "TP": int((m & P["tp"][thr]).sum()),
        "FP_total": int(fp_mask.sum()),
        "FP_background (IoU<0.1 с любым GT)": int((fp_mask & (P["best"] < LOOSE_IOU)).sum()),
        f"FP_localization (0.1<=IoU<{thr})": int((fp_mask & (P["best"] >= LOOSE_IOU) & (P["best"] < thr)).sum()),
        f"FP_duplicate (IoU>={thr}, GT уже занят)": int((fp_mask & (P["best"] >= thr)).sum()),
        "FN_total": int(fn_mask.sum()),
        "FN_missed (рядом нет предсказаний)": int((fn_mask & (G["best"] < LOOSE_IOU)).sum()),
        f"FN_localization (0.1<=IoU<{thr})": int((fn_mask & (G["best"] >= LOOSE_IOU) & (G["best"] < thr)).sum()),
        f"FN_shared (IoU>={thr}, предсказание ушло другому GT)": int((fn_mask & (G["best"] >= thr)).sum()),
    }
    summary["error_types"] = errors
    pd.Series(errors, name="count").to_csv(out / "error_types.csv", header=True)

    # ---------- IoU / локализация ----------
    summary["localization"] = {
        "mean_best_IoU_per_GT(conf>=conf_eval)": float(G["best"].mean()) if n_gt else 0.0,
        "median_best_IoU_per_GT": float(np.median(G["best"])) if n_gt else 0.0,
        "mean_IoU_of_TP": float(P["miou"][thr][m & P["tp"][thr]].mean()) if errors["TP"] else None,
    }

    # ---------- разбивки по GT ----------
    wh = G["xyxy"][:, 2:] - G["xyxy"][:, :2]
    size = np.sqrt(wh[:, 0] * wh[:, 1])
    gd = pd.DataFrame({"orig_cls": G["orig"], "best_iou": G["best"], "size_px": size,
                       **{f"hit@{t}": G["mconf"][t] >= conf for t in rep_thrs}})

    def recall_table(key):
        agg = gd.groupby(key, observed=True).agg(
            n_gt=("best_iou", "size"), mean_best_IoU=("best_iou", "mean"),
            **{f"recall@{t}": (f"hit@{t}", "mean") for t in rep_thrs})
        return agg

    by_cls = recall_table("orig_cls")
    by_cls = by_cls.reindex([c for c in CADICA_ORDER if c in by_cls.index] +
                            [c for c in by_cls.index if c not in CADICA_ORDER])
    by_cls.to_csv(out / "recall_by_orig_cls.csv")
    summary["recall_by_orig_cls"] = by_cls.reset_index().to_dict("records")

    if n_gt >= 8:
        gd["size_bin"] = pd.qcut(gd["size_px"], 4, duplicates="drop").astype(str)
        by_size = recall_table("size_bin")
        by_size.to_csv(out / "recall_by_size.csv")
        summary["recall_by_size"] = by_size.reset_index().to_dict("records")
    else:
        by_size = None

    by_pat = pi.groupby("patient").agg(frames=("stem", "size"), n_gt=("n_gt", "sum"),
                                        TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"))
    by_pat["recall"] = by_pat.TP / by_pat.n_gt.replace(0, np.nan)
    by_pat["FP_per_img"] = by_pat.FP / by_pat.frames
    by_pat.to_csv(out / "by_patient.csv")

    # ---------- статистика GT ----------
    summary["gt_stats"] = {
        "w_px_percentiles(5,25,50,75,95)": np.percentile(wh[:, 0], [5, 25, 50, 75, 95]).round(1).tolist() if n_gt else [],
        "h_px_percentiles(5,25,50,75,95)": np.percentile(wh[:, 1], [5, 25, 50, 75, 95]).round(1).tolist() if n_gt else [],
        "gt_pairs_with_IoU>0.5_in_same_frame": int(pi["gt_pairs_iou>0.5"].sum()),
        "frames_with_such_pairs": int((pi["gt_pairs_iou>0.5"] > 0).sum()),
        "frames_without_label_file": int(sum(not r.label_found for r in recs)),
        "gt_without_orig_cls_match(?)": int((G["orig"] == "?").sum()),
    }

    # ---------- CSV с детекциями и GT ----------
    det_rows = []
    for r in recs:
        for i in np.where(r.p_conf >= args.det_csv_conf)[0]:
            det_rows.append({"image": r.path, "stem": r.stem, "patient": r.patient, "video": r.video,
                             "cls": int(r.p_cls[i]), "conf": float(r.p_conf[i]),
                             "x1": r.p_xyxy[i, 0], "y1": r.p_xyxy[i, 1], "x2": r.p_xyxy[i, 2], "y2": r.p_xyxy[i, 3],
                             "best_iou_gt": float(r.p_best[i]),
                             **{f"tp@{t}": bool(r.tp[t][i]) for t in rep_thrs}})
    pd.DataFrame(det_rows).to_csv(out / "detections.csv", index=False)
    gt_rows = []
    for r in recs:
        for j in range(len(r.g_cls)):
            gt_rows.append({"image": r.path, "stem": r.stem, "patient": r.patient, "video": r.video,
                            "cls": int(r.g_cls[j]), "orig_cls": r.g_orig[j],
                            "x1": r.g_xyxy[j, 0], "y1": r.g_xyxy[j, 1], "x2": r.g_xyxy[j, 2], "y2": r.g_xyxy[j, 3],
                            "w": r.g_xyxy[j, 2] - r.g_xyxy[j, 0], "h": r.g_xyxy[j, 3] - r.g_xyxy[j, 1],
                            "best_iou": float(r.g_best[j]),
                            **{f"match_conf@{t}": float(r.g_mconf[t][j]) for t in rep_thrs}})
    pd.DataFrame(gt_rows).to_csv(out / "gt_boxes.csv", index=False)

    # ---------- кривые ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        order = np.argsort(-P["conf"], kind="stable")
        for t in rep_thrs:
            tp = P["tp"][t][order]
            ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
            rec = ctp / max(n_gt, 1)
            ax[0].plot(rec, ctp / np.maximum(ctp + cfp, 1), label=f"IoU>={t}")
            ax[1].plot(cfp / n_img, rec, label=f"IoU>={t}")
        ax[0].set(xlabel="Recall", ylabel="Precision", title="PR (все классы, conf>=pred_conf)", xlim=(0, 1), ylim=(0, 1.02))
        ax[1].set(xlabel="FP на кадр", ylabel="Recall", title="FROC", xlim=(0, 5), ylim=(0, 1.02))
        for a in ax:
            a.grid(alpha=0.3)
            a.legend()
        fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # график не критичен
        print(f"[warn] curves.png not saved: {e}")

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    # ---------- печать ----------
    print(f"\n===== {n_img} кадров, {n_gt} GT-боксов, conf>={conf}, основной IoU>={thr} =====")
    print("\n--- AP ---")
    print(fmt(ap_out))
    print("\n--- Рабочая точка (TP/FP/FN, P, R, F1, IoU) ---")
    print(fmt(op))
    print("\n--- Лучший F1 по сетке confidence ---")
    for t, v in best.items():
        print(f"IoU>={t}: conf={v['conf']}  F1={v['F1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  FP/кадр={v['FP_per_img']:.2f}")
    print("\n--- Типы ошибок ---")
    for k, v in errors.items():
        print(f"{k}: {v}")
    print("\n--- Локализация ---")
    print(summary["localization"])
    print("\n--- Recall по исходному классу ---")
    print(fmt(by_cls.reset_index()))
    if by_size is not None:
        print("\n--- Recall по размеру бокса (sqrt(w*h), px, квартили) ---")
        print(fmt(by_size.reset_index()))
    print("\n--- Уровень кадра ---")
    print(summary["frame_level"])
    print("\n--- Статистика GT ---")
    print(summary["gt_stats"])
    return pi


# ======================================================================
# Визуализация
# ======================================================================
def draw_box(img, xyxy, color, text, s, bar):
    x1, y1, x2, y2 = [int(round(v * s)) for v in xyxy]
    y1, y2 = y1 + bar, y2 + bar
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    if text:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.4, 1)
        ty = y1 - 3 if y1 - th - 3 > bar else y2 + th + 3
        cv2.putText(img, text, (x1, ty), FONT, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x1, ty), FONT, 0.4, color, 1, cv2.LINE_AA)


def put_parts(img, parts, x, y, scale=0.45):
    for text, color in parts:
        cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, FONT, scale, 1)[0][0]


def render_pair(rec, args, names):
    img = imread_color(rec.path)
    s = float(args.vis_scale)
    if s != 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    bar = 24
    base = cv2.copyMakeBorder(img, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gt_img, pr_img = base.copy(), base.copy()
    thr, conf = args.iou, args.conf

    for xyxy, orig in zip(rec.g_xyxy, rec.g_orig):
        draw_box(gt_img, xyxy, COL["gt"], "" if orig == "n/a" else orig, s, bar)
    put_parts(gt_img, [(f"{rec.stem}  |  GT boxes: {len(rec.g_cls)}", WHITE)], 4, 16)

    n_tp = n_fp = n_fn = 0
    multi = len(names) > 1
    for i in np.where(rec.p_conf >= conf)[0]:
        ok = bool(rec.tp[thr][i])
        n_tp, n_fp = n_tp + ok, n_fp + (not ok)
        label = f"{rec.p_conf[i]:.2f}" + (f" {names.get(int(rec.p_cls[i]), '')}" if multi else "")
        draw_box(pr_img, rec.p_xyxy[i], COL["tp" if ok else "fp"], label, s, bar)
    for j in np.where(rec.g_mconf[thr] < conf)[0]:
        n_fn += 1
        orig = rec.g_orig[j]
        draw_box(pr_img, rec.g_xyxy[j], COL["fn"], "FN" + ("" if orig == "n/a" else f" {orig}"), s, bar)
    put_parts(pr_img, [(f"{rec.stem}  conf>={conf} IoU>={thr}  ", WHITE), (f"TP {n_tp} ", COL["tp"]),
                       (f"FP {n_fp} ", COL["fp"]), (f"FN {n_fn}", COL["fn"])], 4, 16)
    return gt_img, pr_img


def save_pairs(recs, idxs, out_dir, args, names, ranked=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idxs):
        rec = recs[i]
        gt_img, pr_img = render_pair(rec, args, names)
        prefix = f"{k + 1:02d}_{rec.stem}" if ranked else rec.stem
        imwrite(out_dir / f"{prefix}_gt.png", gt_img)
        imwrite(out_dir / f"{prefix}_pred.png", pr_img)
        if args.pair:
            imwrite(out_dir / f"{prefix}_pair.png", np.hstack([gt_img, pr_img]))


def visualize(recs, pi, names, args, out):
    rng = random.Random(args.seed)
    all_idx = list(range(len(recs)))
    if args.n_vis > 0:
        pick = rng.sample(all_idx, min(args.n_vis, len(recs)))
        save_pairs(recs, pick, out / "vis_random", args, names)
    if args.n_fn > 0:
        cand = [i for i in all_idx if pi.FN[i] > 0]
        cand.sort(key=lambda i: (-pi.FN[i], rng.random()))
        save_pairs(recs, cand[:args.n_fn], out / "vis_worst_fn", args, names, ranked=True)
    if args.n_fp > 0:
        cand = [i for i in all_idx if pi.FP[i] > 0]
        cand.sort(key=lambda i: (-pi.max_fp_conf[i], rng.random()))
        save_pairs(recs, cand[:args.n_fp], out / "vis_worst_fp", args, names, ranked=True)


# ======================================================================
# main
# ======================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Анализ ошибок детектора YOLO: метрики + картинки GT / предсказания")
    p.add_argument("path2data_file", type=str, help="path to data.yaml")
    p.add_argument("split", type=str, help="val / train / test")
    p.add_argument("weights", type=str, help="path to weights")
    p.add_argument("imgsize", type=int, help="image size")
    p.add_argument("project", type=str, help="корневая папка результатов")
    p.add_argument("exp_name", type=str, help="имя анализа (подпапка)")
    p.add_argument("--conf", type=float, default=0.25, help="порог confidence рабочей точки (TP/FP/FN, картинки)")
    p.add_argument("--iou", type=float, default=0.5, help="основной порог IoU для TP (картинки, типы ошибок)")
    p.add_argument("--pred-conf", type=float, default=0.001, help="минимальный conf при инференсе (для AP)")
    p.add_argument("--nms-iou", type=float, default=0.6, help="IoU для NMS")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("-bs", "--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--meta", type=str, default=None, help="meta-CSV с колонкой cls (исходные классы тяжести)")
    p.add_argument("--class-agnostic", action="store_true", help="не учитывать класс при сопоставлении")
    p.add_argument("--n-vis", type=int, default=50, help="случайных кадров для картинок (0 - выключить)")
    p.add_argument("--n-fn", type=int, default=20, help="кадров с наибольшим числом FN")
    p.add_argument("--n-fp", type=int, default=20, help="кадров с самыми уверенными FP")
    p.add_argument("--vis-scale", type=float, default=2.0, help="увеличение кадров на картинках")
    p.add_argument("--pair", action="store_true", help="дополнительно сохранять склейку GT|pred")
    p.add_argument("--det-csv-conf", type=float, default=0.01, help="минимальный conf для записи в detections.csv")
    p.add_argument("--limit", type=int, default=0, help="взять случайные N кадров (быстрый тест)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def get_model_and_paths(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(args.path2data_file)
    src = data.get(args.split)
    if not src:
        raise ValueError(f"split '{args.split}' not found in {args.path2data_file}")
    return YOLO(args.weights), collect_images(src)


def main(argv=None):
    args = parse_args(argv)
    args.iou = round(args.iou, 2)
    out = Path(args.project) / args.exp_name
    out.mkdir(parents=True, exist_ok=True)

    model, paths = get_model_and_paths(args)
    if args.limit and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    print(f"images: {len(paths)} ({args.split})")

    meta = load_meta(args.meta) if args.meta else None
    names = {0: "all"} if args.class_agnostic else dict(model.names)
    thrs = all_thrs(args)
    kw = {"device": args.device} if args.device else {}

    recs = []
    pbar = tqdm(total=len(paths), desc="predict")
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        imgs = [imread_color(p) for p in chunk]
        results = model.predict(source=imgs, imgsz=args.imgsize, conf=args.pred_conf, iou=args.nms_iou,
                                max_det=args.max_det, verbose=False, **kw)
        for p, im, r in zip(chunk, imgs, results):
            b = r.boxes
            recs.append(build_record(
                p, im.shape[:2], b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float),
                b.cls.cpu().numpy().astype(int), thrs, args.conf, meta, args.class_agnostic))
        pbar.update(len(chunk))
    pbar.close()

    if sum(len(r.g_cls) for r in recs) == 0:
        raise SystemExit("В выбранном сплите нет ни одного GT-бокса: проверьте пути images/labels.")
    missing = sum(not r.label_found for r in recs)
    if missing:
        print(f"[warn] нет label-файла для {missing} кадров (считаются фоном, как в Ultralytics)")

    pi = report(recs, names, args, out)
    visualize(recs, pi, names, args, out)
    print(f"\nГотово. Результаты: {out.resolve()}")
    print("Цвета на *_pred.png: зелёный = TP, красный = FP, жёлтый = FN (пропущенный GT). На *_gt.png: голубой = GT.")


if __name__ == "__main__":
    main()