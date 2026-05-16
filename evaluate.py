"""
evaluate.py  –  Model Evaluation Module
Computes Precision, Recall, mAP@0.5 and mAP@0.5:0.95 for a trained YOLOv8 model.
Usage:
    python evaluate.py --model yolov8n.pt --data data.yaml
"""

import argparse
import json
import numpy as np
from pathlib import Path


# ─── IoU Calculation ──────────────────────────────────────────────────────────
def compute_iou(box_pred: np.ndarray, box_gt: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt   = (box_gt[2]   - box_gt[0])   * (box_gt[3]   - box_gt[1])
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0.0


# ─── Precision & Recall ───────────────────────────────────────────────────────
def compute_precision_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5
) -> tuple[float, float]:
    """
    Compute precision and recall given predictions and ground truths.

    Each prediction / ground_truth dict should have keys:
        - "box"   : [x1, y1, x2, y2]
        - "class" : int
        - "conf"  : float (predictions only)
    """
    tp = fp = fn = 0
    matched_gt = set()

    # Sort by confidence descending
    preds_sorted = sorted(predictions, key=lambda x: x["conf"], reverse=True)

    for pred in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(ground_truths):
            if gt["class"] != pred["class"] or idx in matched_gt:
                continue
            iou = compute_iou(np.array(pred["box"]), np.array(gt["box"]))
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_iou >= iou_threshold and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(precision, 4), round(recall, 4)


# ─── AP Calculation (via 11-point interpolation) ──────────────────────────────
def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    """
    ap = 0.0
    for threshold in np.linspace(0, 1, 11):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= threshold]
        ap += max(prec_at_rec) if prec_at_rec else 0.0
    return round(ap / 11, 4)


# ─── mAP over multiple IoU thresholds ─────────────────────────────────────────
def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_thresholds: list[float] = None
) -> dict:
    """
    Compute mAP@0.5 and mAP@0.5:0.95 (COCO-style).
    Returns a dict with 'mAP_50' and 'mAP_50_95'.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

    aps = []
    for iou_thresh in iou_thresholds:
        prec_list, rec_list = [], []
        for conf_thresh in np.linspace(0.0, 1.0, 20):
            filtered = [p for p in predictions if p["conf"] >= conf_thresh]
            p, r = compute_precision_recall(filtered, ground_truths, iou_thresh)
            prec_list.append(p)
            rec_list.append(r)
        aps.append(compute_ap(prec_list, rec_list))

    map_50    = aps[0] if aps else 0.0
    map_50_95 = round(np.mean(aps), 4) if aps else 0.0

    return {"mAP_50": map_50, "mAP_50_95": map_50_95}


# ─── Full Evaluation with YOLOv8 val ──────────────────────────────────────────
def run_yolo_evaluation(model_path: str, data_yaml: str):
    """
    Run YOLOv8 model validation on a dataset. Requires ultralytics.
    pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=True)

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Precision   : {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall      : {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"  mAP@0.5     : {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print("=" * 50)

    # Save results to JSON
    results_path = Path("evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.results_dict.items()}, f, indent=2)
    print(f"[INFO] Results saved to {results_path}")

    return metrics


# ─── CLI Entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Model Evaluation")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights")
    parser.add_argument("--data",  type=str, default="data.yaml",  help="Path to dataset YAML")
    args = parser.parse_args()

    print(f"[INFO] Evaluating model: {args.model}")
    print(f"[INFO] Dataset config  : {args.data}")
    run_yolo_evaluation(args.model, args.data)
