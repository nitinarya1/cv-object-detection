import argparse
import json
import numpy as np
import os

def calc_iou(boxA, boxB):
    # compute IoU
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


def get_pr_curve(preds, gts, iou_thresh=0.5):
    tp = 0
    fp = 0
    matched = []

    # sort predictions by confidence
    preds = sorted(preds, key=lambda x: x.get("conf", 0), reverse=True)

    for p in preds:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gts):
            if gt["class"] != p["class"] or i in matched:
                continue
            
            iou = calc_iou(p["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched.append(best_gt_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def calculate_ap(precisions, recalls):
    # 11 point AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_t) if p_at_t else 0.0
    return ap / 11


def run_eval(model_path, data_yaml):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("install ultralytics first")
        return

    model = YOLO(model_path)
    print("running validation...")
    val_res = model.val(data=data_yaml, verbose=False)

    print("\n--- Evaluation Results ---")
    print(f"Precision: {val_res.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall: {val_res.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP@50: {val_res.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@50-95: {val_res.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    
    with open("results.json", "w") as f:
        json.dump({k: float(v) for k, v in val_res.results_dict.items()}, f, indent=4)
    print("saved results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str, default="data.yaml")
    args = parser.parse_args()

    run_eval(args.weights, args.data)
