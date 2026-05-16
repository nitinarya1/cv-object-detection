"""
train.py  –  YOLOv8 Training Script
Fine-tunes a YOLOv8 model on a custom dataset to reduce overfitting via callbacks.
Usage:
    python train.py --data data.yaml --epochs 50 --model yolov8n.pt
"""

import argparse
from pathlib import Path


def train(model_path: str, data_yaml: str, epochs: int = 50,
          img_size: int = 640, batch_size: int = 16, output_dir: str = "runs/train"):
    """
    Train a YOLOv8 model with early stopping and augmentation to prevent overfitting.
    Requires: pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    model = YOLO(model_path)
    print(f"[INFO] Starting training: {model_path} | {data_yaml} | {epochs} epochs")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name="custom_model",
        patience=10,         # Early stopping: stop if no improvement for 10 epochs
        augment=True,        # Enable built-in augmentation to reduce overfitting
        dropout=0.1,         # Dropout regularization
        label_smoothing=0.1, # Label smoothing to reduce overfitting
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        plots=True,          # Save training plots (loss curves, etc.)
        save=True,
        verbose=True
    )

    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    best_model = Path(output_dir) / "custom_model" / "weights" / "best.pt"
    print(f"  Best model : {best_model}")
    print(f"  Final mAP@0.5     : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Final mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("=" * 50)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Custom Training")
    parser.add_argument("--model",  type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--data",   type=str, default="data.yaml",  help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50,           help="Training epochs")
    parser.add_argument("--batch",  type=int, default=16,           help="Batch size")
    parser.add_argument("--imgsz",  type=int, default=640,          help="Image size")
    args = parser.parse_args()

    train(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch
    )
