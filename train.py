import argparse
import os

def start_training(data_cfg, epochs=50, weights="yolov8n.pt", batch=16):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("need ultralytics installed")
        return

    print(f"Training {weights} on {data_cfg} for {epochs} epochs")
    
    model = YOLO(weights)
    
    # train model with some anti-overfitting tricks
    model.train(
        data=data_cfg,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        patience=10, # early stopping
        augment=True,
        dropout=0.1,
        label_smoothing=0.1,
        optimizer="AdamW",
        lr0=0.001,
        save=True,
        project="training_runs",
        name="custom_detect"
    )
    
    print("Training finished!")
    print("Best weights should be in training_runs/custom_detect/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    
    args = parser.parse_args()
    start_training(args.data, args.epochs, args.weights)
