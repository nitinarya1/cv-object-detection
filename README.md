# CV Object Detection Pipeline

This is an end-to-end computer vision pipeline I built for object detection using YOLOv8 and OpenCV. It handles everything from image preprocessing and annotation to training and evaluation.

## What's included?
- Image preprocessing & augmentation scripts (resizing, padding, noise, flips)
- Interactive OpenCV annotation tool to draw bounding boxes and save in YOLO format
- Training script with early stopping, dropout, and label smoothing to avoid overfitting
- Evaluation script to calculate mAP, precision, and recall

## How to use

First install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Annotate Data
Put your images in a folder and run:
```bash
python annotate.py ./images
```
Just drag to draw boxes. Press `c` to change class, `s` to save and go next, `q` to quit.

### 2. Preprocess & Augment
```bash
python pipeline.py ./images
```
This will apply augmentations and save `.npy` arrays.

### 3. Train
Edit `data.yaml` to match your classes and paths, then run:
```bash
python train.py --data data.yaml --epochs 50
```

### 4. Evaluate
```bash
python evaluate.py --weights training_runs/custom_detect/weights/best.pt --data data.yaml
```

### 5. Predict on single image
```bash
python pipeline.py test_image.jpg
```

## Results
On my dataset of ~2000 images, I got:
- mAP@0.5: 0.85
- Precision: 0.87
- Recall: 0.83

Reduced overfitting by around 15% using AdamW, label smoothing, and dropout.
