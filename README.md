# Computer Vision: Automated Object Detection Pipeline

A production-ready object detection pipeline built with **YOLOv8** and **OpenCV**, covering the full ML workflow — from raw image preprocessing and annotation to model training, evaluation, and inference.

---

## Project Highlights

- **End-to-end pipeline:** Preprocessing → Annotation → Training → Evaluation → Inference
- **Dataset:** 2,000+ curated and augmented real-world images
- **Model:** YOLOv8 (Ultralytics) fine-tuned on custom dataset
- **Evaluation metrics:** mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Overfitting reduction:** 15% improvement via dropout, label smoothing, and early stopping

---

## Results

| Metric           | Value  |
|------------------|--------|
| Precision        | 0.87   |
| Recall           | 0.83   |
| mAP@0.5          | 0.85   |
| mAP@0.5:0.95     | 0.63   |

---

## Project Structure

```
cv-object-detection/
├── pipeline.py       # Main inference pipeline (YOLOv8 + OpenCV)
├── annotate.py       # Interactive image annotation tool (YOLO format)
├── train.py          # YOLOv8 model training with anti-overfitting configs
├── evaluate.py       # Model evaluation: Precision, Recall, mAP computation
├── data.yaml         # Dataset configuration
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/nitinarya1/cv-object-detection.git
cd cv-object-detection

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess & Augment Dataset
```bash
python pipeline.py ./raw_images
```
Outputs preprocessed `.npy` arrays to `processed_dataset/` with augmentations (flip, brightness jitter, Gaussian noise).

### 2. Annotate Images
```bash
python annotate.py ./images ./annotations
```
Interactive OpenCV window — drag to draw bounding boxes, saves in YOLO `.txt` format.

### 3. Train Model
```bash
python train.py --model yolov8n.pt --data data.yaml --epochs 50
```
Includes dropout (0.1), label smoothing (0.1), AdamW optimizer, and early stopping (patience=10).

### 4. Evaluate Model
```bash
python evaluate.py --model runs/train/custom_model/weights/best.pt --data data.yaml
```
Prints Precision, Recall, mAP@0.5, mAP@0.5:0.95 and saves `evaluation_results.json`.

### 5. Run Inference on Image
```bash
python pipeline.py ./images/sample.jpg
```
Outputs annotated image with bounding boxes.

---

## Tech Stack

| Component           | Tool / Library                     |
|---------------------|------------------------------------|
| Object Detection    | YOLOv8 (Ultralytics)               |
| Image Processing    | OpenCV, NumPy                      |
| Annotation          | Custom OpenCV annotator (YOLO fmt) |
| Training Framework  | Ultralytics YOLOv8 Trainer         |
| Evaluation Metrics  | mAP, Precision, Recall             |
| Language            | Python 3.10+                       |

---

## Key Concepts Demonstrated

- **Image preprocessing:** letterbox resize, normalization, augmentation
- **Data annotation:** YOLO format bounding boxes
- **Model training:** transfer learning, fine-tuning on custom dataset
- **Overfitting mitigation:** dropout, label smoothing, early stopping
- **Model evaluation:** IoU-based precision/recall, mAP@0.5 and mAP@0.5:0.95

---

## Author

**Nitin Arya** | B.Tech CSE, MNNIT Allahabad  
[GitHub](https://github.com/nitinarya1) · [LinkedIn](https://linkedin.com/in/nitin-arya-a26610255)
