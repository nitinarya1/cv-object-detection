import cv2
import numpy as np
from pathlib import Path
import time

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_WEIGHTS = "yolov8n.pt"   # or yolov8s.pt / yolov8m.pt
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# ─── Preprocessing Pipeline ───────────────────────────────────────────────────
def preprocess_image(image_path: str, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Load and preprocess a single image for YOLOv8 inference.
    Steps: read → resize (letterbox) → BGR→RGB → normalize → expand dims.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Letterbox resize: maintain aspect ratio with padding
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    return img_norm


def augment_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Apply basic augmentations: horizontal flip, brightness jitter, Gaussian noise.
    Returns a list of augmented images.
    """
    augmented = [img]

    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    # Brightness jitter (±30%)
    brightness_factor = np.random.uniform(0.7, 1.3)
    bright = np.clip(img * brightness_factor, 0, 1.0).astype(np.float32)
    augmented.append(bright)

    # Gaussian noise
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 1.0).astype(np.float32)
    augmented.append(noisy)

    return augmented


def preprocess_dataset(dataset_dir: str, output_dir: str, img_size: int = IMG_SIZE):
    """
    Process all images in dataset_dir: preprocess + augment and save to output_dir.
    """
    src = Path(dataset_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in src.rglob("*") if f.suffix.lower() in image_extensions]
    print(f"[INFO] Found {len(images)} images in dataset.")

    processed = 0
    for img_path in images:
        try:
            img = preprocess_image(str(img_path), img_size)
            augmented = augment_image(img)

            for i, aug_img in enumerate(augmented):
                out_name = f"{img_path.stem}_aug{i}.npy"
                np.save(dst / out_name, aug_img)
                processed += 1
        except Exception as e:
            print(f"[WARN] Skipped {img_path.name}: {e}")

    print(f"[INFO] Preprocessed and saved {processed} arrays to '{output_dir}'.")
    return processed


# ─── Inference with YOLOv8 ────────────────────────────────────────────────────
def run_inference(image_path: str, model_path: str = MODEL_WEIGHTS):
    """
    Run YOLOv8 inference on a single image. Requires ultralytics to be installed.
    pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return None

    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )
    return results


def draw_detections(image_path: str, results) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    """
    img = cv2.imread(image_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{result.names[cls_id]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    return img


# ─── Main Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path_or_dataset_dir>")
        print("Example: python pipeline.py images/sample.jpg")
        sys.exit(0)

    input_path = Path(sys.argv[1])

    if input_path.is_dir():
        # Batch preprocessing mode
        preprocess_dataset(str(input_path), output_dir="processed_dataset")
    else:
        # Single image inference mode
        print(f"[INFO] Running inference on: {input_path}")
        start = time.time()
        results = run_inference(str(input_path))
        elapsed = time.time() - start

        if results:
            annotated = draw_detections(str(input_path), results)
            out_path = f"output_{input_path.name}"
            cv2.imwrite(out_path, annotated)
            print(f"[INFO] Inference done in {elapsed:.2f}s → saved to '{out_path}'")
        else:
            print("[ERROR] Inference failed.")
