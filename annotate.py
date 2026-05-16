"""
annotate.py  –  Dataset Annotation Helper
A lightweight tool to annotate images with bounding boxes in YOLO format.
Uses OpenCV for interactive annotation via mouse events.

Controls:
    Left-click + drag  → Draw bounding box
    'c'                → Change class (0–9)
    's'                → Save annotations for current image
    'n'                → Next image
    'q'                → Quit
"""

import cv2
import numpy as np
from pathlib import Path
import json

# ─── Annotation State ─────────────────────────────────────────────────────────
annotations: list[dict] = []
drawing = False
start_x = start_y = 0
current_class = 0
current_box: list = []

CLASS_NAMES = ["person", "car", "bicycle", "motorcycle", "truck",
               "bus", "cat", "dog", "chair", "bottle"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (128, 0, 128), (255, 165, 0),
          (0, 128, 128), (128, 128, 0)]


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = [start_x, start_y, x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            annotations.append({
                "class": current_class,
                "class_name": CLASS_NAMES[current_class],
                "box": [x1, y1, x2, y2]
            })
            print(f"  Added: {CLASS_NAMES[current_class]} [{x1},{y1},{x2},{y2}]")
        current_box = []


def convert_to_yolo(box: list, img_w: int, img_h: int) -> tuple:
    """Convert [x1,y1,x2,y2] to YOLO normalized format (cx,cy,w,h)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def save_annotations(image_path: str, img_shape: tuple, output_dir: Path):
    """Save annotations to both YOLO .txt format and JSON format."""
    img_h, img_w = img_shape[:2]
    stem = Path(image_path).stem

    # YOLO format .txt
    yolo_lines = []
    for ann in annotations:
        cx, cy, w, h = convert_to_yolo(ann["box"], img_w, img_h)
        yolo_lines.append(f"{ann['class']} {cx} {cy} {w} {h}")

    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(yolo_lines))

    # JSON format (for easy inspection)
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump({"image": str(image_path), "annotations": annotations}, f, indent=2)

    print(f"[SAVED] {len(annotations)} annotations → {txt_path.name}, {json_path.name}")


def annotate_dataset(image_dir: str, output_dir: str = "annotations"):
    """
    Interactively annotate all images in image_dir.
    """
    global annotations, current_class

    src = Path(image_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in src.rglob("*")
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not images:
        print(f"[ERROR] No images found in '{image_dir}'")
        return

    print(f"[INFO] {len(images)} images to annotate.")
    print("[INFO] Controls: drag=draw box | 'c'=change class | 's'=save | 'n'=next | 'q'=quit")

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotator", mouse_callback)

    for img_path in images:
        annotations = []
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        print(f"\n[IMAGE] {img_path.name}  (class: {CLASS_NAMES[current_class]})")

        while True:
            display = img.copy()

            # Draw existing annotations
            for ann in annotations:
                x1, y1, x2, y2 = ann["box"]
                color = COLORS[ann["class"]]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, ann["class_name"], (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw live box
            if current_box:
                cv2.rectangle(display, (current_box[0], current_box[1]),
                              (current_box[2], current_box[3]),
                              COLORS[current_class], 2)

            # HUD
            hud = f"Class: {CLASS_NAMES[current_class]} ({current_class}) | Boxes: {len(annotations)}"
            cv2.putText(display, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Annotator", display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('c'):
                current_class = (current_class + 1) % len(CLASS_NAMES)
                print(f"  Class → {CLASS_NAMES[current_class]}")
            elif key == ord('s'):
                save_annotations(str(img_path), img.shape, dst)
            elif key == ord('n') or key == ord('s'):
                save_annotations(str(img_path), img.shape, dst)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n[DONE] Annotation session complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python annotate.py <image_directory> [output_directory]")
        print("Example: python annotate.py ./images ./annotations")
    else:
        img_dir  = sys.argv[1]
        out_dir  = sys.argv[2] if len(sys.argv) > 2 else "annotations"
        annotate_dataset(img_dir, out_dir)
