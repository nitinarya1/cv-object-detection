import cv2
import numpy as np
import os
import time
import sys

# config stuff
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.40
IOU_THRESH = 0.45
IMG_SIZE = 640

def preprocess_image(img_path, target_size=IMG_SIZE):
    # loads and resizes img for yolo
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return None

    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    
    img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # pad image
    pad_h = (target_size - nh) // 2
    pad_w = (target_size - nw) // 2
    img_pad = cv2.copyMakeBorder(
        img_res, pad_h, target_size - nh - pad_h,
        pad_w, target_size - nw - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # convert to rgb and normalize
    img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def augment_img(img):
    # some basic augmentations
    augs = [img]

    # flip
    augs.append(cv2.flip(img, 1))

    # brightness
    bf = np.random.uniform(0.7, 1.3)
    augs.append(np.clip(img * bf, 0, 1.0).astype(np.float32))

    # noise
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    augs.append(np.clip(img + noise, 0, 1.0).astype(np.float32))

    return augs


def process_dataset(src_dir, out_dir, img_size=IMG_SIZE):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    valid_exts = [".jpg", ".jpeg", ".png"]
    images = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_exts):
                images.append(os.path.join(root, file))
                
    print(f"Found {len(images)} images to process")

    count = 0
    for path in images:
        try:
            img = preprocess_image(path, img_size)
            if img is None: continue
            
            augs = augment_img(img)
            fname = os.path.splitext(os.path.basename(path))[0]
            
            for i, aug_img in enumerate(augs):
                out_name = f"{fname}_aug{i}.npy"
                np.save(os.path.join(out_dir, out_name), aug_img)
                count += 1
        except Exception as e:
            print(f"failed on {path}: {e}")

    print(f"Done! Saved {count} processed images to {out_dir}")
    return count

def run_prediction(img_path, model_path=MODEL_WEIGHTS):
    # run inference
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return None

    model = YOLO(model_path)
    res = model.predict(
        source=img_path,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMG_SIZE,
        verbose=False
    )
    return res

def draw_boxes(img_path, results):
    img = cv2.imread(img_path)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = int(box.cls[0])
            conf = float(box.conf[0])
            
            label = f"{r.names[c]} {conf:.2f}"
            
            # draw box and text
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python pipeline.py <img_path_or_dir>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        process_dataset(path, "processed_data")
    else:
        print(f"Running model on: {path}")
        t0 = time.time()
        res = run_prediction(path)
        
        if res:
            img_out = draw_boxes(path, res)
            out_name = "res_" + os.path.basename(path)
            cv2.imwrite(out_name, img_out)
            print(f"took {time.time() - t0:.2f}s, saved to {out_name}")
        else:
            print("prediction failed")
