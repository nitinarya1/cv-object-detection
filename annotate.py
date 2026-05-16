import cv2
import os
import json

boxes = []
drawing = False
ix, iy = -1, -1
curr_cls = 0

classes = ["person", "car", "bike", "motorcycle", "truck", "bus", "cat", "dog", "chair", "bottle"]
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (128, 0, 128), (255, 165, 0),
          (0, 128, 128), (128, 128, 0)]
curr_box = None

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, curr_box, boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            curr_box = [ix, iy, x, y]
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)
        
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            boxes.append({
                "class_id": curr_cls,
                "label": classes[curr_cls],
                "bbox": [x1, y1, x2, y2]
            })
            print(f"Added {classes[curr_cls]} at {x1},{y1},{x2},{y2}")
        curr_box = None

def normalize_box(box, w, h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

def save_data(img_path, shape, out_dir):
    h, w = shape[:2]
    base = os.path.basename(img_path).split('.')[0]
    
    # yolo format
    lines = []
    for b in boxes:
        cx, cy, bw, bh = normalize_box(b["bbox"], w, h)
        lines.append(f"{b['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
    txt_path = os.path.join(out_dir, f"{base}.txt")
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
        
    # json dump just in case
    j_path = os.path.join(out_dir, f"{base}.json")
    with open(j_path, 'w') as f:
        json.dump({"img": img_path, "boxes": boxes}, f)
        
    print(f"Saved {len(boxes)} boxes for {base}")

def start_annotating(img_folder, out_folder="labels"):
    global boxes, curr_cls, curr_box
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    valid = [".jpg", ".png", ".jpeg"]
    imgs = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if any(f.endswith(e) for e in valid)]
    
    if len(imgs) == 0:
        print("No images found!")
        return
        
    cv2.namedWindow('annotate')
    cv2.setMouseCallback('annotate', draw_rect)
    
    print(f"Found {len(imgs)} images. c: change class, s: save/next, q: quit")
    
    for path in imgs:
        boxes = []
        img = cv2.imread(path)
        if img is None: continue
        
        while True:
            temp = img.copy()
            
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                col = colors[b["class_id"]]
                cv2.rectangle(temp, (x1, y1), (x2, y2), col, 2)
                cv2.putText(temp, b["label"], (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1, col, 1)
                
            if curr_box:
                cv2.rectangle(temp, (curr_box[0], curr_box[1]), (curr_box[2], curr_box[3]), colors[curr_cls], 2)
                
            cv2.putText(temp, f"Class: {classes[curr_cls]} ({curr_cls})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow('annotate', temp)
            k = cv2.waitKey(20) & 0xFF
            
            if k == ord('c'):
                curr_cls = (curr_cls + 1) % len(classes)
            elif k == ord('s'):
                save_data(path, img.shape, out_folder)
                break
            elif k == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python annotate.py <img_dir>")
    else:
        start_annotating(sys.argv[1])
