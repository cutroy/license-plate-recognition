import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO

# Paths
ROOT_DIR = "/kaggle/working/data"
TRAIN_DIR = "/kaggle/input/car-object-detection/data/training_images"
CSV_PATH = "/kaggle/input/car-object-detection/data/train_solution_bounding_boxes (1).csv"
TEST_DIR = "/kaggle/input/car-object-detection/data/testing_images"

# Prepare directories
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(ROOT_DIR, sub), exist_ok=True)

# Load annotations
import pandas as pd
annotations = pd.read_csv(CSV_PATH)

def to_yolo(row, w, h):
    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {xc} {yc} {bw} {bh}"

# Split and prepare YOLO-format data
for img_name in annotations['image'].unique():
    img_path = os.path.join(TRAIN_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    subset = 'train' if np.random.rand() < 0.8 else 'val'
    # Copy image
    dst_img = os.path.join(ROOT_DIR, f"images/{subset}/{img_name}")
    cv2.imwrite(dst_img, img)
    # Create label file
    lbl_path = os.path.join(ROOT_DIR, f"labels/{subset}/{os.path.splitext(img_name)[0]}.txt")
    with open(lbl_path, 'w') as f:
        for _, row in annotations[annotations['image'] == img_name].iterrows():
            f.write(to_yolo(row, w, h) + "\n")

# Create config
config = {
    'path': ROOT_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': ['car']
}
with open('car_detection.yaml', 'w') as f:
    import yaml
    yaml.dump(config, f)

# Disable W&B
os.environ['WANDB_MODE'] = 'disabled'

# Train YOLOv8
model = YOLO('yolov8n.pt')
model.train(
    data='car_detection.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    name='car_detection',
    save=True,
    save_period=1,
    save_weights_only=True
)

# Save best weights
model.best(save_dir='runs/train/car_detection')[0].save('best_yolov8.pt')
print("Weights saved as best_yolov8.pt")
