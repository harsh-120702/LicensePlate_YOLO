import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path

# Create sample directories
base_dir = Path("datasets/license_plates")
img_train_dir = base_dir / "images/train"
img_val_dir = base_dir / "images/val"
label_train_dir = base_dir / "labels/train"
label_val_dir = base_dir / "labels/val"

# Sample image URLs (license plates for demonstration)
sample_images = [
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
]

def download_image(url, save_path):
    """Download an image from URL and save it to the specified path."""
    try:
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        if img is not None:
            cv2.imwrite(str(save_path), img)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def create_yolo_label(img_path, label_path, bbox):
    """
    Create a YOLO format label file.
    bbox format: [x_center, y_center, width, height] in normalized coordinates (0-1)
    """
    with open(label_path, 'w') as f:
        # class_id x_center y_center width height
        f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def main():
    # Sample bounding boxes (normalized: [x_center, y_center, width, height])
    sample_bboxes = [
        [0.5, 0.1, 0.2, 0.1],  # For first image
        [0.5, 0.8, 0.3, 0.15], # For second image
        [0.3, 0.2, 0.15, 0.1]  # For third image
    ]
    
    print("Downloading sample images...")
    for i, (img_url, bbox) in enumerate(zip(sample_images, sample_bboxes)):
        # For demo, we'll put 2 in train and 1 in val
        if i < 2:
            img_dir = img_train_dir
            label_dir = label_train_dir
        else:
            img_dir = img_val_dir
            label_dir = label_val_dir
            
        img_path = img_dir / f"sample_{i}.jpg"
        label_path = label_dir / f"sample_{i}.txt"
        
        if download_image(img_url, img_path):
            create_yolo_label(img_path, label_path, bbox)
            print(f"Downloaded and saved {img_path}")
    
    print("\nSample dataset created successfully!")
    print(f"Training images: {len(list(img_train_dir.glob('*.jpg')))}")
    print(f"Validation images: {len(list(img_val_dir.glob('*.jpg')))}")
    print("\nYou can now run the training script with: python train.py")

if __name__ == "__main__":
    main()
