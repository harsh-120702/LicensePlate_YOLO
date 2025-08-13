import cv2
import numpy as np
from ultralytics import YOLO
import os

class LicensePlateDemo:
    def __init__(self, model_path=None):
        """Initialize the demo with a pre-trained YOLOv8 model."""
        # Load a pre-trained model (for demo purposes, we'll use a small one)
        self.model = YOLO('yolov8n.pt')
        
        # If a custom model path is provided, load it
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
    
    def detect_plates(self, image_path, conf=0.5):
        """
        Detect license plates in an image and display the results.
        
        Args:
            image_path (str): Path to the input image
            conf (float): Confidence threshold (0-1)
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Perform detection
        results = self.model(img, conf=conf)
        
        # Process results
        for result in results:
            boxes = result.boxes
            print(f"Detected {len(boxes)} license plate(s)")
            
            # Draw bounding boxes
            annotated_img = result.plot()
            
            # Display the result
            cv2.imshow('License Plate Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save the result
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, annotated_img)
            print(f"Result saved to {output_path}")

def main():
    # Initialize the demo
    demo = LicensePlateDemo()
    
    # Path to the sample images
    sample_images = [
        "datasets/license_plates/images/train/sample_0.jpg",
        "datasets/license_plates/images/train/sample_1.jpg",
        "datasets/license_plates/images/val/sample_2.jpg"
    ]
    
    # Process each sample image
    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"\nProcessing {img_path}...")
            demo.detect_plates(img_path)
        else:
            print(f"Warning: {img_path} not found")

if __name__ == "__main__":
    main()
