import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

class LicensePlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize the License Plate Detector
        :param model_path: Path to pre-trained YOLOv8 model (if available)
        """
        self.model = YOLO('yolov8n.pt')  # Load base YOLOv8 model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        
    def train(self, data_yaml, epochs=50, imgsz=640, batch=16):
        """
        Train the YOLOv8 model on custom dataset
        :param data_yaml: Path to data.yaml file containing dataset configuration
        :param epochs: Number of training epochs
        :param imgsz: Image size for training
        :param batch: Batch size
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='license_plate_detection'
        )
        return results
    
    def detect(self, image_path, conf=0.5, save=True):
        """
        Detect license plates in an image
        :param image_path: Path to input image
        :param conf: Confidence threshold
        :param save: Whether to save the output image
        :return: Detection results
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Perform detection
        results = self.model(img, conf=conf)
        
        # Visualize results
        annotated_img = results[0].plot()
        
        # Save or display results
        if save:
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, annotated_img)
            print(f"Results saved to {output_path}")
        
        return results
    
    def detect_video(self, video_path, output_path='output.mp4', conf=0.5):
        """
        Detect license plates in a video
        :param video_path: Path to input video
        :param output_path: Path to save output video
        :param conf: Confidence threshold
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform detection
            results = self.model(frame, conf=conf)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Write the frame
            out.write(annotated_frame)
        
        # Release resources
        cap.release()
        out.release()
        print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    detector = LicensePlateDetector()
    
    # For training (uncomment and modify paths as needed)
    # detector.train(data_yaml='data.yaml', epochs=50)
    
    # For inference on an image
    # detector.detect('path/to/your/image.jpg')
    
    # For inference on a video
    # detector.detect_video('path/to/your/video.mp4')
