import cv2
from ultralytics import YOLO
from .base_detector import BaseDetector

class ObjectDetector(BaseDetector):
    def __init__(self, model_path="yolov8n.pt", target_classes=None, conf_threshold=0.5):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path (str): Path to YOLO model file (or name for auto-download).
            target_classes (list): List of class IDs to filter for (e.g., [0] for person).
            conf_threshold (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold

    def detect(self, frame, track=True):
        """
        Run object detection/tracking on the frame.
        
        Args:
            frame: Input image
            track (bool): Whether to use tracking (assign IDs)
            
        Returns:
            list: List of detections with IDs if tracking enabled
        """
        if track:
            results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_threshold, tracker="bytetrack.yaml")[0]
        else:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
            
        detections = []

        if results.boxes: 
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter by target classes if specified
                if self.target_classes is not None and class_id not in self.target_classes:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detection = {
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf
                }
                
                # Add Track ID if available
                if box.id is not None:
                    detection["track_id"] = int(box.id[0])
                    
                detections.append(detection)
            
        return detections
