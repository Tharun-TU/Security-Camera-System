import cv2
import numpy as np
from .base_detector import BaseDetector

class FireDetector(BaseDetector):
    def __init__(self, min_area=500):
        """
        Initialize Fire Detector (Color-based).
        
        Args:
            min_area (int): Minimum area of contour to consider as fire.
        """
        self.min_area = min_area
        # HSV range for fire colors (adjust as needed for environment)
        # Typically fire is bright orange/red/yellow
        self.lower_bound = np.array([0, 100, 200]) # Lower HSV
        self.upper_bound = np.array([35, 255, 255]) # Upper HSV

    def detect(self, frame):
        """
        Detect fire-like regions based on color.
        
        Returns:
            list: List of detections [{'class_id': 'fire', 'bbox': [x1, y1, x2, y2], 'conf': 1.0}]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2) # Dilate more to merge flames

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "class_id": "fire", # String ID for special custom classes
                    "bbox": [x, y, x + w, y + h],
                    "conf": 1.0 # Heuristic, so we give it fixed confidence
                })
        
        return detections
