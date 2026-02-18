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

    def filter_detections(self, fire_detections, person_detections):
        """
        Filter out fire detections that are likely false positives (e.g., orange shirt).
        Logic: If fire bbox is inside a person bbox, ignore it.
        """
        filtered = []
        for fire in fire_detections:
            fx1, fy1, fx2, fy2 = fire['bbox']
            fire_area = (fx2 - fx1) * (fy2 - fy1)
            is_false_positive = False
            
            for person in person_detections:
                if person['class_id'] != 0: # 0 is person
                    continue
                
                px1, py1, px2, py2 = person['bbox']
                
                # Check for overlap
                ox1 = max(fx1, px1)
                oy1 = max(fy1, py1)
                ox2 = min(fx2, px2)
                oy2 = min(fy2, py2)
                
                if ox1 < ox2 and oy1 < oy2:
                    overlap_area = (ox2 - ox1) * (oy2 - oy1)
                    # If > 50% of the "fire" is inside a "person", it's likely a shirt
                    if overlap_area > 0.5 * fire_area:
                        is_false_positive = True
                        break
            
            if not is_false_positive:
                filtered.append(fire)
                
        return filtered
