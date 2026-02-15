from .base_detector import BaseDetector

class ActivityDetector(BaseDetector):
    def __init__(self, history_manager):
        """
        Initialize Activity Detector.
        
        Args:
            history_manager: Instance of TrackHistory
        """
        self.history = history_manager

    def detect(self, detections):
        """
        Analyze current detections and history for specific activities.
        Note: This overrides base detect signature slightly as it takes detections, not frame.
        """
        # Update history with current detections
        self.history.update(detections)
        
        # Analyze
        activities = self.history.analyze()
        
        # Map back to detections
        # We attach activity labels to the detection objects or return new alerts
        results = []
        for tid, acts in activities.items():
            for act in acts:
                results.append({
                    "class_id": act, # 'loitering', 'running'
                    "track_id": tid,
                    "conf": 1.0
                })
        return results
