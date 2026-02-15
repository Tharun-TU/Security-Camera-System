import time
import cv2
import os
import threading
from datetime import datetime

class AlertManager:
    def __init__(self, snapshots_dir="snapshots", cooldown=5):
        """
        Manages alerts to avoid spamming.
        
        Args:
            snapshots_dir (str): Directory to save snapshots.
            cooldown (int): Seconds to wait before repeating an alert for the same event type.
        """
        self.snapshots_dir = snapshots_dir
        self.cooldown = cooldown
        self.last_alert_time = {} # {event_type: timestamp}
        
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)

    def trigger_alert(self, event_type, frame, details=""):
        """
        Trigger an alert if cooldown has passed.
        
        Args:
            event_type (str): Type of event (e.g., 'Fire', 'Person', 'Weapon').
            frame (numpy.ndarray): Current video frame.
            details (str): Extra info.
        """
        current_time = time.time()
        
        # Check cooldown
        if event_type in self.last_alert_time:
            if current_time - self.last_alert_time[event_type] < self.cooldown:
                return # Skip alert
        
        self.last_alert_time[event_type] = current_time
        
        # 1. Console Alert
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ALERT] {timestamp} - Detected: {event_type} - {details}")
        
        # 2. Save Snapshot (in background to not block main thread)
        thread = threading.Thread(target=self._save_snapshot, args=(frame, event_type))
        thread.start()

    def _save_snapshot(self, frame, event_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.snapshots_dir}/{event_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Snapshot saved: {filename}")
