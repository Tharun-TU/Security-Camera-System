import csv
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_file="events.log"):
        self.log_file = log_file
        
        # Create file with header if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event Type", "Details"])

    def log(self, event_type, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, details])
