import time
import math
from collections import deque

class TrackHistory:
    def __init__(self, max_history=30):
        """
        Manages history of tracked objects.
        
        Args:
            max_history (int): Number of frames to keep history for velocity calc.
        """
        self.history = {} # {track_id: deque([(x, y, time), ...])}
        self.max_history = max_history
        self.stationary_threshold = 20 # Pixels (movement radius for loitering)
        self.loiter_time_threshold = 5.0 # Seconds
        self.run_speed_threshold = 15.0 # Pixels per frame (approx)

    def update(self, detections):
        """
        Update history with new detections.
        """
        current_time = time.time()
        active_ids = set()

        for det in detections:
            if "track_id" not in det:
                continue
                
            tid = det["track_id"]
            active_ids.add(tid)
            bbox = det["bbox"]
            centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.max_history)
            
            self.history[tid].append((centroid, current_time))

        # Clean up old IDs
        # (Optional: remove IDs not seen for a while)
        for tid in list(self.history.keys()):
            if tid not in active_ids and (current_time - self.history[tid][-1][1] > 5.0):
                del self.history[tid]

    def analyze(self, frame_fps=30):
        """
        Analyze history for activities.
        
        Returns:
            dict: {track_id: ['loitering', 'running']}
        """
        activity_results = {}

        for tid, points in self.history.items():
            if len(points) < 10:
                continue
            
            start_pos, start_time = points[0]
            curr_pos, curr_time = points[-1]
            
            # 1. Loitering Detection
            # If time difference is large but displacement is small
            duration = curr_time - start_time
            displacement = math.sqrt((curr_pos[0] - start_pos[0])**2 + (curr_pos[1] - start_pos[1])**2)
            
            if duration > self.loiter_time_threshold and displacement < self.stationary_threshold:
               if tid not in activity_results: activity_results[tid] = []
               activity_results[tid].append("loitering")

            # 2. Running Detection
            # Check instantaneous velocity over last few frames
            if len(points) >= 5:
                prev_pos, prev_time = points[-5]
                # dist pixels / time diff
                # speed = pixels per second
                dist = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                # To be robust, just key off pixels per frame (assuming constant fps) or pixels/sec
                # pixels/sec
                speed = dist / (curr_time - prev_time + 1e-5) 
                
                # Heuristic: > 200 pixels/sec is "fast" for a webcam
                if speed > 200: 
                   if tid not in activity_results: activity_results[tid] = []
                   activity_results[tid].append("running")

        return activity_results
