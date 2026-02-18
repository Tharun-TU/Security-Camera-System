import cv2
import config
from detector.object_detector import ObjectDetector
from detector.fire_detector import FireDetector
from detector.activity_detector import ActivityDetector
from tracker.track_history import TrackHistory
from alerts.alert_manager import AlertManager
from logger.event_logger import EventLogger
from utils.visualizer import draw_detections, draw_activities

def main():
    print("[INFO] Starting Security Camera System...")
    
    # 1. Initialize Components
    obj_detector = ObjectDetector(
        model_path=config.YOLO_MODEL_NAME,
        target_classes=config.TARGET_CLASSES,
        conf_threshold=config.CONFIDENCE_THRESHOLD
    )
    fire_detector = FireDetector(min_area=config.FIRE_MIN_AREA)
    
    history_manager = TrackHistory()
    act_detector = ActivityDetector(history_manager)
    
    alert_manager = AlertManager(snapshots_dir=config.SNAPSHOTS_DIR, cooldown=config.ALERT_COOLDOWN_SECONDS)
    event_logger = EventLogger()

    # 2. Open Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source {config.CAMERA_INDEX}")
        return

    print("[INFO] System Ready. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended.")
            break

        # Resize for performance (optional)
        # frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # --- DETECTION ---
        # 1. Object Detection + Tracking
        obj_detections = obj_detector.detect(frame, track=True)
        
        # 2. Fire Detection
        fire_detections = fire_detector.detect(frame)
        
        # Filter Fire Detections (Remove false positives inside people)
        # We pass only PERSON detections (class_id 0) for filtering
        person_detections = [d for d in obj_detections if d['class_id'] == 0]
        fire_detections = fire_detector.filter_detections(fire_detections, person_detections)
        
        # 3. Activity Analysis
        # Update history with object detections that have IDs
        tracked_objects = [d for d in obj_detections if "track_id" in d]
        activity_alerts = act_detector.detect(tracked_objects) # Returns list of activity dicts

        # --- ALERTING ---
        all_detections = obj_detections + fire_detections
        
        # Fire Alerts
        if fire_detections:
            alert_manager.trigger_alert("FIRE", frame, details="Fire detected via color analysis")
            event_logger.log("FIRE", "Fire detected")

        # Weapon Alerts (Knife)
        for det in obj_detections:
            cid = det['class_id']
            if cid == 43: # Knife
                label = config.CLASS_NAMES.get(cid, "Weapon")
                alert_manager.trigger_alert("WEAPON", frame, details=f"{label} detected")
                event_logger.log("WEAPON", f"{label} detected")

        # Activity Alerts
        for act in activity_alerts:
            atype = act['class_id']
            tid = act['track_id']
            alert_manager.trigger_alert(atype.upper(), frame, details=f"Track ID {tid}")
            event_logger.log(atype.upper(), f"Track ID {tid}")

        # --- VISUALIZATION ---
        draw_detections(frame, all_detections, config.CLASS_NAMES)
        draw_activities(frame, activity_alerts)

        cv2.imshow("Smart Security Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System Stopped.")

if __name__ == "__main__":
    main()
