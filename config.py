import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")

# Camera
CAMERA_INDEX = 0  # 0 for default webcam, or path to video file
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Detection Models (Standard YOLOv8n)
# We will use the standard model's classes:
# 0: person
# 43: knife
# 76: scissors (proxy for weapon)
YOLO_MODEL_NAME = "yolov8n.pt"  # Will automatically download if not present

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Alert Configuration
ALERT_COOLDOWN_SECONDS = 5  # Seconds between alerts for the same event type
ENABLE_SOUND = False

# Classes to Monitor (COCO indices)
# Refer to https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
TARGET_CLASSES = [0, 43] 
CLASS_NAMES = {
    0: "Person",
    43: "Knife"
}

# Fire Detection (Color Based)
FIRE_HUE_LOWER = 0
FIRE_HUE_UPPER = 20  # Orange/Red range
FIRE_SAT_LOWER = 100
FIRE_VAL_LOWER = 100
FIRE_MIN_AREA = 500  # Minimum pixel area to consider as fire
