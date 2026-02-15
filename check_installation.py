import sys
import os

print("Checking environment...")

try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
except ImportError:
    print("ERROR: OpenCV not installed. Run 'pip install opencv-python'")

try:
    import ultralytics
    from ultralytics import YOLO
    print(f"Ultralytics Version: {ultralytics.__version__}")
except ImportError:
    print("ERROR: Ultralytics not installed. Run 'pip install ultralytics'")

try:
    import numpy
    print(f"Numpy Version: {numpy.__version__}")
except ImportError:
    print("ERROR: Numpy not installed. Run 'pip install numpy'")

print("\nChecking project modules...")
try:
    from detector.object_detector import ObjectDetector
    from detector.fire_detector import FireDetector
    from alerts.alert_manager import AlertManager
    print("All project modules imported successfully.")
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}")
    sys.exit(1)

print("\nEnvironment check passed! You can run 'python main.py'.")
