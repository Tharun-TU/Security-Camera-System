# Smart Security Camera Alert System

A real-time AI surveillance system that detects **Fire**, **Weapons** (Knives/Scissors), **Persons**, and **Suspicious Activity** (Loitering, Running) using a webcam.

## Features
- **Real-time Detection**: Uses YOLOv8 for objects and Color Analysis for fire.
- **Activity Analysis**: Detects Loitering and Running based on movement history.
- **Alerts**: Console output and Snapshot saving (`snapshots/`).
- **Logging**: Events logged to `events.log`.

## Setup
1.  **Install Python 3.8+**
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the System**:
    ```bash
    python main.py
    ```

## Configuration
Edit `config.py` to adjust:
- `CAMERA_INDEX`: 0 for webcam, or path to video file.
- `CONFIDENCE_THRESHOLD`: Sensitivity of detection.
- `ALERT_COOLDOWN_SECONDS`: prevent spamming.

## Modules
- `detector/`: Logic for YOLO and Fire detection.
- `tracker/`: Activity analysis and object tracking.
- `alerts/`: Snapshot saving and alert management.
- `logger/`: Event logging.

## Troubleshooting
- **No Camera**: Check `CAMERA_INDEX` in `config.py`.
- **Missing Model**: `yolov8n.pt` will automatically download on first run.
