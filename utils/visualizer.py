import cv2

def draw_detections(frame, detections, class_names):
    """
    Draw bounding boxes and labels on frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_id = det.get('class_id')
        conf = det.get('conf', 0)
        track_id = det.get('track_id')
        
        # Determine label and color
        if cls_id == "fire":
            label = f"FIRE {conf:.2f}"
            color = (0, 0, 255) # Red
        elif cls_id == "loitering":
             # Activity detection won't be in main loop detections usually, but handled separately
             continue 
        elif cls_id == "running":
             continue
        else:
            name = class_names.get(cls_id, f"ID {cls_id}")
            label = f"{name} {conf:.2f}"
            color = (0, 255, 0) # Green
            
            if track_id is not None:
                label += f" ID:{track_id}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def draw_activities(frame, activities):
    """
    Draw activity alerts on frame (e.g. "LOITERING DETECTED")
    activities: list of {'class_id': 'loitering', 'track_id': 1}
    """
    y_offset = 30
    for act in activities:
        atype = act['class_id']
        tid = act['track_id']
        text = f"WARNING: {atype.upper()} - ID {tid}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
    return frame
