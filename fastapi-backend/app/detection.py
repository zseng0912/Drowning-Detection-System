"""
Detection and Enhancement Functions

This module contains all detection, tracking, and enhancement functions
for the drowning detection system.
"""

import time
import cv2
import numpy as np
import torch
import pygame
import tensorflow as tf
from utils.data_utils import preprocess, deprocess


# Global variables for tracking
consecutive_drowning_count = 0
alarm_sound_path = "sound/alarm.mp3"
realtime_metrics = {
    "processing_speed_ms": 0.0,
    "real_time_speed_fps": 0.0,
    "latency_per_frame_ms": 0.0,
    "last_updated": 0.0
}

# Color mapping for detection classes
CLASS_COLORS = {
    "normal": (0, 255, 0),      # Green for normal behavior
    "drowning": (0, 0, 255)    # Red for drowning detection
}


def play_alarm_sound():
    """Play the drowning alarm sound using pygame."""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(alarm_sound_path)
        pygame.mixer.music.play()
        print("🚨 DROWNING ALARM TRIGGERED! Playing alarm sound...")
    except Exception as e:
        print(f"❌ Error playing alarm sound: {e}")


def _color_for_class(name: str):
    """Get BGR color tuple for a given class name."""
    name = (name or "").lower()
    if name == "normal":
        return (0, 255, 0)      # Green
    if name == "drowning":
        return (0, 0, 255)      # Red
    return (255, 255, 0)        # Yellow (default)


def draw_fancy_box(frame, x1, y1, x2, y2, color, label=None):
    """Draw a styled bounding box with optional label on the frame."""
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        th = th + baseline
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1, cv2.LINE_AA)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def check_drowning_and_trigger_alarm(detections, frame):
    """Check for drowning detections and trigger alarm if needed."""
    global consecutive_drowning_count
    
    # Check if any drowning detection exists in current frame
    drowning_detected = False
    
    # Handle different detection formats
    if detections and len(detections) > 0:
        if isinstance(detections[0], dict):  # From track_with_deepsort
            drowning_detected = any(detection["class"].lower() == "drowning" for detection in detections)
        elif isinstance(detections[0], tuple):  # From run_detection
            drowning_detected = any(detection[4].lower() == "drowning" for detection in detections)
    
    if drowning_detected:
        consecutive_drowning_count += 1
        
        # Trigger alarm every 5 consecutive detections (at 5, 10, 15, etc.)
        if consecutive_drowning_count % 5 == 0:
            play_alarm_sound()
            print(f"🚨 ALARM TRIGGERED! Drowning detected {consecutive_drowning_count} times consecutively!")
    else:
        # Reset counter if no drowning detected
        consecutive_drowning_count = 0
    
    return frame


def run_detection(model, frame, conf_thresh: float = 0.50):
    """Run YOLO detection on a single frame with confidence threshold."""
    with torch.no_grad():
        results = model(frame, conf=conf_thresh, verbose=False)
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                conf = float(box.conf[0].cpu())
                cls_name = result.names[int(box.cls[0].cpu())]
                detections.append((x1, y1, x2, y2, cls_name, conf))
    return detections


def draw_detections(frame, detections):
    """Draw detection bounding boxes on frame."""
    for x1, y1, x2, y2, cls_name, conf in detections:
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        label = f"{cls_name} {conf:.2f}" if conf is not None else cls_name
        frame = draw_fancy_box(frame, x1, y1, x2, y2, color, label)
    return frame


def track_with_deepsort(model, frame, deepsort, conf_thresh: float = 0.50, min_area: float = 500.0, is_video: bool = True):
    """Perform object detection with YOLO and tracking with DeepSORT."""
    # YOLO inference
    with torch.no_grad():
        res = model(frame, conf=conf_thresh, iou=0.4, verbose=False)[0]

    xywhs, confs, class_ids, raw_boxes = [], [], [], []
    if res.boxes is not None:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
            if w * h < min_area:
                continue
            xc, yc = x1 + w / 2, y1 + h / 2
            xywhs.append([xc, yc, w, h])
            confs.append(conf)
            class_ids.append(cid)
            raw_boxes.append((x1, y1, x2, y2, cid, conf))

    tracked = []

    if is_video:  
        # DeepSORT update
        outputs = None
        if len(xywhs) > 0:
            xywhs = torch.tensor(xywhs)
            confs = torch.tensor(confs)
            outputs = deepsort.update(xywhs, confs, class_ids, frame)

        # Build results
        if outputs is not None and len(outputs) > 0:
            for out in outputs:
                x1, y1, x2, y2, tid, cid = out
                cname = res.names.get(int(cid), str(cid))
                conf = None
                # find matching confidence (optional)
                for b in raw_boxes:
                    if abs(b[0]-x1) < 5 and abs(b[1]-y1) < 5:
                        conf = b[-1]
                        break

                color = _color_for_class(cname)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{tid} {cname}" + (f" {conf:.2f}" if conf else ""),
                            (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                tracked.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "track_id": int(tid),
                    "class": cname,
                    "class_id": int(cid),
                    "confidence": conf if conf else 0.0
                })

    else:  
        # Single image mode: restart IDs for each image
        local_id_counter = 0
        for (x1, y1, x2, y2, cid, conf) in raw_boxes:
            local_id_counter += 1
            cname = res.names.get(cid, str(cid))
            color = _color_for_class(cname)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{local_id_counter} {cname} {conf:.2f}",
                        (int(x1), max(0, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            tracked.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "track_id": local_id_counter,
                "class": cname,
                "class_id": cid,
                "confidence": conf
            })

    # Active tracks count overlay
    active_count = len(tracked)
    count_text = f"Active Tracks: {active_count}"

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]

    # Box padding
    pad_x, pad_y = 12, 8
    x, y = 10, 40
    box_x1, box_y1 = x - pad_x, y - text_size[1] - pad_y
    box_x2, box_y2 = x + text_size[0] + pad_x, y + pad_y

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw border
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)

    # Put text on top
    cv2.putText(frame, count_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Drowning alert check
    frame = check_drowning_and_trigger_alarm(tracked, frame)

    return frame, tracked


def detection_with_enhance(frame, model, gan_model, enhancement=False, conf_thresh: float = 0.50):
    """Run detection with optional enhancement and confidence threshold."""
    # Step 1: Enhance only if requested + underwater + GAN available
    if enhancement and gan_model is not None:
        enhanced_frame, success = enhance_frame(frame, gan_model, side_by_side=False)
        if success:
            frame = enhanced_frame

    # Step 2: Run detection with confidence threshold
    detections = run_detection(model, frame, conf_thresh)

    # Step 3: Draw results
    processed_frame = draw_detections(frame, detections)

    # Step 4: Check drowning + trigger alarm
    processed_frame = check_drowning_and_trigger_alarm(detections, processed_frame)

    return processed_frame, detections


def enhance_frame(frame, gan_model, side_by_side=False):
    """Enhance a single frame using GAN model for underwater image enhancement."""
    global realtime_metrics
    
    if gan_model is None:
        return frame, False
    
    try:
        start_time = time.time()
        
        # Convert BGR->RGB and resize to model's expected input (256x256)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)

        # Normalize [-1,1], predict, deprocess back to [0,255] uint8 RGB
        im = preprocess(rgb_small)
        im = np.expand_dims(im, axis=0)
        
        # Time the model prediction
        prediction_start = time.time()
        gen = gan_model.predict(im, verbose=0)
        prediction_end = time.time()
        
        gen_rgb_small = deprocess(gen)[0]  # (256,256,3) uint8

        # Upscale enhanced back to original frame size
        gen_rgb = cv2.resize(gen_rgb_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        gen_bgr = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR)
        
        # Calculate metrics
        end_time = time.time()
        total_processing_time = end_time - start_time
        prediction_time = prediction_end - prediction_start
        
        # Update global metrics
        realtime_metrics["processing_speed_ms"] = round(prediction_time * 1000, 2)
        realtime_metrics["real_time_speed_fps"] = round(1.0 / total_processing_time if total_processing_time > 0 else 0, 2)
        realtime_metrics["latency_per_frame_ms"] = round(total_processing_time * 1000, 2)
        realtime_metrics["last_updated"] = time.time()

        if side_by_side:
            out = np.hstack((frame, gen_bgr))
            return out, True
        else:
            return gen_bgr, True

    except Exception as e:
        print(f"❌ Error enhancing frame: {e}")
        return frame, False


def enhance_video_frames_batch(frames, gan_model):
    """Process frames one by one to avoid GPU memory issues."""
    if gan_model is None or not frames:
        return [(frame, False) for frame in frames], False
    
    results = []
    for frame in frames:
        try:
            enhanced_frame, success = enhance_frame(frame, gan_model, side_by_side=False)
            results.append((enhanced_frame, success))
        except Exception as e:
            print(f"❌ Error enhancing frame: {e}")
            results.append((frame, False))
    
    return results, True


def get_realtime_metrics():
    """Get current real-time processing metrics."""
    global realtime_metrics
    
    # Check if metrics are recent (within last 5 seconds)
    current_time = time.time()
    metrics_age = current_time - realtime_metrics["last_updated"]
    
    if metrics_age > 5.0:  # If metrics are older than 5 seconds, return default values
        return {
            "processing_speed_ms": 40.0,
            "real_time_speed_fps": 25.0,
            "latency_per_frame_ms": 40.0,
            "metrics_available": False,
            "message": "No recent metrics available"
        }
    
    return {
        "processing_speed_ms": realtime_metrics["processing_speed_ms"],
        "real_time_speed_fps": realtime_metrics["real_time_speed_fps"],
        "latency_per_frame_ms": realtime_metrics["latency_per_frame_ms"],
        "metrics_available": True,
        "last_updated": realtime_metrics["last_updated"]
    }
