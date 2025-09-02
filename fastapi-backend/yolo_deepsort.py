import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from tqdm import tqdm   # progress bar

# --- CONFIG ---
VIDEO_PATH = "testing123.mp4"
MODEL_PATH = "model/detection_model/abovewaterModel.pt"
OUTPUT_PATH = "output_tracked8.mp4"
CONF_THRESH = 0.25
FILTER_PERSON_ONLY = False

# --- Load YOLO ---
model = YOLO(MODEL_PATH)
names = model.names

# --- DeepSORT ---
use_gpu = torch.cuda.is_available()
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3,
                   embedder="mobilenet", half=use_gpu,
                   bgr=True, embedder_gpu=use_gpu)

# --- Video setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# --- Counting ---
unique_ids = set()

def get_color(idx):
    if not isinstance(idx, int):
        idx = abs(hash(idx)) % (2**32)
    np.random.seed(idx + 12345)
    return tuple(int(x) for x in np.random.randint(60, 255, size=3))

# --- Progress bar ---
pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # --- Run YOLOv8 ---
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        if FILTER_PERSON_ONLY and cls_name != "person":
            continue

        detections.append(([x1, y1, x2, y2], conf, cls_name))

    # --- DeepSORT update ---
    tracks = tracker.update_tracks(detections, frame=frame)

    current_ids = set()

    # --- Draw ---
    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        unique_ids.add(tid)
        current_ids.add(tid)

        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, r, b])

        color = get_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tid}"
        if track.get_det_class():
            label = f"{track.get_det_class()} | {label}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Overlay Counts ---
    total_count = len(unique_ids)
    active_count = len(current_ids)

    cv2.putText(frame, f"Total unique: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, f"Active: {active_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    out.write(frame)

    # --- Update progress ---
    pbar.update(1)

cap.release()
out.release()
pbar.close()
print("✅ Done. Saved:", OUTPUT_PATH)
