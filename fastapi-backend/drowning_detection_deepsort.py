import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from tqdm import tqdm

# ---------------- CONFIG ----------------
VIDEO_PATH = "123.mp4"       # <<-- your video file
MODEL_PATH = "model/detection_model/abovewaterModel.pt"            # replace with abovewaterModel.pt if needed
OUTPUT_PATH = "output_tracked5.mp4"
CONF_THRESH = 0.40

# ---------------- YOLO ----------------
model = YOLO(MODEL_PATH)  # ultralytics YOLOv8 model
names = model.names

# ---------------- DeepSORT ----------------
cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, 
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, 
                    n_init=cfg.DEEPSORT.N_INIT, 
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=torch.cuda.is_available())

# ---------------- Video Setup ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# ---------------- Helper Functions ----------------
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

def compute_color_for_labels(label):
    """Set fixed color for each class"""
    return tuple(int((p * (label ** 2 - label + 1)) % 255) for p in palette)

def draw_boxes(img, bbox, identities, object_id):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(j) for j in box]
        id = int(identities[i]) if identities is not None else 0
        cls_id = int(object_id[i])
        cls_name = names[cls_id]

        # --- Fixed colors for specific classes ---
        if cls_name.lower() == "normal":
            color = (0, 255, 0)   # Green for normal
        elif cls_name.lower() == "drowning":
            color = (0, 0, 255)   # Red for drowning
        else:
            color = (255, 255, 0) # Yellow for any other class

        label = f"{cls_name} ID:{id}"

        # Draw rectangle and label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
    return img

# ---------------- Main Loop ----------------
pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESH, iou=0.4, verbose=False)[0]

    xywhs = []
    confs = []
    oids = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        w = x2 - x1
        h = y2 - y1
        x_c, y_c = x1 + w / 2, y1 + h / 2

        # --- FILTER small boxes (to remove splashes/reflections) ---
        if w * h < 500:   # adjust this threshold depending on your video resolution
            continue

        xywhs.append([x_c, y_c, w, h])
        confs.append(conf)
        oids.append(cls_id)


    if len(xywhs) > 0:
        xywhs = torch.Tensor(xywhs)
        confs = torch.Tensor(confs)
        outputs = deepsort.update(xywhs, confs, oids, frame)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, identities, object_id)

    out.write(frame)
    pbar.update(1)

cap.release()
out.release()
pbar.close()
print("✅ Done. Saved:", OUTPUT_PATH)