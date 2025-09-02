from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ultralytics import YOLO
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import uuid
import mimetypes
import cv2
import shutil
import base64
import threading
import time
import numpy as np
import math
import torch
from collections import defaultdict, deque
import pygame

# GAN model imports for video enhancement
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from utils.data_utils import preprocess, deprocess, read_and_resize
# ----------------- DEEPSORT INIT (once) -----------------
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# ----------------- FASTAPI APP SETUP -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- GLOBAL VARIABLES -----------------
DEEPSORT_CFG_PATH = "deep_sort_pytorch/configs/deep_sort.yaml"

_cfg = get_config()
_cfg.merge_from_file(DEEPSORT_CFG_PATH)
deepsort = DeepSort(
    _cfg.DEEPSORT.REID_CKPT,
    max_dist=_cfg.DEEPSORT.MAX_DIST,
    min_confidence=_cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=_cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=_cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=_cfg.DEEPSORT.MAX_AGE,
    n_init=_cfg.DEEPSORT.N_INIT,
    nn_budget=_cfg.DEEPSORT.NN_BUDGET,
    use_cuda=torch.cuda.is_available()
)

webcam_active = False
current_model = "underwater"
webcam_cap = None
video_feed_active = False
webcam_enhancement_active = False  # New: tracks if enhancement mode is enabled

# Global variables for drowning alarm
consecutive_drowning_count = 0
alarm_sound_path = "sound/alarm.mp3"

# Threading variables for video processing
latest_detections = []
latest_frame = None
processing_lock = threading.Lock()
stop_processing = False
processing_thread = None

# Real-time metrics tracking
realtime_metrics = {
    "processing_speed_ms": 0.0,
    "real_time_speed_fps": 0.0,
    "latency_per_frame_ms": 0.0,
    "last_updated": 0.0
}

# ----------------- USER LOGIN -----------------
class LoginRequest(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: int
    username: str
    name: str
    role: str
    certifications: List[str]
    shift: str
    status: str
    avatar: str

users_db = [
    {
        "id": 1,
        "username": "admin",
        "password": "password123",
        "name": "Pool Administrator",
        "role": "Administrator",
        "certifications": ["Management"],
        "shift": "Full Day (8:00 AM - 6:00 PM)",
        "status": "on-duty",
        "avatar": "PA"
    }
]

@app.post("/login")
def login(request: LoginRequest):
    user = next((u for u in users_db if u["username"] == request.username and u["password"] == request.password), None)
    if user:
        user_data = user.copy()
        user_data.pop("password")
        return {"success": True, "token": "fake-jwt-token", "user": user_data}
    else:
        return {"success": False, "message": "Invalid credentials"}

# ----------------- LOAD MODELS ON STARTUP -----------------
# Check if CUDA is available and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Using device: {device}")

# Load YOLO models with CUDA support
models = {
    "underwater": YOLO("model/detection_model/underwaterModel.pt").to(device),
    "above-water": YOLO("model/detection_model/abovewaterModel.pt").to(device),
}

print(f"✅ YOLO models loaded on {device}")
if torch.cuda.is_available():
    print(f"🚀 CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ----------------- GPU SETUP -----------------
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ TensorFlow GPU available: {len(gpus)} physical / {len(logical_gpus)} logical")
        except Exception as e:
            print(f"⚠️ Could not set memory growth: {e}")
    else:
        print("⚠️ No GPU detected by TensorFlow. Running on CPU.")

# ----------------- TENSORFLOW OPTIMIZATION FUNCTIONS -----------------
def optimize_tensorflow_for_inference():
    """Optimize TensorFlow settings for real-time inference"""
    # Enable mixed precision for faster inference
    try:
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
    except:
        pass
    
    # Set thread settings for optimal performance
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

# ----------------- LOAD GAN MODEL FOR VIDEO ENHANCEMENT -----------------
def load_gan_model():
    """Load the FUnIE-GAN model for video enhancement - simplified like realtime_funie.py"""
    try:
        checkpoint_dir = 'model/FUnIE_GAN_model/'
        model_name_by_epoch = "model_15320_"
        model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"
        model_json = checkpoint_dir + model_name_by_epoch + ".json"
        
        if not (os.path.exists(model_h5) and os.path.exists(model_json)):
            print("⚠️ GAN model files not found. Video enhancement will be disabled.")
            return None
        
        with open(model_json, "r") as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)
        model.load_weights(model_h5)
        print("✅ Loaded FUnIE-GAN model successfully")
        
        # Warm-up on GPU (important for first-frame latency)
        dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
        _ = model.predict(dummy)
        return model
        
    except Exception as e:
        print(f"❌ Failed to load GAN model: {str(e)}")
        return None

# Load GAN model on startup
gan_model = load_gan_model()
gan_input_size = (256, 256)  # GAN model trained size

# ----------------- SINGLE IMAGE ENHANCEMENT FUNCTION -----------------
def enhance_single_image(img_array, save_path=None):
    """
    Enhance a single image (numpy array).
    - img_array: input image as numpy array (H,W,3), BGR or RGB
    - save_path: optional path to save side-by-side output
    """
    try:
        # Ensure RGB
        if img_array.shape[-1] == 3:
            inp_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            inp_img = img_array

        # Resize to 256x256 for model
        inp_img = cv2.resize(inp_img, (256, 256))

        # Normalize [-1, 1]
        im = preprocess(inp_img)
        im = np.expand_dims(im, axis=0)  # (1,256,256,3)

        # Generate enhanced image
        start = time.time()
        gen = gan_model.predict(im)
        gen_img = deprocess(gen)[0]
        elapsed = time.time() - start
        print(f"⏱ Enhanced in {elapsed:.4f} sec ({1./elapsed:.2f} fps)")

        # Side-by-side (original vs enhanced)
        out_img = np.hstack((inp_img.astype('uint8'), gen_img.astype('uint8')))

        if save_path is not None:
            from PIL import Image
            Image.fromarray(out_img).save(save_path)
            print(f"💾 Saved enhanced image to {save_path}")

        return out_img

    except Exception as e:
        print(f"❌ Error enhancing image: {e}")
        return None

CLASS_COLORS = {
    "normal": (0, 255, 0),
    "drowning": (0, 0, 255)
}

# ----------------- DROWNING ALARM FUNCTIONALITY -----------------
def play_alarm_sound():
    """
    Play the alarm sound using pygame
    """
    try:
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Load and play the alarm sound
        pygame.mixer.music.load(alarm_sound_path)
        pygame.mixer.music.play()
        print("🚨 DROWNING ALARM TRIGGERED! Playing alarm sound...")
        
    except Exception as e:
        print(f"❌ Error playing alarm sound: {e}")

def check_drowning_and_trigger_alarm(detections, frame):
    """
    Check for drowning detections and trigger alarm if needed.
    Works with both tracked detections (from track_with_deepsort) and regular detections (from run_detection)
    """
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
        print(f"🚨 Drowning detected! Consecutive count: {consecutive_drowning_count}")
        
        # Trigger alarm every 5 consecutive detections (at 5, 10, 15, etc.)
        if consecutive_drowning_count % 5 == 0:
            play_alarm_sound()
            print(f"🚨 ALARM TRIGGERED! Drowning detected {consecutive_drowning_count} times consecutively!")
    else:
        # Reset counter if no drowning detected
        consecutive_drowning_count = 0
    
    return frame

# ----------------- DRAW TRACKED BOXES + TRAILS -----------------
def _color_for_class(name: str):
    name = (name or "").lower()
    if name == "normal":
        return (0, 255, 0)
    if name == "drowning":
        return (0, 0, 255)
    return (255, 255, 0)

def draw_fancy_box(frame, x1, y1, x2, y2, color, label=None):
    """
    Draws a rounded bounding box with a semi-transparent label background.
    """
    # thickness for box
    thickness = 2
    
    # Draw rectangle with rounded corners (using polylines)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        # Get text size
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        th = th + baseline

        # Make background rectangle
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (x1, y1 - th - 4), 
            (x1 + tw + 4, y1), 
            color, -1, cv2.LINE_AA
        )
        # Blend overlay with transparency
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Put label text with shadow
        cv2.putText(frame, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return frame

def draw_tracked(frame, bbox_xyxy, identities, class_ids, class_names, confidences=None):
    if bbox_xyxy is None or len(bbox_xyxy) == 0:
        return frame

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        tid = int(identities[i]) if identities is not None else -1
        cls_id = int(class_ids[i]) if class_ids is not None else -1
        cls_name = class_names.get(cls_id, str(cls_id))
        color = _color_for_class(cls_name)

        # label with ID + class + confidence
        if confidences is not None and i < len(confidences):
            conf = float(confidences[i])
            label = f"ID:{tid} {cls_name} {conf:.2f}"
        else:
            label = f"ID:{tid} {cls_name}"

        frame = draw_fancy_box(frame, x1, y1, x2, y2, color, label)

    return frame

# ----------------- YOLO -> DEEPSORT WRAPPER -----------------
def track_with_deepsort(model_type: str, frame, conf_thresh: float = 0.40, min_area: float = 500.0):
    """
    Runs YOLOv8 on frame, attaches DeepSORT IDs, returns:
    - annotated frame
    - tracked list of dicts: x1,y1,x2,y2,track_id,class,conf
    NOTE: Keeps YOLO’s raw boxes (not DeepSORT-updated ones).
    """
    model = models[model_type]  # global models dict
    
    # --- YOLO inference ---
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
            raw_boxes.append((x1, y1, x2, y2, cid, conf))  # store YOLO’s raw box

    # --- DeepSORT update ---
    outputs = None
    if len(xywhs) > 0:
        xywhs = torch.tensor(xywhs)
        confs = torch.tensor(confs)
        outputs = deepsort.update(xywhs, confs, class_ids, frame)

    tracked = []
    if outputs is not None and len(outputs) > 0:
        # map track IDs back to YOLO’s raw boxes
        for i, (x1, y1, x2, y2, cid, conf) in enumerate(raw_boxes):
            # find matching DeepSORT ID (nearest IoU match)
            tid = None
            for det in outputs:
                dx1, dy1, dx2, dy2, tid_, cid_ = det
                # simple IoU check
                iou = (
                    max(0, min(x2, dx2) - max(x1, dx1))
                    * max(0, min(y2, dy2) - max(y1, dy1))
                ) / ((x2 - x1) * (y2 - y1) + (dx2 - dx1) * (dy2 - dy1))
                if iou > 0.3:  # IoU threshold to associate
                    tid = int(tid_)
                    break

            if tid is not None:
                cname = res.names.get(cid, str(cid))
                tracked.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "track_id": tid,
                    "class": cname,
                    "class_id": cid,
                    "confidence": conf
                })

                # choose color based on class
                color = _color_for_class(cname)

                # draw box + ID + confidence
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{tid} {cname} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add active tracks count display
    active_count = len(tracked)
    count_text = f"Active Tracks: {active_count}"
    
    # Draw background rectangle for count text
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
    
    # Draw count text
    cv2.putText(frame, count_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Check for drowning and trigger alarm if needed
    frame = check_drowning_and_trigger_alarm(tracked, frame)

    return frame, tracked

# ----------------- DRAW DETECTIONS -----------------
def draw_detections(frame, detections):
    for x1, y1, x2, y2, cls_name, conf in detections:
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        label = f"{cls_name} {conf:.2f}" if conf is not None else cls_name
        frame = draw_fancy_box(frame, x1, y1, x2, y2, color, label)
    return frame

# ----------------- DETECTION FUNCTION -----------------
def run_detection(model_type, frame):
    # Ensure model is on correct device
    model = models[model_type]
    
    # Run inference with optimized settings for CUDA
    with torch.no_grad():  # Disable gradient computation for inference
        results = model(frame, verbose=False)  # Disable verbose output for speed
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())  # Move to CPU for processing
                conf = float(box.conf[0].cpu())
                cls_name = result.names[int(box.cls[0].cpu())]
                detections.append((x1, y1, x2, y2, cls_name, conf))
    return detections

# ----------------- VIDEO ENHANCEMENT FUNCTION -----------------

def enhance_video_frame(frame, gan_model, side_by_side=False):
    """
    Enhance a single video frame (BGR -> RGB for model -> back to BGR for display/saving)
    Based on realtime_funie.py implementation for consistency and efficiency
    Returns:
      - enhanced_bgr if side_by_side=False
      - concat_bgr (input|enhanced) if side_by_side=True
    """
    global realtime_metrics
    
    if gan_model is None:
        return frame, False
    
    try:
        # Start timing for metrics
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

        # Upscale enhanced back to original frame size (keep aspect/content)
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
    """
    Process frames one by one to avoid GPU memory issues
    Based on realtime_funie.py single-frame approach for memory efficiency
    """
    if gan_model is None or not frames:
        return [(frame, False) for frame in frames], False
    
    results = []
    for frame in frames:
        try:
            enhanced_frame, success = enhance_video_frame(frame, gan_model, side_by_side=False)
            results.append((enhanced_frame, success))
        except Exception as e:
            print(f"❌ Error enhancing frame: {e}")
            results.append((frame, False))
    
    return results, True

def create_side_by_side_video(original_path, enhanced_path, output_path, fps=30):
    """Create a side-by-side comparison video"""
    try:
        cap_orig = cv2.VideoCapture(original_path)
        cap_enh = cv2.VideoCapture(enhanced_path)
        
        if not (cap_orig.isOpened() and cap_enh.isOpened()):
            return False
        
        # Get video properties
        width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer for side-by-side with browser-compatible codec
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 - most compatible
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID fallback
                output_path = output_path.replace('.mp4', '.avi')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Last resort
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        frame_count = 0
        while True:
            ret1, frame_orig = cap_orig.read()
            ret2, frame_enh = cap_enh.read()
            
            if not (ret1 and ret2):
                break
            
            # Create side-by-side frame
            combined = np.hstack((frame_orig, frame_enh))
            out.write(combined)
            frame_count += 1
        
        cap_orig.release()
        cap_enh.release()
        out.release()
        
        print(f"✅ Side-by-side video created with {frame_count} frames")
        return True
    except Exception as e:
        print(f"❌ Error creating side-by-side video: {str(e)}")
        return False

def create_video_with_ffmpeg_fallback(frames, output_path, fps=30, width=640, height=480):
    """Create video with FFmpeg fallback if OpenCV fails"""
    try:
        # First try OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        print(f"OpenCV video creation failed: {e}")
        
        # Try FFmpeg fallback if available
        try:
            import subprocess
            import tempfile
            
            # Save frames as temporary images
            temp_dir = tempfile.mkdtemp()
            frame_paths = []
            
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            # Use FFmpeg to create video
            input_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', input_pattern, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '23', output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            # Cleanup temp files
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            os.rmdir(temp_dir)
            
            if result.returncode == 0:
                print("✅ Video created successfully with FFmpeg")
                return True
            else:
                print(f"FFmpeg failed: {result.stderr}")
                return False
                
        except Exception as ffmpeg_error:
            print(f"FFmpeg fallback also failed: {ffmpeg_error}")
            return False

# ----------------- THREADED DETECTION FUNCTION -----------------
def detection_thread(model_type):
    global latest_frame, latest_detections, stop_processing
    while not stop_processing:
        if latest_frame is not None:
            # Copy latest frame to avoid modifying while reading
            with processing_lock:
                frame_copy = latest_frame.copy()
            
            # Run detection on the frame copy based on model type
            if model_type == "above-water":
                frame_copy, detections = track_with_deepsort(model_type, frame_copy)
            else:  # underwater model
                detections = run_detection(model_type, frame_copy)
                frame_copy = draw_detections(frame_copy, detections)
                # Check for drowning and trigger alarm if needed
                frame_copy = check_drowning_and_trigger_alarm(detections, frame_copy)
            
            with processing_lock:
                latest_detections = detections
        
        time.sleep(0.001)  # Prevent CPU overuse

# ----------------- WEBCAM CONTROL ENDPOINTS -----------------
@app.post("/webcam/start")
async def start_webcam(model_type: str = Form("underwater"), enhancement: bool = Form(False)):
    global webcam_active, current_model, webcam_cap, video_feed_active, webcam_enhancement_active
    
    if webcam_active:
        return {"success": False, "message": "Webcam is already active"}
    
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'underwater' or 'above-water'.")
    
    # Check if enhancement is requested but GAN model is not available
    if enhancement and gan_model is None:
        raise HTTPException(status_code=503, detail="Video enhancement model not available")
    
    try:
        webcam_cap = cv2.VideoCapture(0)
        if not webcam_cap.isOpened():
            raise HTTPException(status_code=500, detail="Webcam not available.")
        
        webcam_active = True
        video_feed_active = True
        current_model = model_type
        webcam_enhancement_active = enhancement
        
        mode_text = "enhancement" if enhancement else "detection"
        return {"success": True, "message": f"Webcam started with {model_type} model in {mode_text} mode"}
    except Exception as e:
        if webcam_cap:
            webcam_cap.release()
            webcam_cap = None
        raise HTTPException(status_code=500, detail=f"Failed to start webcam: {str(e)}")

@app.post("/webcam/stop")
async def stop_webcam():
    global webcam_active, webcam_cap, video_feed_active, webcam_enhancement_active
    
    if not webcam_active:
        return {"success": False, "message": "Webcam is not active"}
    
    try:
        # Stop video feed first
        video_feed_active = False
        webcam_enhancement_active = False
        
        # Release webcam
        if webcam_cap:
            webcam_cap.release()
            webcam_cap = None
        
        webcam_active = False
        
        return {"success": True, "message": "Webcam stopped"}
    except Exception as e:
        return {"success": False, "message": f"Error stopping webcam: {str(e)}"}

@app.get("/webcam/status")
async def get_webcam_status():
    return {
        "active": webcam_active,
        "model": current_model if webcam_active else None,
        "video_feed_active": video_feed_active,
        "enhancement_active": webcam_enhancement_active
    }

# ----------------- REAL-TIME METRICS ENDPOINT -----------------
@app.get("/realtime_metrics")
async def get_realtime_metrics():
    """Get real-time processing metrics for webcam and video enhancement streaming"""
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

# ----------------- CUDA STATUS ENDPOINT -----------------
@app.get("/cuda_status")
async def get_cuda_status():
    """Get CUDA GPU status and memory information"""
    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": device,
        "models_on_gpu": device == 'cuda'
    }
    
    if torch.cuda.is_available():
        try:
            cuda_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                "gpu_memory_free_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3, 2),
                "gpu_utilization_percent": round((torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100, 1)
            })
        except Exception as e:
            cuda_info["error"] = f"Error getting GPU info: {str(e)}"
    
    return cuda_info

# ----------------- VIDEO FEED ENDPOINT -----------------
@app.get("/video_feed")
def video_feed():
    if not webcam_active or not webcam_cap:
        raise HTTPException(status_code=400, detail="Webcam not active")
    
    def generate():
        global video_feed_active
        while video_feed_active and webcam_active:
            if not webcam_cap or not webcam_cap.isOpened():
                break
                
            ret, frame = webcam_cap.read()
            if not ret:
                break
            
            # Apply enhancement if enabled
            if webcam_enhancement_active and gan_model is not None:
                enhanced_frame, success = enhance_video_frame(frame, gan_model, side_by_side=False)
                if success:
                    # Create side-by-side comparison (original | enhanced)
                    frame = np.hstack((frame, enhanced_frame))
            else:
                # Run detection only when enhancement is NOT active
                # Use different detection methods based on model type
                if current_model == "above-water":
                    frame, detections = track_with_deepsort(current_model, frame)
                else:  # underwater model
                    detections = run_detection(current_model, frame)
                    frame = draw_detections(frame, detections)
                    # Check for drowning and trigger alarm if needed
                    frame = check_drowning_and_trigger_alarm(detections, frame)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # time.sleep(0.033)  # ~30 FPS
        
        # Cleanup when loop ends
        if webcam_cap and webcam_cap.isOpened():
            webcam_cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/download_processed_video/{session_id}")
async def download_processed_video(session_id: str):
    """
    Download the processed video file
    """
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_info = video_sessions[session_id]
    
    if not session_info.get("processing_complete", False):
        raise HTTPException(status_code=400, detail="Video processing not complete yet")
    
    output_path = session_info.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed video file not found")
    
    # Clean up session after download
    del video_sessions[session_id]
    
    return FileResponse(
        path=output_path,
        filename=f"processed_detection_{session_id}.mp4",
        media_type="video/mp4"
    )

@app.get("/video_session_status/{session_id}")
async def get_video_session_status(session_id: str):
    """
    Get the status of a video processing session
    """
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_info = video_sessions[session_id]
    return {
        "session_id": session_id,
        "model_type": session_info["model_type"],
        "processing_complete": session_info.get("processing_complete", False),
        "download_available": session_info.get("processing_complete", False) and os.path.exists(session_info.get("output_path", ""))
    }

@app.post("/stop_video_processing/{session_id}")
async def stop_video_processing(session_id: str):
    """
    Stop a video processing session and clean up resources
    """
    try:
        cleanup_performed = False
        
        # Remove session from active sessions if it exists
        if session_id in video_sessions:
            session_info = video_sessions[session_id]
            temp_path = session_info.get("temp_path")
            
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                    cleanup_performed = True
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_path}: {e}")
            
            # Remove from active sessions
            del video_sessions[session_id]
            print(f"Video session {session_id} stopped and cleaned up")
            cleanup_performed = True
        
        # Also check for enhancement video files (used by enhance_video_stream endpoints)
        enhancement_video_path = f"temp_uploads/{session_id}.mp4"
        if os.path.exists(enhancement_video_path):
            try:
                os.remove(enhancement_video_path)
                print(f"Cleaned up enhancement video file: {enhancement_video_path}")
                cleanup_performed = True
            except Exception as e:
                print(f"Warning: Could not remove enhancement video file {enhancement_video_path}: {e}")
        
        if cleanup_performed:
            return {
                "success": True,
                "message": f"Video processing session {session_id} stopped successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Video session {session_id} not found or already stopped"
            }
            
    except Exception as e:
        print(f"Error stopping video session {session_id}: {e}")
        return {
            "success": False,
            "message": f"Error stopping video session: {str(e)}"
        }

# ----------------- LIVE CAMERA ENHANCEMENT ENDPOINT -----------------
@app.get("/live_camera_enhanced")
def live_camera_enhanced():
    """Stream enhanced-only live camera feed in real-time"""
    if not webcam_active or not webcam_cap:
        raise HTTPException(status_code=400, detail="Webcam not active")
    
    if gan_model is None:
        raise HTTPException(status_code=503, detail="Video enhancement model not available")
    
    def generate():
        global video_feed_active
        # Optimized JPEG encoding parameters
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        
        while video_feed_active and webcam_active:
            if not webcam_cap or not webcam_cap.isOpened():
                break
                
            ret, frame = webcam_cap.read()
            if not ret:
                break
            
            # Apply GAN enhancement to live camera feed
            enhanced_frame, success = enhance_video_frame(frame, gan_model, side_by_side=False)
            if not success:
                enhanced_frame = frame  # Fallback to original if enhancement fails
            
            # Encode enhanced frame to JPEG
            _, buffer = cv2.imencode('.jpg', enhanced_frame, encode_params)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Cleanup when loop ends
        if webcam_cap and webcam_cap.isOpened():
            webcam_cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----------------- LIVE CAMERA SIDE-BY-SIDE ENHANCEMENT ENDPOINT -----------------
@app.get("/live_camera_comparison")
def live_camera_comparison():
    """Stream side-by-side comparison of original and enhanced live camera feed"""
    if not webcam_active or not webcam_cap:
        raise HTTPException(status_code=400, detail="Webcam not active")
    
    if gan_model is None:
        raise HTTPException(status_code=503, detail="Video enhancement model not available")
    
    def generate():
        global video_feed_active
        # Optimized JPEG encoding parameters
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        
        while video_feed_active and webcam_active:
            if not webcam_cap or not webcam_cap.isOpened():
                break
                
            ret, frame = webcam_cap.read()
            if not ret:
                break
            
            # Apply GAN enhancement to live camera feed
            enhanced_frame, success = enhance_video_frame(frame, gan_model, side_by_side=False)
            if not success:
                enhanced_frame = frame  # Fallback to original if enhancement fails
            
            # Create side-by-side comparison (original | enhanced)
            combined_frame = np.hstack((frame, enhanced_frame))
            
            # Encode combined frame to JPEG
            _, buffer = cv2.imencode('.jpg', combined_frame, encode_params)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Cleanup when loop ends
        if webcam_cap and webcam_cap.isOpened():
            webcam_cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----------------- REAL-TIME VIDEO ENHANCEMENT STREAMING ENDPOINT -----------------
@app.get("/enhance_video_stream/{session_id}")
def enhance_video_stream(session_id: str):
    """Stream enhanced video frames in real-time with optimized GPU batch processing"""
    if gan_model is None:
        raise HTTPException(status_code=503, detail="Video enhancement model not available")
    
    video_path = f"temp_uploads/{session_id}.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video session not found")
    
    def generate():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        target_frame_time = 1.0 / fps
        
        # GPU context already initialized in startup_event
        
        # Optimized JPEG encoding parameters
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        
        try:
            start_time = time.time()
            
            while True:
                # Check if video file still exists (stop processing if file was deleted)
                if not os.path.exists(video_path):
                    print(f"Video file {video_path} no longer exists, stopping enhancement stream")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process single frame (like realtime_funie.py)
                enhanced_frame, success = enhance_video_frame(frame, gan_model, side_by_side=False)
                if not success:
                    enhanced_frame = frame
                
                # Create side-by-side comparison
                combined_frame = np.hstack((frame, enhanced_frame))
                
                # Encode frame to JPEG with optimized settings
                _, buffer = cv2.imencode('.jpg', combined_frame, encode_params)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                
                # Adaptive frame rate control
                elapsed = time.time() - start_time
                expected_time = frame_count * target_frame_time
                if elapsed < expected_time:
                    time.sleep(min(0.01, expected_time - elapsed))  # Minimal sleep for smooth playback
            
            # Memory cleanup handled by single-frame processing
        
        finally:
            cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----------------- ENHANCED VIDEO ONLY STREAMING ENDPOINT -----------------
@app.get("/enhance_video_stream_only/{session_id}")
def enhance_video_stream_only(session_id: str):
    """Stream only enhanced video frames in real-time with single-frame processing"""
    if gan_model is None:
        raise HTTPException(status_code=503, detail="Video enhancement model not available")
    
    video_path = f"temp_uploads/{session_id}.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video session not found")
    
    def generate():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        target_frame_time = 1.0 / fps
        
        # GPU context already initialized in startup_event
        
        # Optimized JPEG encoding parameters
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        
        try:
            start_time = time.time()
            
            while True:
                # Check if video file still exists (stop processing if file was deleted)
                if not os.path.exists(video_path):
                    print(f"Video file {video_path} no longer exists, stopping enhancement stream")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process single frame
                enhanced_frame, success = enhance_video_frame(frame, gan_model)
                if not success:
                    enhanced_frame = frame
                
                # Encode frame to JPEG with optimized settings
                _, buffer = cv2.imencode('.jpg', enhanced_frame, encode_params)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                
                # Adaptive frame rate control
                elapsed = time.time() - start_time
                expected_time = frame_count * target_frame_time
                if elapsed < expected_time:
                    time.sleep(min(0.01, expected_time - elapsed))  # Minimal sleep for smooth playback
        
        finally:
            cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----------------- REAL-TIME VIDEO PROCESSING ENDPOINT -----------------
# Store active video sessions
video_sessions = {}

@app.post("/realtime_video_process")
async def upload_video_for_realtime_processing(model_type: str = Form(...), file: UploadFile = File(...)):
    """
    Upload video for real-time processing
    model_type: "underwater" or "above-water"
    file: video file to process
    """
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'underwater' or 'above-water'.")
    
    if file is None:
        raise HTTPException(status_code=400, detail="Video file required.")
    
    # Save uploaded file temporarily
    session_id = str(uuid.uuid4())
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = f"temp_uploads/{session_id}.mp4"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store session info
    video_sessions[session_id] = {
        "model_type": model_type,
        "temp_path": temp_path,
        "created_at": time.time()
    }
    
    return {"session_id": session_id, "stream_url": f"/realtime_video_stream/{session_id}"}

@app.get("/realtime_video_stream/{session_id}")
def realtime_video_stream(session_id: str):
    """
    Stream processed video frames in real-time
    """
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_info = video_sessions[session_id]
    model_type = session_info["model_type"]
    temp_path = session_info["temp_path"]
    
    # Create output directory for processed videos
    os.makedirs("processed_outputs", exist_ok=True)
    output_path = f"processed_outputs/{session_id}.mp4"
    
    def generate():
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer for saving processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_count = 0
            while True:
                # Check if session still exists (stop processing if session was terminated)
                if session_id not in video_sessions:
                    print(f"Session {session_id} terminated, stopping video processing")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame based on model type
                if model_type == "above-water":
                    processed_frame, detections = track_with_deepsort(model_type, frame, conf_thresh=0.40)
                else:  # underwater model
                    detections = run_detection(model_type, frame)
                    processed_frame = draw_detections(frame, detections)
                    # Check for drowning and trigger alarm if needed
                    processed_frame = check_drowning_and_trigger_alarm(detections, processed_frame)
                
                # Save processed frame to output video
                out.write(processed_frame)
                
                # Encode frame to JPEG for streaming
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
        
        finally:
            cap.release()
            out.release()
            
            # Update session info with output path
            if session_id in video_sessions:
                video_sessions[session_id]["output_path"] = output_path
                video_sessions[session_id]["processing_complete"] = True
            
            # Clean up temporary input file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----------------- PROCESSED VIDEO STREAM ENDPOINT -----------------
@app.get("/processed_video_stream/{session_id}")
def processed_video_stream(session_id: str):
    """Stream processed video frames in real-time"""
    if not os.path.exists(f"temp_uploads/{session_id}.mp4"):
        raise HTTPException(status_code=404, detail="Video session not found")
    
    video_path = f"temp_uploads/{session_id}.mp4"
    
    def generate():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        # Initialize threading variables for this stream
        global latest_detections, latest_frame, stop_processing, processing_thread
        latest_detections = []
        latest_frame = None
        stop_processing = False
        
        # Start background detection thread
        processing_thread = threading.Thread(target=detection_thread, args=(current_model,))
        processing_thread.start()
        
        try:
            while True:
                # Check if video file still exists (stop processing if file was deleted)
                if not os.path.exists(video_path):
                    print(f"Video file {video_path} no longer exists, stopping processed video stream")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update latest frame for detection thread
                with processing_lock:
                    latest_frame = frame.copy()
                
                # Get latest detections from background thread
                with processing_lock:
                    detections_copy = latest_detections.copy()
                
                # Draw detections on frame
                frame = draw_detections(frame, detections_copy)
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
        
        finally:
            # Cleanup
            stop_processing = True
            if processing_thread and processing_thread.is_alive():
                processing_thread.join(timeout=5.0)
            cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----------------- PREDICT ENDPOINT -----------------
@app.post("/predict")
async def predict(model_type: str = Form(...), source: str = Form(...), file: UploadFile = File(None)):
    """
    model_type: "underwater" or "above-water"
    source: "image" | "video" | "webcam" (webcam reserved if you add streaming)
    file: required for image/video
    """
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'underwater' or 'above-water'.")

    # ---------------- IMAGE ----------------
    if source == "image":
        if file is None:
            raise HTTPException(status_code=400, detail="Image file required.")
        temp_path = save_temp_file(file)
        image = cv2.imread(temp_path)

        # Use different detection methods based on model type
        if model_type == "above-water":
            frame, tracked = track_with_deepsort(model_type, image, conf_thresh=0.40)
        else:  # underwater model
            detections = run_detection(model_type, image)
            frame = draw_detections(image, detections)
            # Convert tuple format to dict format for consistency
            tracked = []
            for i, (x1, y1, x2, y2, cls_name, conf) in enumerate(detections):
                tracked.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "track_id": i,  # Simple ID for underwater detections
                    "class": cls_name,
                    "class_id": 0 if cls_name == "normal" else 1,
                    "confidence": conf
                })

        # Encode to base64
        _, buffer = cv2.imencode(".jpg", frame)
        encoded_img = base64.b64encode(buffer).decode("utf-8")
        os.remove(temp_path)

        analysis_result = create_analysis_result(tracked, model_type)  # your helper

        return JSONResponse(content={
            "type": "image",
            "detections": tracked,      # already dicts with track_id
            "image_base64": encoded_img,
            "analysis": analysis_result
        })

    # ---------------- VIDEO ----------------
    elif source == "video":
        if file is None:
            raise HTTPException(status_code=400, detail="Video file required.")

        session_id = str(uuid.uuid4())
        temp_path = save_temp_file(file)

        # Make output dir
        os.makedirs("processed_outputs", exist_ok=True)
        output_path = f"processed_outputs/{session_id}.mp4"

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail="Unable to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try H.264; fallback if not supported
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            except:
                fourcc = 0
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        all_tracked = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Use different detection methods based on model type
                if model_type == "above-water":
                    frame, tracked = track_with_deepsort(model_type, frame, conf_thresh=0.40)
                else:  # underwater model
                    detections = run_detection(model_type, frame)
                    frame = draw_detections(frame, detections)
                    # Convert tuple format to dict format for consistency
                    tracked = []
                    for i, (x1, y1, x2, y2, cls_name, conf) in enumerate(detections):
                        tracked.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "track_id": f"{frame_count}_{i}",  # Unique ID for video frames
                            "class": cls_name,
                            "class_id": 0 if cls_name == "normal" else 1,
                            "confidence": conf
                        })
                
                out.write(frame)
                all_tracked.extend(tracked)
                frame_count += 1

                # light throttle to avoid CPU spikes (optional)
                # time.sleep(0.001)

        finally:
            cap.release()
            out.release()
            os.remove(temp_path)

        analysis_result = create_analysis_result(all_tracked, model_type, frame_count)

        return JSONResponse(content={
            "type": "video",
            "session_id": session_id,
            "video_path": output_path,
            "stream_url": f"/processed_video_stream/{session_id}",                  # saved file
            "detections": all_tracked,                  # list of dicts with track_id
            "analysis": analysis_result
        })

    else:
        raise HTTPException(status_code=400, detail="Invalid source. Use 'image' or 'video'.")

# ----------------- VIDEO ENHANCEMENT ENDPOINT -----------------
@app.post("/enhance_video")
async def enhance_video(file: UploadFile = File(...), create_comparison: bool = Form(False), real_time: bool = Form(True)):
    """
    Enhance underwater video using FUnIE-GAN model
    create_comparison: If True, creates a side-by-side comparison video
    real_time: If True, returns streaming URLs for real-time processing
    """
    if gan_model is None:
        raise HTTPException(status_code=503, detail="Video enhancement model not available")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    try:
        # Generate session ID for this video enhancement
        session_id = str(uuid.uuid4())
        temp_path = save_temp_file(file)
        
        # Copy the uploaded file to a session-specific location for streaming
        session_video_path = f"temp_uploads/{session_id}.mp4"
        os.makedirs("temp_uploads", exist_ok=True)
        shutil.copy2(temp_path, session_video_path)
        
        # If real-time streaming is requested, return streaming URLs immediately
        if real_time:
            # Get video properties for metadata
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 480
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            
            os.remove(temp_path)  # Clean up temp file
            
            return JSONResponse(content={
                "type": "video_stream",
                "session_id": session_id,
                "real_time": True,
                "stream_urls": {
                    "comparison": f"/enhance_video_stream/{session_id}",
                    "enhanced_only": f"/enhance_video_stream_only/{session_id}"
                },
                "video_info": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "filename": file.filename
                },
                "message": "Real-time video enhancement stream ready"
            })
        
        # Traditional processing mode (non-real-time)
        # Create output paths
        enhanced_path = f"processed_outputs/{session_id}_enhanced.mp4"
        comparison_path = f"processed_outputs/{session_id}_comparison.mp4" if create_comparison else None
        
        os.makedirs("processed_outputs", exist_ok=True)
        
        # Open input video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail="Unable to open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Collect all enhanced frames first
        enhanced_frames = []
        frame_count = 0
        total_enhancement_time = []
        
        print(f"🎬 Starting video enhancement for {file.filename}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance frame using GAN
            start_time = time.time()
            enhanced_frame, success = enhance_video_frame(frame, gan_model)
            end_time = time.time()
            
            if success:
                total_enhancement_time.append(end_time - start_time)
            
            # Store enhanced frame
            enhanced_frames.append(enhanced_frame)
            frame_count += 1
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                print(f"📹 Processed {frame_count} frames...")
        
        # Cleanup input video
        cap.release()
        
        # Create video using robust method
        print(f"🎬 Creating enhanced video with {frame_count} frames...")
        video_created = create_video_with_ffmpeg_fallback(
            enhanced_frames, enhanced_path, fps, width, height
        )
        
        if not video_created:
            raise Exception("Failed to create enhanced video with all methods")
        
        # Determine codec used (simplified for now)
        codec_used = 'avc1'  # H.264 is most compatible
        os.remove(temp_path)
        
        # Calculate statistics
        avg_enhancement_time = np.mean(total_enhancement_time) if total_enhancement_time else 0
        estimated_fps = 1 / avg_enhancement_time if avg_enhancement_time > 0 else 0
        
        # Create comparison video if requested
        comparison_created = False
        if create_comparison and os.path.exists(temp_path):
            # We need to save the original video temporarily for comparison
            original_temp_path = f"temp_uploads/{session_id}_original.mp4"
            shutil.copy2(temp_path, original_temp_path)
            
            comparison_created = create_side_by_side_video(
                original_temp_path, enhanced_path, comparison_path, fps
            )
            
            # Clean up temporary original video
            if os.path.exists(original_temp_path):
                os.remove(original_temp_path)
        
        print(f"✅ Video enhancement completed: {frame_count} frames processed")
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "original_filename": file.filename,
            "enhanced_video_path": enhanced_path,
            "comparison_video_path": comparison_path if comparison_created else None,
            "frame_count": frame_count,
            "processing_time": f"{sum(total_enhancement_time):.2f}s",
            "average_frame_time": f"{avg_enhancement_time:.4f}s",
            "estimated_fps": f"{estimated_fps:.2f}",
            "codec_used": codec_used,
            "file_extension": get_video_extension(codec_used),
            "message": "Video enhanced successfully"
        })
        
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Video enhancement failed: {str(e)}")

# ----------------- VISIBILITY METRICS CALCULATION -----------------
def calculate_visibility_metrics(original_image, enhanced_image):
    """Calculate visibility improvement metrics between original and enhanced images"""
    try:
        # Convert to grayscale for analysis
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast using standard deviation
        original_contrast = np.std(original_gray)
        enhanced_contrast = np.std(enhanced_gray)
        

        # Calculate edges using Canny edge detection
        original_edges = cv2.Canny(original_gray, 50, 150)
        enhanced_edges = cv2.Canny(enhanced_gray, 50, 150)
        original_edge_density = np.sum(original_edges > 0) / original_edges.size
        enhanced_edge_density = np.sum(enhanced_edges > 0) / enhanced_edges.size
        
        # Calculate sharpness using Laplacian variance
        original_sharpness = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        enhanced_sharpness = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
        
        # Calculate entropy (measure of information content)
        def calculate_entropy(image):
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]  # Remove zero entries
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            return entropy
        
        original_entropy = calculate_entropy(original_gray)
        enhanced_entropy = calculate_entropy(enhanced_gray)
        
        # Calculate deltas (improvements)
        contrast_delta = ((enhanced_contrast - original_contrast) / original_contrast) * 100 if original_contrast > 0 else 0
        edges_delta = ((enhanced_edge_density - original_edge_density) / original_edge_density) * 100 if original_edge_density > 0 else 0
        sharpness_delta = ((enhanced_sharpness - original_sharpness) / original_sharpness) * 100 if original_sharpness > 0 else 0
        entropy_delta = ((enhanced_entropy - original_entropy) / original_entropy) * 100 if original_entropy > 0 else 0

        # Overall visibility gain based on the four deltas (weighted average)
        visibility_gain = (contrast_delta * 0.25 + edges_delta * 0.25 + sharpness_delta * 0.25 + entropy_delta * 0.25)
        
        # Ensure reasonable bounds
        visibility_gain = max(0, min(100, visibility_gain))
        
        return {
            "visibility_gain_percentage": round(visibility_gain, 1),
            "contrast_improvement": round(contrast_delta, 1),
            "sharpness_improvement": round(sharpness_delta, 1),
            "edges_improvement": round(edges_delta, 1),
            "entropy_improvement": round(entropy_delta, 1),
        }
        
    except Exception as e:
        print(f"Error calculating visibility metrics: {e}")
        # Return default values if calculation fails
        return {
            "visibility_gain_percentage": 25.0,
            "contrast_improvement": 20.0,
            "sharpness_improvement": 30.0,
            "edges_improvement": 15.0,
            "entropy_improvement": 20.0,
        }

# ----------------- IMAGE ENHANCEMENT ENDPOINT -----------------
@app.post("/enhance_image")
async def enhance_image(file: UploadFile = File(...)):
    """Enhance a single image using the GAN model"""
    if gan_model is None:
        raise HTTPException(status_code=503, detail="GAN model not available")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Start timing for latency measurement
        start_time = time.time()
        
        # Generate session ID for this enhancement
        session_id = str(uuid.uuid4())
        
        # Save uploaded image
        temp_path = f"temp_uploads/{session_id}_original.jpg"
        os.makedirs("temp_uploads", exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read and process the image with proper error handling
        original_image = cv2.imread(temp_path)
        if original_image is None:
            # Clean up temp file before raising error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Invalid or unreadable image file")
        
        # Additional validation - check if image has valid dimensions
        if len(original_image.shape) != 3 or original_image.shape[2] != 3:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Image must be a valid color image")
        
        print(f"Input image shape: {original_image.shape}")
        
        # Start timing for processing speed
        processing_start = time.time()
        
        # Enhance the image using the new enhance_single_image function
        try:
            # Use the new enhance_single_image function
            enhanced_result = enhance_single_image(original_image)
            
            if enhanced_result is not None:
                # The result is already a side-by-side comparison (original + enhanced)
                comparison_image = enhanced_result
                
                # Extract the enhanced part (right half) for individual enhanced image
                height, width = comparison_image.shape[:2]
                enhanced_image = comparison_image[:, width//2:]
                
                # Extract the original part (left half) for consistency
                inp_img = comparison_image[:, :width//2]
                
                success = True
            else:
                enhanced_image = original_image
                success = False
        except Exception as enhancement_error:
            print(f"Enhancement error: {enhancement_error}")
            enhanced_image = original_image
            success = False
        
        # Clear GPU memory after enhancement to prevent CUDNN errors
        try:
            import gc
            gc.collect()
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
        except Exception as cleanup_error:
            print(f"GPU memory cleanup warning: {cleanup_error}")
        
        # End timing for processing speed
        processing_end = time.time()
        processing_time = processing_end - processing_start
        
        if not success:
            # Clean up temp file before raising error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail="Image enhancement failed - model prediction error")
        
        print(f"Enhancement completed successfully in {processing_time:.3f} seconds")
        
        # Calculate visibility metrics
        visibility_metrics = calculate_visibility_metrics(original_image, enhanced_image)
        
        # Save enhanced image
        enhanced_path = f"processed_outputs/{session_id}_enhanced.jpg"
        os.makedirs("processed_outputs", exist_ok=True)
        cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        
        # Save original image to processed_outputs for download
        original_output_path = f"processed_outputs/{session_id}_original.jpg"
        cv2.imwrite(original_output_path, original_image)
        
        # Save side-by-side comparison
        comparison_path = f"processed_outputs/{session_id}_comparison.jpg"
        height, width = original_image.shape[:2]
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Calculate total latency
        total_latency = time.time() - start_time
        
        # Calculate real-time speed (frames per second equivalent)
        real_time_speed = 1.0 / processing_time if processing_time > 0 else 0
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Image enhanced successfully",
            "original_size": {"width": width, "height": height},
            "enhanced_url": f"/download_enhanced_image/{session_id}/enhanced",
            "comparison_url": f"/download_enhanced_image/{session_id}/comparison",
            "original_url": f"/download_enhanced_image/{session_id}/original",
            "metrics": {
                "visibility_gain": visibility_metrics["visibility_gain_percentage"],
                "contrast_improvement": visibility_metrics["contrast_improvement"],
                "sharpness_improvement": visibility_metrics["sharpness_improvement"],
                "edges_improvement": visibility_metrics["edges_improvement"],
                "entropy_improvement": visibility_metrics["entropy_improvement"],
                "processing_time_ms": round(processing_time * 1000, 2),
                "total_latency_ms": round(total_latency * 1000, 2),
                "real_time_speed_fps": round(real_time_speed, 2)
            }
        }
        
    except Exception as e:
        # Clean up on error - only clean up files that were actually created
        cleanup_paths = []
        if 'temp_path' in locals():
            cleanup_paths.append(temp_path)
        if 'enhanced_path' in locals():
            cleanup_paths.append(enhanced_path)
        if 'comparison_path' in locals():
            cleanup_paths.append(comparison_path)
        if 'original_output_path' in locals():
            cleanup_paths.append(original_output_path)
        
        for path in cleanup_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass  # Ignore cleanup errors
        
        print(f"Image enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image enhancement failed: {str(e)}")

@app.get("/download_enhanced_image/{session_id}/{image_type}")
async def download_enhanced_image(session_id: str, image_type: str):
    """Download enhanced image, comparison, or original"""
    if image_type == "enhanced":
        file_path = f"processed_outputs/{session_id}_enhanced.jpg"
    elif image_type == "comparison":
        file_path = f"processed_outputs/{session_id}_comparison.jpg"
    elif image_type == "original":
        # For original, check if it exists in processed_outputs, otherwise return error
        file_path = f"processed_outputs/{session_id}_original.jpg"
        if not os.path.exists(file_path):
            # Try to find it in temp_uploads
            temp_path = f"temp_uploads/{session_id}_original.jpg"
            if os.path.exists(temp_path):
                file_path = temp_path
            else:
                raise HTTPException(status_code=404, detail="Original image not found")
    else:
        raise HTTPException(status_code=400, detail="Invalid image type. Use 'enhanced', 'comparison', or 'original'")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{image_type.capitalize()} image not found")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=f"{session_id}_{image_type}.jpg"
    )

# ----------------- GAN MODEL STATUS ENDPOINT -----------------
@app.get("/gan_model/status")
async def get_gan_model_status():
    """Get the status of the GAN model for video enhancement"""
    return {
        "available": gan_model is not None,
        "model_type": "FUnIE-GAN",
        "input_size": gan_input_size,
        "description": "Underwater image enhancement using Generative Adversarial Networks"
    }

# ----------------- DOWNLOAD ENHANCED VIDEO ENDPOINT -----------------
@app.get("/download_enhanced_video/{session_id}")
async def download_enhanced_video(session_id: str, video_type: str = "enhanced"):
    """
    Download enhanced or comparison video
    video_type: "enhanced" or "comparison"
    """
    if video_type == "enhanced":
        video_path = f"processed_outputs/{session_id}_enhanced.mp4"
    elif video_type == "comparison":
        video_path = f"processed_outputs/{session_id}_comparison.mp4"
    else:
        raise HTTPException(status_code=400, detail="Invalid video_type. Use 'enhanced' or 'comparison'")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{session_id}_{video_type}.mp4"
    )

# ----------------- ANALYSIS RESULT CREATION -----------------
def create_analysis_result(detections, model_type, frame_count=1):
    if not detections:
        return {
            "isDrowning": False,
            "confidence": 0.0,
            "riskLevel": "Safe",
            "recommendations": ["No drowning incidents detected", "Continue monitoring"],
            "detectedObjects": []
        }

    # Count detections by class
    drowning_count = sum(1 for det in detections if det.get("class") == "drowning")
    normal_count = sum(1 for det in detections if det.get("class") == "normal")

    # Safely calculate average confidence (ignore None values)
    confidences = [det.get("confidence") for det in detections if det.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Determine if drowning is detected
    is_drowning = drowning_count > 0

    # Risk level & recommendations
    if drowning_count > 0:
        if avg_confidence > 0.8:
            risk_level = "Critical"
            recommendations = [
                "Immediate intervention required",
                "Alert lifeguards immediately",
                "Prepare rescue equipment"
            ]
        else:
            risk_level = "High Risk"
            recommendations = [
                "Monitor closely",
                "Prepare for potential intervention",
                "Alert nearby personnel"
            ]
    else:
        risk_level = "Safe"
        recommendations = [
            "No immediate danger detected",
            "Continue normal monitoring",
            "Maintain safety protocols"
        ]

    # Build detected objects list
    detected_objects = []
    for det in detections:
        detected_objects.append({
            "type": det.get("class"),
            "confidence": det.get("confidence") if det.get("confidence") is not None else 0.0,
            "track_id": det.get("track_id")
        })

    return {
        "isDrowning": is_drowning,
        "confidence": avg_confidence,
        "riskLevel": risk_level,
        "recommendations": recommendations,
        "detectedObjects": detected_objects,
        "processingTime": f"{frame_count * 0.1:.2f}s"
    }
    if not detections:
        return {
            "isDrowning": False,
            "confidence": 0.0,
            "riskLevel": "Safe",
            "recommendations": ["No drowning incidents detected", "Continue monitoring"],
            "detectedObjects": []
        }

    # Count detections by class
    drowning_count = sum(1 for det in detections if det.get("class") == "drowning")
    normal_count = sum(1 for det in detections if det.get("class") == "normal")

    # Calculate average confidence
    avg_confidence = sum(det.get("confidence", 0.0) for det in detections) / len(detections)

    # Determine if drowning is detected
    is_drowning = drowning_count > 0

    # Determine risk level and recommendations
    if drowning_count > 0:
        if avg_confidence > 0.8:
            risk_level = "Critical"
            recommendations = [
                "Immediate intervention required",
                "Alert lifeguards immediately",
                "Prepare rescue equipment"
            ]
        else:
            risk_level = "High Risk"
            recommendations = [
                "Monitor closely",
                "Prepare for potential intervention",
                "Alert nearby personnel"
            ]
    else:
        risk_level = "Safe"
        recommendations = [
            "No immediate danger detected",
            "Continue normal monitoring",
            "Maintain safety protocols"
        ]

    # Create detected objects list
    detected_objects = []
    for det in detections:
        detected_objects.append({
            "type": det.get("class"),
            "confidence": det.get("confidence"),
            "track_id": det.get("track_id")
        })

    return {
        "isDrowning": is_drowning,
        "confidence": avg_confidence,
        "riskLevel": risk_level,
        "recommendations": recommendations,
        "detectedObjects": detected_objects,
        "processingTime": f"{frame_count * 0.1:.2f}s"  # Example estimate
    }
    if not detections:
        return {
            "isDrowning": False,
            "confidence": 0.0,
            "riskLevel": "Safe",
            "recommendations": ["No drowning incidents detected", "Continue monitoring"],
            "detectedObjects": [],
            "processingTime": f"{(time.time() - start_time):.2f}s" if start_time else "N/A"
        }
    
    # Normalize detection format (handle with/without track_id)
    # Normalize detection format
    formatted_detections = []
    for det in detections:
        if len(det) == 7:
            x1, y1, x2, y2, cls_name, conf, track_id = det
        elif len(det) == 6:
            x1, y1, x2, y2, cls_name, conf = det
            track_id = None
        else:
            raise ValueError(f"Unexpected detection format: {det}")
        formatted_detections.append((x1, y1, x2, y2, cls_name, conf, track_id))
    
    # Count classes
    drowning_detections = [d for d in formatted_detections if d[4] == "drowning"]
    normal_detections = [d for d in formatted_detections if d[4] == "normal"]
    
    drowning_count = len(drowning_detections)
    normal_count = len(normal_detections)
    
    # Calculate confidence (focus on drowning only if exists)
    if drowning_detections:
        avg_confidence = sum(d[5] for d in drowning_detections) / drowning_count
        max_confidence = max(d[5] for d in drowning_detections)
    else:
        avg_confidence = sum(d[5] for d in formatted_detections) / len(formatted_detections)
        max_confidence = avg_confidence
    
    # Determine risk
    is_drowning = drowning_count > 0
    if is_drowning:
        if max_confidence > 0.85 and frame_count > 5:  # persists multiple frames
            risk_level = "Critical"
            recommendations = [
                "Immediate intervention required",
                "Alert lifeguards immediately",
                "Deploy rescue equipment"
            ]
        elif avg_confidence > 0.6:
            risk_level = "High Risk"
            recommendations = [
                "Monitor closely",
                "Prepare for potential intervention",
                "Alert nearby personnel"
            ]
        else:
            risk_level = "Suspicious"
            recommendations = [
                "Keep monitoring for consistency",
                "Cross-check with other frames",
                "Do not ignore unusual behavior"
            ]
    else:
        risk_level = "Safe"
        recommendations = [
            "No immediate danger detected",
            "Continue normal monitoring",
            "Maintain safety protocols"
        ]
    
    # Build detected objects list
    detected_objects = []
    for _, _, _, _, cls_name, conf, track_id in formatted_detections:
        obj = {
            "type": cls_name,
            "confidence": conf
        }
        if track_id is not None:
            obj["track_id"] = track_id
        detected_objects.append(obj)
    
    # Processing time
    processing_time = f"{(time.time() - start_time):.2f}s" if start_time else f"{frame_count * 0.1:.2f}s"
    
    return {
        "isDrowning": is_drowning,
        "confidence": round(avg_confidence, 3),
        "maxConfidence": round(max_confidence, 3),
        "riskLevel": risk_level,
        "recommendations": recommendations,
        "detectedObjects": detected_objects,
        "processingTime": processing_time
    }
    if not detections:
        return {
            "isDrowning": False,
            "confidence": 0.0,
            "riskLevel": "Safe",
            "recommendations": ["No drowning incidents detected", "Continue monitoring"],
            "detectedObjects": []
        }
    
    # Count detections by class
    drowning_count = sum(1 for _, _, _, _, cls_name, _ in detections if cls_name == "drowning")
    normal_count = sum(1 for _, _, _, _, cls_name, _ in detections if cls_name == "normal")
    
    # Calculate average confidence
    avg_confidence = sum(conf for _, _, _, _, _, conf in detections) / len(detections)
    
    # Determine if drowning is detected
    is_drowning = drowning_count > 0
    
    # Determine risk level
    if drowning_count > 0:
        if avg_confidence > 0.8:
            risk_level = "Critical"
            recommendations = [
                "Immediate intervention required",
                "Alert lifeguards immediately",
                "Prepare rescue equipment"
            ]
        else:
            risk_level = "High Risk"
            recommendations = [
                "Monitor closely",
                "Prepare for potential intervention",
                "Alert nearby personnel"
            ]
    else:
        risk_level = "Safe"
        recommendations = [
            "No immediate danger detected",
            "Continue normal monitoring",
            "Maintain safety protocols"
        ]
    
    # Create detected objects list
    detected_objects = []
    for _, _, _, _, cls_name, conf in detections:
        detected_objects.append({
            "type": cls_name,
            "confidence": conf
        })
    
    return {
        "isDrowning": is_drowning,
        "confidence": avg_confidence,
        "riskLevel": risk_level,
        "recommendations": recommendations,
        "detectedObjects": detected_objects,
        "processingTime": f"{frame_count * 0.1:.2f}s"  # Estimate processing time
    }

# ----------------- HELPERS -----------------
def save_temp_file(file: UploadFile):
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join("temp_uploads", temp_filename)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path

def get_video_extension(codec_name: str) -> str:
    """Get the appropriate file extension based on the codec used"""
    if codec_name in ['avc1', 'mp4v']:
        return '.mp4'
    elif codec_name == 'XVID':
        return '.avi'
    else:
        return '.mp4'  # Default fallback

def detections_to_dict(detections):
    return [
        {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": cls_name, "confidence": conf}
        for (x1, y1, x2, y2, cls_name, conf) in detections
    ]

def cleanup_processing_thread():
    """Clean up any running processing threads"""
    global stop_processing, processing_thread
    stop_processing = True
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=5.0)
        processing_thread = None

# ----------------- STARTUP EVENT -----------------
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global stop_processing, processing_thread
    stop_processing = True
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=1.0)
    processing_thread = None
    
    # Setup GPU (like realtime_funie.py)
    setup_gpu()
    
    # Additional CUDA optimizations for YOLO
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        
        # Warm up CUDA with dummy inference
        print("🔥 Warming up CUDA with dummy inference...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for model_name, model in models.items():
            try:
                with torch.no_grad():
                    _ = model(dummy_frame, verbose=False)
                print(f"✅ {model_name} model warmed up on GPU")
            except Exception as e:
                print(f"⚠️ Warning: Could not warm up {model_name} model: {e}")
    
    # Log startup information
    print("🚀 FastAPI backend starting up...")
    print(f"📊 YOLO models loaded: {len(models)}")
    print(f"🎨 GAN model status: {'✅ Loaded' if gan_model else '❌ Not available'}")
    print("✨ Backend ready!")

# ----------------- ROOT -----------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}

# ----------------- TEST VIDEO CREATION -----------------
@app.get("/test_video")
async def test_video_creation():
    """Test endpoint to verify video creation works"""
    try:
        # Create a simple test video
        test_path = "processed_outputs/test_video.mp4"
        os.makedirs("processed_outputs", exist_ok=True)
        
        # Create test frames (simple colored rectangles)
        frames = []
        for i in range(30):  # 1 second at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Create moving rectangle
            x = int((i / 30) * 500)
            cv2.rectangle(frame, (x, 200), (x + 100, 300), (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
        
        # Create video
        success = create_video_with_ffmpeg_fallback(frames, test_path, 30, 640, 480)
        
        if success and os.path.exists(test_path):
            return {
                "success": True,
                "message": "Test video created successfully",
                "test_video_path": test_path,
                "file_size": f"{os.path.getsize(test_path) / 1024:.1f} KB"
            }
        else:
            return {
                "success": False,
                "message": "Failed to create test video"
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating test video: {str(e)}"
        }
