"""
FastAPI Routes

This module contains all API endpoints for the drowning detection system.
"""

import os
import uuid
import shutil
import base64
import time
import threading
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List

from .detection import (
    track_with_deepsort, detection_with_enhance, enhance_frame, 
    get_realtime_metrics, check_drowning_and_trigger_alarm
)
from .utils import (
    save_temp_file, get_video_extension, calculate_visibility_metrics,
    create_analysis_result, create_side_by_side_video, create_video_with_ffmpeg_fallback
)
from .gpu_utils import get_cuda_status


# Create router
router = APIRouter()

# Dependency to get models from app state
def get_models(request: Request):
    """Get models from app state."""
    return {
        "yolo_models": request.app.state.models,
        "deepsort": request.app.state.deepsort,
        "gan_model": request.app.state.gan_model
    }

# Global variables for webcam and video processing
webcam_active = False
current_model = "underwater"
webcam_cap = None
video_feed_active = False
webcam_enhancement_active = False
stop_processing = False
processing_thread = None
video_sessions = {}
current_confidence_threshold = 0.30  # Default confidence threshold

# Data models
class LoginRequest(BaseModel):
    """Pydantic model for user login requests."""
    email: str
    password: str

class User(BaseModel):
    """Pydantic model for user data."""
    id: int
    email: str
    name: str
    role: str
    certifications: List[str]
    shift: str
    status: str
    avatar: str

# Mock user database
users_db = [
    {
        "id": 1,
        "email": "lifeguard@admin.com",
        "password": "password123",
        "name": "Lifeguard / Pool Admin",
        "role": "Administrator",
        "certifications": ["Management"],
        "shift": "Full Day (8:00 AM - 6:00 PM)",
        "status": "on-duty",
        "avatar": "PA"
    }
]


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@router.post("/login")
def login(request: LoginRequest):
    """Authenticate user login credentials."""
    user = next((u for u in users_db if u["email"] == request.email and u["password"] == request.password), None)
    if user:
        user_data = user.copy()
        user_data.pop("password")
        return {"success": True, "token": "fake-jwt-token", "user": user_data}
    else:
        return {"success": False, "message": "Invalid email or password. Try Again!"}


# =============================================================================
# WEBCAM CONTROL ENDPOINTS
# =============================================================================

@router.post("/webcam/start")
async def start_webcam(request: Request, model_type: str = Form("underwater"), enhancement: bool = Form(False), confidence_threshold: float = Form(0.50)):
    """Start webcam capture with specified detection model, enhancement options, and confidence threshold."""
    global webcam_active, current_model, webcam_cap, video_feed_active, webcam_enhancement_active, current_confidence_threshold
    
    if webcam_active:
        return {"success": False, "message": "Webcam is already active"}
    
    # Get models from app state
    models = request.app.state.models
    gan_model = request.app.state.gan_model
    
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'underwater' or 'above-water'.")
    
    # Validate confidence threshold
    if not (0.0 <= confidence_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
    
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
        current_confidence_threshold = confidence_threshold
        
        mode_text = "enhancement" if enhancement else "detection"
        return {"success": True, "message": f"Webcam started with {model_type} model in {mode_text} mode"}
    except Exception as e:
        if webcam_cap:
            webcam_cap.release()
            webcam_cap = None
        raise HTTPException(status_code=500, detail=f"Failed to start webcam: {str(e)}")

@router.post("/webcam/stop")
async def stop_webcam():
    """Stop webcam capture."""
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

@router.get("/webcam/status")
async def get_webcam_status():
    """Get current webcam status."""
    return {
        "active": webcam_active,
        "model": current_model if webcam_active else None,
        "video_feed_active": video_feed_active,
        "enhancement_active": webcam_enhancement_active,
        "confidence_threshold": current_confidence_threshold if webcam_active else None
    }

@router.post("/webcam/set_confidence")
async def set_confidence_threshold(confidence_threshold: float = Form(...)):
    """Update the confidence threshold for detection."""
    global current_confidence_threshold
    
    # Validate confidence threshold
    if not (0.0 <= confidence_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
    
    current_confidence_threshold = confidence_threshold
    
    return {
        "success": True,
        "message": f"Confidence threshold updated to {confidence_threshold}",
        "confidence_threshold": current_confidence_threshold
    }


# =============================================================================
# SYSTEM STATUS AND MONITORING ENDPOINTS
# =============================================================================

@router.get("/realtime_metrics")
async def get_realtime_metrics_endpoint():
    """Get current real-time processing metrics for webcam and video processing."""
    return get_realtime_metrics()

@router.get("/cuda_status")
async def get_cuda_status_endpoint():
    """Get CUDA GPU status and memory information for system diagnostics."""
    return get_cuda_status()

@router.get("/gan_model/status")
async def get_gan_model_status(request: Request):
    """Get the status of the GAN model for video enhancement."""
    gan_model = request.app.state.gan_model
    return {
        "available": gan_model is not None,
        "model_type": "FUnIE-GAN",
        "description": "Underwater image enhancement using Generative Adversarial Networks"
    }


# =============================================================================
# VIDEO STREAMING ENDPOINTS
# =============================================================================

@router.get("/video_feed")
def video_feed(request: Request):
    """Stream real-time video feed with drowning detection and optional enhancement."""
    if not webcam_active or not webcam_cap:
        raise HTTPException(status_code=400, detail="Webcam not active")
    
    models = request.app.state.models
    gan_model = request.app.state.gan_model
    deepsort = request.app.state.deepsort
    
    def generate():
        global video_feed_active
        while video_feed_active and webcam_active:
            if not webcam_cap or not webcam_cap.isOpened():
                break
                
            ret, frame = webcam_cap.read()
            if not ret:
                break
            
            # Apply enhancement and detection based on model and settings
            if current_model == "above-water":
                # Above-water model: run detection with DeepSORT (no enhancement)
                frame, detections = track_with_deepsort(models[current_model], frame, deepsort, conf_thresh=current_confidence_threshold, is_video=True)
            else:  # underwater model
                # Underwater model: apply enhancement if enabled, then run detection
                if webcam_enhancement_active and gan_model is not None:
                    # Apply enhancement first, then run detection on enhanced frame
                    frame, detections = detection_with_enhance(frame, models[current_model], gan_model, enhancement=True, conf_thresh=current_confidence_threshold)
                else:
                    # Run detection without enhancement
                    frame, detections = detection_with_enhance(frame, models[current_model], gan_model, enhancement=False, conf_thresh=current_confidence_threshold)

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Cleanup when loop ends
        if webcam_cap and webcam_cap.isOpened():
            webcam_cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/live_camera_comparison")
def live_camera_comparison(request: Request):
    """Stream side-by-side comparison of original and enhanced live camera feed."""
    if not webcam_active or not webcam_cap:
        raise HTTPException(status_code=400, detail="Webcam not active")
    
    gan_model = request.app.state.gan_model
    
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
            enhanced_frame, success = enhance_frame(frame, gan_model, side_by_side=False)
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


# =============================================================================
# DETECTION ENDPOINTS
# =============================================================================

@router.post("/predict")
async def predict(request: Request, model_type: str = Form(...), source: str = Form(...), file: UploadFile = File(None), enhancement: bool = Form(False), confidence_threshold: float = Form(0.50)):
    """Perform drowning detection on uploaded images with optional enhancement and confidence threshold."""
    models = request.app.state.models
    gan_model = request.app.state.gan_model
    deepsort = request.app.state.deepsort
    
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'underwater' or 'above-water'.")
    
    # Validate confidence threshold
    if not (0.0 <= confidence_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")

    # IMAGE PROCESSING
    if source == "image":
        if file is None:
            raise HTTPException(status_code=400, detail="Image file required.")
        temp_path = save_temp_file(file)
        image = cv2.imread(temp_path)

        # Use different detection methods based on model type
        if model_type == "above-water":
            frame, tracked = track_with_deepsort(models[model_type], image, deepsort, conf_thresh=confidence_threshold, is_video=False)
        else:  # underwater model
            frame, detections = detection_with_enhance(image, models[model_type], gan_model, enhancement=enhancement, conf_thresh=confidence_threshold)
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

        analysis_result = create_analysis_result(tracked, model_type)

        return JSONResponse(content={
            "type": "image",
            "detections": tracked,
            "image_base64": encoded_img,
            "analysis": analysis_result
        })

    else:
        raise HTTPException(status_code=400, detail="Invalid source. Use 'image' or 'video'.")


# =============================================================================
# VIDEO PROCESSING ENDPOINTS
# =============================================================================

@router.post("/realtime_video_process")
async def upload_video_for_realtime_processing(request: Request, model_type: str = Form(...), file: UploadFile = File(...), enhancement: bool = Form(False), confidence_threshold: float = Form(0.50)):
    """Upload video for real-time processing with YOLO detection, optional enhancement, and confidence threshold."""
    models = request.app.state.models
    gan_model = request.app.state.gan_model
    
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'underwater' or 'above-water'.")
    
    # Validate confidence threshold
    if not (0.0 <= confidence_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
    
    if file is None:
        raise HTTPException(status_code=400, detail="Video file required.")
    
    # Check enhancement parameter for underwater model
    if enhancement and model_type == "underwater" and gan_model is None:
        raise HTTPException(status_code=400, detail="GAN model not available for enhancement.")
    
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
        "enhancement": enhancement,
        "confidence_threshold": confidence_threshold,
        "created_at": time.time()
    }
    
    return {"session_id": session_id, "stream_url": f"/realtime_video_stream/{session_id}"}

@router.get("/realtime_video_stream/{session_id}")
def realtime_video_stream(request: Request, session_id: str):
    """Stream processed video frames with real-time YOLO detection and enhancement."""
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    models = request.app.state.models
    gan_model = request.app.state.gan_model
    deepsort = request.app.state.deepsort
    
    session_info = video_sessions[session_id]
    model_type = session_info["model_type"]
    temp_path = session_info["temp_path"]
    enhancement = session_info.get("enhancement", False)
    confidence_threshold = session_info.get("confidence_threshold", 0.50)
    
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
                    processed_frame, detections = track_with_deepsort(models[model_type], frame, deepsort, conf_thresh=confidence_threshold, is_video=True)
                else:  # underwater model
                    processed_frame, detections = detection_with_enhance(frame, models[model_type], gan_model, enhancement=enhancement, conf_thresh=confidence_threshold)
                
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

@router.get("/download_processed_video/{session_id}")
async def download_processed_video(session_id: str):
    """Download the processed video file for a given session."""
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

@router.get("/video_session_status/{session_id}")
async def get_video_session_status(session_id: str):
    """Get the current status of a video processing session."""
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_info = video_sessions[session_id]
    return {
        "session_id": session_id,
        "model_type": session_info["model_type"],
        "processing_complete": session_info.get("processing_complete", False),
        "download_available": session_info.get("processing_complete", False) and os.path.exists(session_info.get("output_path", ""))
    }

@router.post("/stop_video_processing/{session_id}")
async def stop_video_processing(session_id: str):
    """Stop a video processing session and clean up associated resources."""
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


# =============================================================================
# IMAGE AND VIDEO ENHANCEMENT ENDPOINTS
# =============================================================================

@router.post("/enhance_image")
async def enhance_image(request: Request, file: UploadFile = File(...)):
    """Enhance a single image using the GAN model."""
    gan_model = request.app.state.gan_model
    
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
        
        # Enhance the image using the enhance_frame function
        try:
            # Use the enhance_frame function
            comparison_image, success = enhance_frame(original_image, gan_model, side_by_side=True)

            if success and comparison_image is not None:
                # Extract the enhanced part (right half)
                height, width = comparison_image.shape[:2]
                enhanced_image = comparison_image[:, width//2:]
                
                # Extract the original part (left half)
                inp_img = comparison_image[:, :width//2]
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
        cv2.imwrite(enhanced_path, enhanced_image)
        
        # Save original image to processed_outputs for download
        original_output_path = f"processed_outputs/{session_id}_original.jpg"
        cv2.imwrite(original_output_path, original_image)
        
        # Save side-by-side comparison
        comparison_path = f"processed_outputs/{session_id}_comparison.jpg"
        height, width = original_image.shape[:2]
        cv2.imwrite(comparison_path, comparison_image)
        
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

@router.get("/download_enhanced_image/{session_id}/{image_type}")
async def download_enhanced_image(session_id: str, image_type: str):
    """Download enhanced image, comparison, or original."""
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

@router.post("/enhance_video")
async def enhance_video(request: Request, file: UploadFile = File(...), create_comparison: bool = Form(False), real_time: bool = Form(True)):
    """Enhance underwater video using FUnIE-GAN model with streaming support."""
    gan_model = request.app.state.gan_model
    
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
            enhanced_frame, success = enhance_frame(frame, gan_model)
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

@router.get("/enhance_video_stream/{session_id}")
def enhance_video_stream(request: Request, session_id: str):
    """Stream enhanced video frames in real-time with optimized GPU processing."""
    gan_model = request.app.state.gan_model
    
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
                enhanced_frame, success = enhance_frame(frame, gan_model, side_by_side=False)
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
            
        finally:
            cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/download_enhanced_video/{session_id}")
async def download_enhanced_video(session_id: str, video_type: str = "enhanced"):
    """Download enhanced or comparison video."""
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
