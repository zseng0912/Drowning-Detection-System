"""
Utility Functions

This module contains helper functions for file handling, metrics calculation,
visualization, and other utility operations.
"""

import os
import uuid
import shutil
import time
import cv2
import numpy as np
from fastapi import UploadFile


def save_temp_file(file: UploadFile):
    """Save uploaded file to temporary location."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join("temp_uploads", temp_filename)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


def get_video_extension(codec_name: str) -> str:
    """Get the appropriate file extension based on the codec used."""
    if codec_name in ['avc1', 'mp4v']:
        return '.mp4'
    elif codec_name == 'XVID':
        return '.avi'
    else:
        return '.mp4'  # Default fallback


def calculate_visibility_metrics(original_image, enhanced_image): 
    """Calculate visibility improvement metrics between original and enhanced images."""
    try:
        # Ensure both images are in the same color space
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_image
        
        if len(enhanced_image.shape) == 3:
            enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_gray = enhanced_image

        # Contrast (std deviation of pixel intensities)
        original_contrast = float(np.std(original_gray))
        enhanced_contrast = float(np.std(enhanced_gray))

        # Edge density (Canny)
        original_edges = cv2.Canny(original_gray, 50, 150)
        enhanced_edges = cv2.Canny(enhanced_gray, 50, 150)
        original_edge_density = np.mean(original_edges > 0)
        enhanced_edge_density = np.mean(enhanced_edges > 0)

        # Sharpness (variance of Laplacian)
        original_sharpness = float(cv2.Laplacian(original_gray, cv2.CV_64F).var())
        enhanced_sharpness = float(cv2.Laplacian(enhanced_gray, cv2.CV_64F).var())

        # Entropy (Shannon)
        def calculate_entropy(image):
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            prob = hist / (hist.sum() + 1e-8)
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log2(prob)))

        original_entropy = calculate_entropy(original_gray)
        enhanced_entropy = calculate_entropy(enhanced_gray)

        # Calculate safe relative deltas
        def safe_delta(new, old, inverse=False):
            if old <= 1e-6:
                return 0.0
            change = ((new - old) / old) * 100.0
            return -change if inverse else change

        contrast_delta = safe_delta(enhanced_contrast, original_contrast)
        edges_delta = safe_delta(enhanced_edge_density, original_edge_density, inverse=True)
        sharpness_delta = safe_delta(enhanced_sharpness, original_sharpness, inverse=True)
        entropy_delta = safe_delta(enhanced_entropy, original_entropy)

        # Weighted visibility gain
        visibility_gain = (
            contrast_delta * 0.3 +
            edges_delta * 0.3 +
            sharpness_delta * 0.3 +
            entropy_delta * 0.1
        )

        return {
            "visibility_gain_percentage": round(max(0, visibility_gain), 1),
            "contrast_improvement": round(contrast_delta, 1),
            "sharpness_improvement": round(sharpness_delta, 1),
            "edges_improvement": round(edges_delta, 1),
            "entropy_improvement": round(entropy_delta, 1),
        }

    except Exception as e:
        print(f"Error calculating visibility metrics: {e}")
        return {
            "visibility_gain_percentage": 0.0,
            "contrast_improvement": 0.0,
            "sharpness_improvement": 0.0,
            "edges_improvement": 0.0,
            "entropy_improvement": 0.0,
        }


def create_analysis_result(detections, model_type, frame_count=1):
    """Create analysis result from detections."""
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

    # Safely calculate average confidence
    confidences = [det.get("confidence") for det in detections if det.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Determine if drowning is detected
    is_drowning = drowning_count > 0

    # Risk level & recommendations
    if drowning_count > 0:
        if avg_confidence > 0.7:
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


def create_side_by_side_video(original_path, enhanced_path, output_path, fps=30):
    """Create a side-by-side comparison video."""
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
    """Create video with FFmpeg fallback if OpenCV fails."""
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
