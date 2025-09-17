# FastAPI Backend

A comprehensive backend system for drowning detection and underwater video enhancement using YOLO models and FUnIE-GAN.

## 🚀 Features

- **Drowning Detection**: Real-time detection using YOLO models (underwater and above-water)
- **Video Enhancement**: Underwater video quality improvement using FUnIE-GAN
- **Webcam Support**: Live video feed with real-time detection
- **File Processing**: Support for image and video uploads
- **Side-by-side Comparison**: Enhanced vs. original video comparison

## 📋 Prerequisites

- Python 3.8+
- FastAPI and Uvicorn
- OpenCV (cv2)
- TensorFlow/Keras
- PyTorch (for YOLO models)
- Ultralytics

## 🛠️ Installation

1. Install required packages:
   ```bash
   pip install fastapi uvicorn opencv-python tensorflow torch ultralytics
   ```

2. Ensure model files are in the correct directories:
   - YOLO models: `model/detection_model/`
   - GAN model: `model/FUnIE_GAN_model/`

## 🏃‍♂️ How to Run

1. Navigate to the `fastapi-backend` directory:
   ```bash
   cd fastapi-backend
   ```

2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

3. Visit [http://localhost:8000](http://localhost:8000) in your browser

4. API documentation available at [http://localhost:8000/docs](http://localhost:8000/docs)

## 🔌 API Endpoints

### Core Detection
- `POST /predict` - Process images/videos for drowning detection
- `POST /webcam/start` - Start webcam with detection
- `POST /webcam/stop` - Stop webcam
- `GET /video_feed` - Live webcam feed with detection

### Video Enhancement
- `POST /enhance_video` - Enhance underwater videos using GAN
- `GET /gan_model/status` - Check GAN model availability
- `GET /download_enhanced_video/{session_id}` - Download enhanced videos

### Authentication
- `POST /login` - User authentication

## 🔧 Configuration

- **Model Paths**: Configured in `main.py` under model loading sections
- **Output Directories**: Videos saved to `processed_outputs/`
- **Temporary Files**: Stored in `temp_uploads/`

## 🚨 Troubleshooting

- **Dependencies**: Ensure all required packages are installed
- **Model Files**: Verify model files exist in correct directories
- **Logs**: Check console output for detailed error messages 
