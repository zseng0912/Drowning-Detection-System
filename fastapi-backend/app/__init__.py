"""
Drowning Detection System - FastAPI Backend

A modular FastAPI backend for real-time drowning detection with YOLO and GAN enhancement.

Modules:
- main.py: FastAPI app, startup event, and router includes
- models_loader.py: YOLO + GAN model loading and warm-up
- gpu_utils.py: TensorFlow + PyTorch GPU setup
- detection.py: Detection and enhancement helper functions
- routes.py: All API endpoints
- utils.py: Metrics calculation, file saving, cleanup
"""

from .main import app

__all__ = ["app"]
