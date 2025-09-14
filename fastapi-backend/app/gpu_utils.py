"""
GPU Setup and Optimization Utilities

This module handles GPU configuration for both TensorFlow and PyTorch,
including CUDA optimizations and memory management.
"""

import torch
import tensorflow as tf


def setup_gpu():
    """Configure GPU settings for TensorFlow."""
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


def optimize_tensorflow_for_inference():
    """Optimize TensorFlow settings for real-time inference."""
    # Enable mixed precision for faster inference
    try:
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
    except:
        pass
    
    # Set thread settings for optimal performance
    # tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
    # tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores


def setup_pytorch_cuda():
    """Setup PyTorch CUDA optimizations."""
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        return True
    return False


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_cuda_status():
    """Get CUDA GPU status and memory information for system diagnostics."""
    cuda_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": get_device(),
        "models_on_gpu": get_device() == 'cuda'
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
