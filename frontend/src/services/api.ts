// API service module for AquaGuard AI pool surveillance system backend communication
// API base URL
const API_BASE_URL = 'http://127.0.0.1:8000';

// TypeScript interfaces for API request/response data structures
export interface LoginCredentials {
  email: string;
  password: string;
}

export interface User {
  id: number;
  email: string;
  name: string;
  role: string;
  certifications: string[];
  shift: string;
  status: string;
  avatar: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface WebcamStatus {
  active: boolean;
  model: string | null;
  enhancement_active?: boolean;
}

export interface PredictionResult {
  type: 'image' | 'video' | 'camera';
  detections?: any[];
  image_base64?: string;
  video_path?: string;
  session_id?: string;
  stream_url?: string;
  analysis?: {
    isDrowning: boolean;
    confidence: number;
    riskLevel: string;
    recommendations: string[];
    detectedObjects: any[];
    processingTime: string;
  };
  // Camera-specific properties
  isStreamActive?: boolean;
  model?: string;
  message?: string;
}

export interface VideoEnhancementResult {
  success: boolean;
  session_id: string;
  original_filename: string;
  enhanced_video_path: string;
  comparison_video_path?: string;
  frame_count: number;
  processing_time: string;
  average_frame_time: string;
  estimated_fps: string;
  codec_used?: string;
  file_extension?: string;
  message: string;
  // Real-time streaming properties
  real_time?: boolean;
  stream_url?: string;
  stream_only_url?: string;
  fps?: number;
  width?: number;
  height?: number;
}

export interface GanModelStatus {
  available: boolean;
  model_type: string;
}

export interface ImageEnhancementResult {
  success: boolean;
  session_id: string;
  message: string;
  original_size: {
    width: number;
    height: number;
  };
  enhanced_url: string;
  comparison_url: string;
  original_url: string;
  metrics?: {
    visibility_gain: number;
    contrast_improvement: number;
    sharpness_improvement: number;
    edges_improvement: number;
    entropy_improvement: number;
    processing_time_ms: number;
    total_latency_ms: number;
    real_time_speed_fps: number;
  };
}

export interface RealTimeMetrics {
  processing_speed_ms: number;
  real_time_speed_fps: number;
  latency_per_frame_ms: number;
  metrics_available: boolean;
  last_updated?: number;
  message?: string;
}

// Main API object containing all backend communication functions
export const api = {
  // Webcam control functions for live camera feed management
  async startWebcam(modelType: string = 'underwater', enhancement: boolean = false): Promise<{ success: boolean; message: string }> {
    const formData = new FormData();
    formData.append('model_type', modelType);
    formData.append('enhancement', enhancement.toString());
    
    const response = await fetch(`${API_BASE_URL}/webcam/start`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Stop webcam feed and cleanup resources
  async stopWebcam(): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE_URL}/webcam/stop`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Get current webcam status and configuration
  async getWebcamStatus(): Promise<WebcamStatus> {
    const response = await fetch(`${API_BASE_URL}/webcam/status`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // AI prediction functions for drowning detection analysis
  async predict(modelType: string, source: 'image' | 'video', file: File, enhancement: boolean = false): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('model_type', modelType);
    formData.append('source', source);
    formData.append('file', file);
    if (modelType === 'underwater') {
      formData.append('enhancement', enhancement.toString());
    }
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Authentication functions for user login
  async login(email: string, password: string): Promise<{ success: boolean; token?: string; user?: any; message?: string }> {
    const response = await fetch(`${API_BASE_URL}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // User logout and session cleanup
  async logout(): Promise<void> {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
  },

  // Video enhancement functions using GAN models for underwater clarity improvement
  async enhanceVideo(file: File, createComparison: boolean = false, realTime: boolean = false): Promise<VideoEnhancementResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('create_comparison', createComparison.toString());
    formData.append('real_time', realTime.toString());
    
    const response = await fetch(`${API_BASE_URL}/enhance_video`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Check GAN model availability and status
  async getGanModelStatus(): Promise<GanModelStatus> {
    const response = await fetch(`${API_BASE_URL}/gan_model/status`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Generate URL for downloading enhanced video results
  getEnhancedVideoUrl(sessionId: string, videoType: 'enhanced' | 'comparison' = 'enhanced'): string {
    return `${API_BASE_URL}/download_enhanced_video/${sessionId}?video_type=${videoType}`;
  },

  // Get real-time side-by-side comparison stream URL
  getRealTimeSideBySideUrl(sessionId: string): string {
    return `${API_BASE_URL}/enhance_video_stream/${sessionId}`;
  },

  // Image enhancement functions for single frame processing
  async enhanceImage(file: File): Promise<ImageEnhancementResult> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/enhance_image`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Image enhancement failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Generate URL for downloading enhanced image results
  getEnhancedImageUrl(sessionId: string, imageType: 'enhanced' | 'comparison' | 'original' = 'enhanced'): string {
    return `${API_BASE_URL}/download_enhanced_image/${sessionId}/${imageType}`;
  },

  // Get live camera comparison stream URL
  getLiveCameraComparisonUrl(): string {
    return `${API_BASE_URL}/live_camera_comparison`;
  },

  // Get webcam video feed stream URL
  getWebcamVideoFeedUrl(): string {
    return `${API_BASE_URL}/video_feed`;
  },

  // Real-time performance metrics for processing speed monitoring
  async getRealTimeMetrics(): Promise<RealTimeMetrics> {
    const response = await fetch(`${API_BASE_URL}/realtime_metrics`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Real-time video processing with live streaming capabilities
  async processVideoRealtime(modelType: string, file: File, enhancement: boolean = false): Promise<string> {
    const formData = new FormData();
    formData.append('model_type', modelType);
    formData.append('file', file);
    if (modelType === 'underwater') {
      formData.append('enhancement', enhancement.toString());
    }
    
    // Upload video and get session info
    const response = await fetch(`${API_BASE_URL}/realtime_video_process`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Return the streaming URL and session ID
    return {
      streamUrl: `${API_BASE_URL}${data.stream_url}`,
      sessionId: data.session_id,
      modelType: data.model_type
    };
  },

  // Check status of ongoing video processing session
  async getVideoSessionStatus(sessionId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/video_session_status/${sessionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },

  // Download processed video with detection results
  async downloadProcessedVideo(sessionId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/download_processed_video/${sessionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `processed_detection_${sessionId}.mp4`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },

  // Stop ongoing video processing session and cleanup
  async stopVideoProcessing(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE_URL}/stop_video_processing/${sessionId}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }
};

// Utility functions for authentication and local storage management
export const setAuthToken = (token: string) => {
  localStorage.setItem('authToken', token);
};

// Retrieve stored authentication token
export const getAuthToken = (): string | null => {
  return localStorage.getItem('authToken');
};

// Store user data in local storage
export const setUser = (user: User) => {
  localStorage.setItem('user', JSON.stringify(user));
};

// Retrieve stored user data
export const getUser = (): User | null => {
  const userStr = localStorage.getItem('user');
  return userStr ? JSON.parse(userStr) : null;
};

// Check if user is currently authenticated
export const isAuthenticated = (): boolean => {
  return !!getAuthToken();
};