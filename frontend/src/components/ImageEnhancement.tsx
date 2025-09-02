import React, { useState, useEffect } from 'react';
import { ArrowRight, Play, Pause, RotateCcw, Settings, Zap, Eye, Upload, Camera, Monitor, FileVideo, Image as ImageIcon, Download, AlertCircle } from 'lucide-react';
import { api, VideoEnhancementResult, GanModelStatus, ImageEnhancementResult, RealTimeMetrics } from '../services/api';

export default function ImageEnhancement() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputMode, setInputMode] = useState<'upload' | 'camera'>('upload');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [isBackendWebcamActive, setIsBackendWebcamActive] = useState(false);
  const [webcamStatus, setWebcamStatus] = useState<any>(null);
  const [enhancementResult, setEnhancementResult] = useState<VideoEnhancementResult | null>(null);
  const [imageEnhancementResult, setImageEnhancementResult] = useState<ImageEnhancementResult | null>(null);
  const [ganModelStatus, setGanModelStatus] = useState<GanModelStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [createComparison, setCreateComparison] = useState(true);
  const [isRealTimeStreaming, setIsRealTimeStreaming] = useState(false);
  const [streamSessionId, setStreamSessionId] = useState<string | null>(null);
  const [realTimeMetrics, setRealTimeMetrics] = useState<RealTimeMetrics | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const videoRef = React.useRef<HTMLVideoElement>(null);

  // Check GAN model status on component mount
  useEffect(() => {
    checkGanModelStatus();
  }, []);

  // Fetch real-time metrics when streaming is active
   useEffect(() => {
     let interval: NodeJS.Timeout;
     
     if (isRealTimeStreaming || (inputMode === 'camera' && isBackendWebcamActive)) {
       interval = setInterval(async () => {
         try {
           const metrics = await api.getRealTimeMetrics();
           setRealTimeMetrics(metrics);
         } catch (error) {
           console.error('Failed to fetch real-time metrics:', error);
         }
       }, 1000); // Update every second
     }
     
     return () => {
       if (interval) {
         clearInterval(interval);
       }
     };
   }, [isRealTimeStreaming, inputMode, isBackendWebcamActive]);

  const checkGanModelStatus = async () => {
    try {
      const status = await api.getGanModelStatus();
      setGanModelStatus(status);
      if (!status.available) {
        setError('GAN model is not available. Video enhancement will be disabled.');
      }
    } catch (err) {
      console.error('Failed to check GAN model status:', err);
      setError('Failed to check GAN model status.');
    }
  };

  const startCamera = async () => {
    try {
      // Start backend webcam with enhancement if GAN model is available
      const enhancementEnabled = ganModelStatus?.available || false;
      const result = await api.startWebcam('underwater', enhancementEnabled);
      
      if (result.success) {
        setIsBackendWebcamActive(true);
        setIsStreamActive(true);
        
        // Check webcam status to get enhancement info
        const status = await api.getWebcamStatus();
        setWebcamStatus(status);
        
        console.log('Backend webcam started:', result.message);
      } else {
        throw new Error(result.message || 'Failed to start webcam');
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      setError(`Unable to start camera: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const stopCamera = async () => {
    try {
      if (isBackendWebcamActive) {
        const result = await api.stopWebcam();
        console.log('Backend webcam stopped:', result.message);
      }
      
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
      }
      
      setIsStreamActive(false);
      setIsBackendWebcamActive(false);
      setWebcamStatus(null);
      
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    } catch (error) {
      console.error('Error stopping camera:', error);
      setError(`Error stopping camera: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Cleanup camera stream on component unmount or mode change
  React.useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  React.useEffect(() => {
    if (inputMode === 'upload') {
      stopCamera();
    }
  }, [inputMode]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleProcessing = async () => {
    if (inputMode === 'upload' && !uploadedFile) return;
    if (inputMode === 'camera' && !isStreamActive) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      if (inputMode === 'upload' && uploadedFile) {
        // For video files, use real-time streaming
        if (uploadedFile.type.startsWith('video/')) {
          const result = await api.enhanceVideo(uploadedFile, createComparison, true);
          if (result.real_time && result.session_id) {
            setStreamSessionId(result.session_id);
            setIsRealTimeStreaming(true);
            setEnhancementResult(result);
            console.log('Real-time enhancement started:', result);
          } else {
            setEnhancementResult(result);
            console.log('Enhancement completed:', result);
          }
        } else {
          // For images, use the image enhancement endpoint
          const result = await api.enhanceImage(uploadedFile);
          setImageEnhancementResult(result);
          console.log('Image enhancement completed:', result);
        }
      } else if (inputMode === 'camera' && isStreamActive) {
        // Camera mode with backend enhancement
        if (!isBackendWebcamActive) {
          setError('Backend webcam is not active. Please start the camera first.');
          return;
        }
        
        // The enhancement is already running on the backend
        // We just need to show the enhanced stream
        setError(null);
        console.log('Camera enhancement is active via backend webcam');
      }
    } catch (err) {
      console.error('Enhancement failed:', err);
      setError(err instanceof Error ? err.message : 'Video enhancement failed');
      setIsRealTimeStreaming(false);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-600 to-blue-600 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2">Underwater Image Enhancement</h2>
        <p className="text-teal-100">Advanced AI-powered image processing for underwater surveillance</p>
        
        {/* GAN Model Status */}
        {/* {ganModelStatus && (
          <div className="mt-4 p-3 bg-white bg-opacity-20 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${ganModelStatus.available ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">
                GAN Model: {ganModelStatus.available ? 'Available' : 'Not Available'} 
                ({ganModelStatus.model_type})
              </span>
            </div>
          </div>
        )} */}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <span className="text-red-800">{error}</span>
          </div>
        </div>
      )}

      {/* Success Display */}
      {enhancementResult && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 bg-green-600 rounded-full flex items-center justify-center">
              <span className="text-white text-xs">✓</span>
            </div>
            <span className="text-green-800">
              Video enhanced successfully! Processed {enhancementResult.frame_count} frames in {enhancementResult.processing_time}.
            </span>
          </div>
        </div>
      )}
      
      {imageEnhancementResult && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 bg-green-600 rounded-full flex items-center justify-center">
              <span className="text-white text-xs">✓</span>
            </div>
            <span className="text-green-800">
              {imageEnhancementResult.message}
            </span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Enhancement Settings
          </h3>
          
          <div className="space-y-4">
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Enhancement Model</label>
              <div className="p-4 border-2 border-blue-500 bg-blue-50 rounded-lg">
                <div className="flex items-start space-x-3">
                  <input
                    type="radio"
                    name="model"
                    value="gan"
                    checked={true}
                    readOnly
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900">FUnIE-GAN Enhancement Model</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      Lightweight Generative Adversarial Network for real-time underwater image enhancement
                    </p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      <span>Real-time: ~25–30 FPS</span>
                      <span>Optimized for Edge Devices</span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {[
                        'Color Restoration',
                        'Contrast Adjustment',
                        'Deblurring',
                        'Detail Preservation',
                        'Lightweight & Fast (U-Net based)'
                      ].map((feature, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-gray-100 text-xs rounded"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Input Mode Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Input Source</label>
              <div className="flex space-x-2">
                <button
                  onClick={() => {
                    setInputMode('upload');
                    stopCamera();
                    setUploadedFile(null);
                    setPreviewUrl(null);
                    setEnhancementResult(null);
                  }}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg font-medium transition-colors text-sm ${
                    inputMode === 'upload'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <Upload className="w-4 h-4" />
                  <span>Upload</span>
                </button>
                <button
                  onClick={() => {
                    setInputMode('camera');
                    stopCamera();
                    setUploadedFile(null);
                    setPreviewUrl(null);
                    setEnhancementResult(null);
                  }}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg font-medium transition-colors text-sm ${
                    inputMode === 'camera'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <Camera className="w-4 h-4" />
                  <span>Live Camera</span>
                </button>
              </div>
            </div>

            {/* Comparison Toggle */}
            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={createComparison}
                  onChange={(e) => setCreateComparison(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-gray-700">Create side-by-side comparison</span>
              </label>
              <p className="text-xs text-gray-500 mt-1">Shows original and enhanced videos side by side</p>
            </div>

            {/* File Upload or Camera Selection */}
            {inputMode === 'upload' ? (
              <div className="space-y-4">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-colors"
                >
                  <div className="flex flex-col items-center space-y-2">
                    <div className="flex space-x-2">
                      <FileVideo className="w-6 h-6 text-gray-400" />
                      <ImageIcon className="w-6 h-6 text-gray-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">Upload Media</p>
                      <p className="text-xs text-gray-500">Video or Image</p>
                    </div>
                  </div>
                </div>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*,image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />

                {uploadedFile && (
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    {uploadedFile.type.startsWith('video/') ? (
                      <FileVideo className="w-5 h-5 text-blue-600" />
                    ) : (
                      <ImageIcon className="w-5 h-5 text-green-600" />
                    )}
                    <div>
                      <p className="font-medium text-sm">{uploadedFile.name}</p>
                      <p className="text-xs text-gray-500">
                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-gray-100 rounded-lg p-4 text-center space-y-3">
                <Camera className="w-8 h-8 text-gray-400 mx-auto" />
                <p className="text-sm text-gray-600">
                  {isBackendWebcamActive ? 
                    `Camera is active ${webcamStatus?.enhancement_active ? 'with enhancement' : 'in detection mode'}` : 
                    'Click to start camera with AI enhancement'}
                </p>
                {webcamStatus?.enhancement_active && (
                  <div className="text-xs text-green-600 font-medium">
                    ✓ Real-time GAN enhancement enabled
                  </div>
                )}
                {!ganModelStatus?.available && (
                  <div className="text-xs text-orange-600 font-medium">
                    ⚠ GAN model unavailable - detection mode only
                  </div>
                )}
                {!isStreamActive ? (
                  <button
                    onClick={startCamera}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    Start Camera
                  </button>
                ) : (
                  <button
                    onClick={stopCamera}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    Stop Camera
                  </button>
                )}
              </div>
            )}

            <button
              onClick={handleProcessing}
              disabled={isProcessing || (inputMode === 'upload' && !uploadedFile) || (inputMode === 'camera' && !isBackendWebcamActive) || (inputMode === 'camera' && !ganModelStatus?.available)}
              className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white py-3 px-4 rounded-lg font-semibold transition-colors"
            >
              <Zap className="w-5 h-5" />
              <span>
                {isProcessing ? 'Processing...' : 
                 inputMode === 'camera' && !isBackendWebcamActive ? 'Start Camera First' :
                 inputMode === 'camera' && !ganModelStatus?.available ? 'GAN Model Required for Enhancement' :
                 inputMode === 'camera' && webcamStatus?.enhancement_active ? 'Enhancement Active' :
                 inputMode === 'camera' ? 'Camera Ready (Detection Mode)' :
                 !ganModelStatus?.available ? 'GAN Model Unavailable' : 
                 'Apply Enhancement'}
              </span>
            </button>

            {/* Processing Status */}
            {isProcessing && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-sm text-blue-700">Processing video with GAN model...</span>
                </div>
                <div className="mt-2 text-xs text-blue-600">
                  This may take several minutes depending on video length and resolution.
                </div>
              </div>
            )}

            {/* Reset Button */}
            {(enhancementResult || imageEnhancementResult || isRealTimeStreaming) && !isProcessing && (
              <button
                onClick={() => {
                  setEnhancementResult(null);
                  setImageEnhancementResult(null);
                  setUploadedFile(null);
                  setPreviewUrl(null);
                  setError(null);
                  setIsRealTimeStreaming(false);
                  setStreamSessionId(null);
                  setRealTimeMetrics(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                  }
                }}
                className="w-full flex items-center justify-center space-x-2 bg-gray-500 hover:bg-gray-600 text-white py-3 px-4 rounded-lg font-semibold transition-colors mt-4"
              >
                <RotateCcw className="w-5 h-5" />
                <span>Start Over</span>
              </button>
            )}
          </div>
        </div>

        {/* Before/After Comparison */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Eye className="w-5 h-5 mr-2" />
              Enhancement Comparison
            </h3>
            
            <div className="space-y-6">
              {/* Labels Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-gray-900">
                    Original Media
                  </h4>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">
                    Enhanced Media
                  </h4>
                </div>
              </div>

              {/* Enhanced Video - Full Width */}
              <div className="space-y-3">
                {isProcessing && (
                  <div className="flex items-center justify-center py-2">
                    <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mr-2"></div>
                    <span className="text-sm text-gray-600">Processing...</span>
                  </div>
                )}
                <div className="relative rounded-lg overflow-hidden h-[400px]">
                  {isRealTimeStreaming && streamSessionId ? (
                    // Show real-time side-by-side streaming
                    <div className="w-full h-full">
                      <img 
                        src={api.getRealTimeSideBySideUrl(streamSessionId)}
                        alt="Real-time enhancement"
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                        Real-time Enhancement
                      </div>
                    </div>
                  ) : enhancementResult ? (
                    // Show actual enhanced video
                    <div>
                      <video
                        src={api.getEnhancedVideoUrl(enhancementResult.session_id, 'enhanced')}
                        controls
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          console.error('Video playback error:', e);
                          setError('Enhanced video cannot be played. The video may be corrupted or in an unsupported format.');
                        }}
                      />
                      {enhancementResult.codec_used && (
                        <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                          Codec: {enhancementResult.codec_used}
                        </div>
                      )}
                      {/* Video Format Info */}
                      <div className="mt-2 text-xs text-gray-600 text-center">
                        Format: {enhancementResult.file_extension || 'mp4'} | 
                        Frames: {enhancementResult.frame_count} | 
                        Time: {enhancementResult.processing_time}
                      </div>
                    </div>
                  ) : imageEnhancementResult ? (
                    // Show image enhancement results with side-by-side comparison
                    <div className="w-full h-[400px] flex">
                      {/* Original Image - Left Side */}
                      <div className="w-1/2 relative">
                        <img
                          src={api.getEnhancedImageUrl(imageEnhancementResult.session_id, 'original')}
                          alt="Original Image"
                          className="w-full h-full object-contain bg-gray-100"
                        />
                        <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                          Original
                        </div>
                      </div>
                      
                      {/* Enhanced Image - Right Side */}
                      <div className="w-1/2 relative">
                        <img
                          src={api.getEnhancedImageUrl(imageEnhancementResult.session_id, 'enhanced')}
                          alt="Enhanced Image"
                          className="w-full h-full object-contain bg-gray-100"
                        />
                        <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                          Enhanced
                        </div>
                      </div>
                    </div>
                  ) : isProcessing ? (
                    // Show processing state
                    <div className="w-full h-[400px] bg-gradient-to-b from-blue-300 to-blue-500 flex items-center justify-center">
                      <div className="text-center text-white">
                        <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                        <div className="text-lg font-bold">Processing...</div>
                        <div className="text-sm opacity-75">Enhancing your {uploadedFile?.type.startsWith('video/') ? 'video' : 'image'}</div>
                      </div>
                    </div>
                  ) : inputMode === 'camera' && isBackendWebcamActive && webcamStatus?.enhancement_active ? (
                    // Show live camera enhancement comparison
                    <div className="w-full h-[400px] relative">
                      <img 
                        src={api.getLiveCameraComparisonUrl()}
                        alt="Live camera enhancement comparison"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          console.error('Live camera stream error:', e);
                          setError('Unable to load live camera enhancement stream.');
                        }}
                      />
                      <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                        Live Enhancement Active
                      </div>
                      <div className="absolute top-2 left-2 bg-green-600 bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                        ● LIVE
                      </div>
                    </div>
                  ) : inputMode === 'camera' && isBackendWebcamActive ? (
                    // Show regular webcam feed (no enhancement)
                    <div className="w-full h-[400px] relative">
                      <img 
                        src={api.getWebcamVideoFeedUrl()}
                        alt="Live camera feed"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          console.error('Webcam feed error:', e);
                          setError('Unable to load webcam feed.');
                        }}
                      />
                      <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                        Detection Mode
                      </div>
                      <div className="absolute top-2 left-2 bg-blue-600 bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                        ● LIVE
                      </div>
                    </div>
                  ) : (
                    // Show placeholder with side-by-side layout
                    <div className="w-full h-[400px] flex">
                      {/* Original Feed - Left Side */}
                      <div className="w-1/2 bg-gradient-to-b from-gray-300 to-gray-500 flex items-center justify-center relative">
                        <div className="text-center text-white">
                          <div className="text-lg font-bold mb-2">Original Feed</div>
                          <div className="text-sm opacity-75">{inputMode === 'camera' ? 'Live Feed' : 'Original Media'}</div>
                          <div className="absolute top-8 left-8 w-12 h-12 bg-gray-400 rounded-full border-2 border-white animate-pulse"></div>
                          <div className="absolute bottom-12 right-12 w-8 h-8 bg-gray-600 rounded-full border-2 border-white"></div>
                        </div>
                      </div>
                      
                      {/* Enhanced Feed - Right Side */}
                      <div className="w-1/2 bg-gradient-to-b from-blue-300 to-blue-500 flex items-center justify-center relative">
                        <div className="text-center text-white">
                          <div className="text-lg font-bold mb-2">Enhanced Feed</div>
                          <div className="text-sm opacity-75">AI Enhanced {inputMode === 'camera' ? 'Feed' : 'Media'}</div>
                          {/* Enhanced visibility indicators */}
                          <div className="absolute top-8 left-8 w-12 h-12 bg-yellow-400 rounded-full border-2 border-white animate-pulse"></div>
                          <div className="absolute bottom-12 right-12 w-8 h-8 bg-orange-400 rounded-full border-2 border-white"></div>
                        </div>
                      </div>
                    </div>
                  )}
                  {/* <div className="absolute bottom-2 left-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                    {enhancementResult ? `Clarity: ${Math.round((enhancementResult.estimated_fps as any) * 10)}%` : 'Clarity: 92%'}
                  </div> */}
                </div>
              </div>
            </div>

            {/* Enhancement Arrow */}
            <div className="flex justify-center my-6">
              <div className="flex items-center space-x-4 bg-gray-100 px-6 py-3 rounded-full">
                <span className="text-sm font-medium text-gray-700">FUnIE-GAN</span>
                <ArrowRight className="w-6 h-6 text-blue-600" />
                <span className="text-sm font-medium text-green-600">
                   {realTimeMetrics ? `${Math.round(realTimeMetrics.processing_speed_ms)}ms Processing` :
                    imageEnhancementResult?.metrics ? `+${Math.round(imageEnhancementResult.metrics.visibility_gain)}% Visibility` : '+45% Visibility'}
                 </span>
              </div>
            </div>

            {/* Enhancement Metrics */}
            <div className="grid grid-cols-3 gap-4 mt-6">
              {/* For real-time video enhancement or webcam */}
              {(isRealTimeStreaming || (inputMode === 'camera' && isBackendWebcamActive)) ? (
                <>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                     <div className="text-2xl font-bold text-blue-600">
                       {realTimeMetrics ? `${Math.round(realTimeMetrics.processing_speed_ms)}ms` : '45ms'}
                     </div>
                     <div className="text-sm text-gray-600">Processing Speed</div>
                   </div>
                   <div className="bg-gray-50 p-4 rounded-lg text-center">
                     <div className="text-2xl font-bold text-green-600">
                       {realTimeMetrics ? `${Math.round(realTimeMetrics.real_time_speed_fps)} FPS` : '25 FPS'}
                     </div>
                     <div className="text-sm text-gray-600">Real-Time Speed</div>
                   </div>
                   <div className="bg-gray-50 p-4 rounded-lg text-center">
                     <div className="text-2xl font-bold text-purple-600">
                       {realTimeMetrics ? `${Math.round(realTimeMetrics.latency_per_frame_ms)}ms` : '40ms'}
                     </div>
                     <div className="text-sm text-gray-600">Latency / Frame</div>
                   </div>
                </>
              ) : (
                /* For image enhancement */
                <>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {imageEnhancementResult?.metrics ? `${Math.round(imageEnhancementResult.metrics.visibility_gain)}%` : '90%'}
                    </div>
                    <div className="text-sm text-gray-600">Visibility Gain</div>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {imageEnhancementResult?.metrics ? `${imageEnhancementResult.metrics.real_time_speed_fps} FPS` : '25 FPS'}
                    </div>
                    <div className="text-sm text-gray-600">Processing Speed</div>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {imageEnhancementResult?.metrics ? `${imageEnhancementResult.metrics.total_latency_ms}ms` : '40ms'}
                    </div>
                    <div className="text-sm text-gray-600">Total Latency</div>
                  </div>
                </>
              )}
            </div>

            {/* Detailed Enhancement Metrics - Only show for image enhancement, not for real-time video or webcam */}
            {imageEnhancementResult?.metrics && !isRealTimeStreaming && !(inputMode === 'camera' && isBackendWebcamActive) && (
              <div className="mt-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4 text-center">Detailed Enhancement Analysis</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white border border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-gray-900">Contrast Enhancement</h5>
                      <span className="text-lg font-bold text-orange-600">
                        {imageEnhancementResult.metrics.contrast_improvement > 0 ? '+' : ''}{Math.round(imageEnhancementResult.metrics.contrast_improvement)}%
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">Improvement in image contrast and dynamic range</p>
                  </div>
                  
                  <div className="bg-white border border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-gray-900">Edge Definition</h5>
                      <span className="text-lg font-bold text-teal-600">
                        {imageEnhancementResult.metrics.edges_improvement > 0 ? '+' : ''}{Math.round(imageEnhancementResult.metrics.edges_improvement)}%
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">Enhancement of edge clarity and object boundaries</p>
                  </div>
                  
                  <div className="bg-white border border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-gray-900">Sharpness Boost</h5>
                      <span className="text-lg font-bold text-indigo-600">
                        {imageEnhancementResult.metrics.sharpness_improvement > 0 ? '+' : ''}{Math.round(imageEnhancementResult.metrics.sharpness_improvement)}%
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">Increase in image sharpness and detail clarity</p>
                  </div>
                  
                  <div className="bg-white border border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-gray-900">Information Content</h5>
                      <span className="text-lg font-bold text-pink-600">
                        {imageEnhancementResult.metrics.entropy_improvement > 0 ? '+' : ''}{Math.round(imageEnhancementResult.metrics.entropy_improvement)}%
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">Enhancement of image information and texture detail</p>
                  </div>
                </div>
              </div>
            )}

            {/* Download Links */}
            {enhancementResult && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-3">Download Enhanced Videos</h4>
                <div className="flex flex-wrap gap-3">
                  <a
                    href={api.getEnhancedVideoUrl(enhancementResult.session_id, 'enhanced')}
                    download
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    <span>Enhanced Video</span>
                  </a>
                  {enhancementResult.comparison_video_path && (
                    <a
                      href={api.getEnhancedVideoUrl(enhancementResult.session_id, 'comparison')}
                      download
                      className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      <span>Comparison Video</span>
                    </a>
                  )}
                </div>
              </div>
            )}
            
            {/* Image Download Links */}
            {imageEnhancementResult && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-3">Download Enhanced Images</h4>
                <div className="flex flex-wrap gap-3">
                   <a
                     href={api.getEnhancedImageUrl(imageEnhancementResult.session_id, 'enhanced')}
                     download
                     className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                   >
                     <Download className="w-4 h-4" />
                     <span>Enhanced Image</span>
                   </a>
                   {imageEnhancementResult.comparison_url && (
                     <a
                       href={api.getEnhancedImageUrl(imageEnhancementResult.session_id, 'comparison')}
                       download
                       className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                     >
                       <Download className="w-4 h-4" />
                       <span>Comparison Image</span>
                     </a>
                   )}
                   <a
                     href={api.getEnhancedImageUrl(imageEnhancementResult.session_id, 'original')}
                     download
                     className="flex items-center space-x-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm font-medium transition-colors"
                   >
                     <Download className="w-4 h-4" />
                     <span>Original Image</span>
                   </a>
                 </div>
              </div>
            )}

            {/* Side-by-Side Comparison Video */}
            {enhancementResult && enhancementResult.comparison_video_path && (
              <div className="mt-6">
                <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                  <Eye className="w-5 h-5 mr-2" />
                  Side-by-Side Comparison
                </h4>
                <div className="aspect-video rounded-lg overflow-hidden bg-gray-100">
                  <video
                    src={api.getEnhancedVideoUrl(enhancementResult.session_id, 'comparison')}
                    controls
                    className="w-full h-full object-cover"
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Left: Original Video | Right: Enhanced Video
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}