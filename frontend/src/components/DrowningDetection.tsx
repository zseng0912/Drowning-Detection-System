import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Brain, FileVideo, Image as ImageIcon, AlertTriangle, CheckCircle, Clock, Target, Camera, Monitor, Square } from 'lucide-react';
import { api, WebcamStatus, PredictionResult } from '../services/api';

export default function DrowningDetection() {
  const [selectedModel, setSelectedModel] = useState('underwater');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [streamingUrl, setStreamingUrl] = useState<string | null>(null);
  const [realtimeStreamUrl, setRealtimeStreamUrl] = useState<string | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [downloadAvailable, setDownloadAvailable] = useState<boolean>(false);
  const [inputMode, setInputMode] = useState<'upload' | 'camera'>('upload');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [webcamStatus, setWebcamStatus] = useState<WebcamStatus | null>(null);
  const statusIntervalRef = useRef<number | null>(null);

  const models = [
    {
      id: 'underwater',
      name: 'Underwater Drowning Detection',
      description: 'YOLOv8l model for detecting drowning incidents in underwater camera feeds',
      accuracy: '0.937 mAP@0.5',
      processingTime: '2.453s',
      trainingData: '5167 samples',
      features: ['Object Detection','Object Tracking']
    },
    {
      id: 'above-water',
      name: 'Above Water Drowning Detection + DeepSORT',
      description: 'YOLOv8l model optimized for surface-level drowning detection and distress signals',
      accuracy: '0.916 mAP@0.5',
      processingTime: '2.407s',
      trainingData: '2843 samples',
      features: ['Object Detection', 'Object Counting', 'Multi-person Tracking with DeepSort','Unique Swimmer IDs']
    }
  ];

  // Check webcam status on component mount
  useEffect(() => {
    checkWebcamStatus();
  }, []);

  const checkWebcamStatus = async () => {
    try {
      const status = await api.getWebcamStatus();
      setWebcamStatus(status);
      setIsStreamActive(status.active);
    } catch (error) {
      console.error('Error checking webcam status:', error);
    }
  };

  // Start status polling when webcam is active
  const startStatusPolling = () => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
    }
    statusIntervalRef.current = setInterval(checkWebcamStatus, 3000);
    console.log('Status polling started');
  };

  // Stop status polling
  const stopStatusPolling = () => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
      statusIntervalRef.current = null;
      console.log('Status polling stopped');
    }
  };

  const startCamera = async () => {
    try {
      setIsProcessing(true);
      const result = await api.startWebcam(selectedModel);
      
      if (result.success) {
        setIsStreamActive(true);
        setWebcamStatus({ active: true, model: selectedModel });
        startStatusPolling(); // Start polling for status updates
      } else {
        alert('Failed to start camera: ' + result.message);
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      alert('Error starting camera. Please check if the backend is running.');
    } finally {
      setIsProcessing(false);
    }
  };

  const stopCamera = async () => {
    try {
      setIsProcessing(true);
      const result = await api.stopWebcam();
      
      if (result.success) {
        setIsStreamActive(false);
        setWebcamStatus({ active: false, model: null });
        setPreviewUrl(null);
        setPredictionResult(null); // Clear any previous results
        stopStatusPolling(); // Stop polling for status updates
        console.log('Camera stopped successfully');
      } else {
        alert('Failed to stop camera: ' + result.message);
      }
    } catch (error) {
      console.error('Error stopping camera:', error);
      alert('Error stopping camera. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopStatusPolling();
      if (isStreamActive) {
        api.stopWebcam().catch(console.error);
      }
    };
  }, [isStreamActive]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setUploadedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setStreamingUrl(null);
      setPredictionResult(null);
    }
  };
  
  const checkProcessingStatus = async (sessionId: string) => {
    try {
      const status = await api.getVideoSessionStatus(sessionId);
      if (status.processing_complete && status.download_available) {
        setDownloadAvailable(true);
      } else {
        // Check again after 2 seconds if not complete
        setTimeout(() => checkProcessingStatus(sessionId), 2000);
      }
    } catch (error) {
      console.error('Error checking processing status:', error);
    }
  };

  const handleDownloadVideo = async () => {
    if (!currentSessionId) return;
    
    try {
      await api.downloadProcessedVideo(currentSessionId);
    } catch (error) {
      console.error('Error downloading video:', error);
      alert('Failed to download video. Please try again.');
    }
  };

  const handleClearResults = async () => {
    try {
      // Stop webcam if active
      if (isStreamActive) {
        await stopCamera();
      }
      
      // Stop video processing session if active
      if (currentSessionId) {
        try {
          await api.stopVideoProcessing(currentSessionId);
          console.log('Video processing session stopped');
        } catch (error) {
          console.error('Error stopping video processing:', error);
          // Continue with cleanup even if stopping fails
        }
      }
      
      // Clear all state
      setPredictionResult(null);
      setPreviewUrl(null);
      setStreamingUrl(null);
      setRealtimeStreamUrl(null);
      setCurrentSessionId(null);
      setDownloadAvailable(false);
      setUploadedFile(null);
      setIsProcessing(false);
      
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      console.log('All processes terminated and results cleared');
    } catch (error) {
      console.error('Error during cleanup:', error);
      // Still clear the UI state even if some cleanup fails
      setPredictionResult(null);
      setPreviewUrl(null);
      setStreamingUrl(null);
      setRealtimeStreamUrl(null);
      setCurrentSessionId(null);
      setDownloadAvailable(false);
      setUploadedFile(null);
      setIsProcessing(false);
      
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handlePredict = async () => {
    if (inputMode === 'upload' && uploadedFile) {
      setIsProcessing(true);
      
      try {
        const source = uploadedFile.type.startsWith('video/') ? 'video' as const : 'image' as const;
        
        if (source === 'video') {
          // Use real-time video processing for videos
          const result = await api.processVideoRealtime(selectedModel, uploadedFile);
          setRealtimeStreamUrl(result.streamUrl);
          setCurrentSessionId(result.sessionId);
          setDownloadAvailable(false);
          setStreamingUrl(null);
          setPreviewUrl(null);
          setPredictionResult({
            type: 'video',
            message: 'Real-time video processing active',
            model: selectedModel
          });
          
          // Start checking for processing completion
          checkProcessingStatus(result.sessionId);
        } else {
          // Use regular prediction for images
          const data = await api.predict(selectedModel, source, uploadedFile);
          setPredictionResult(data);
          setPreviewUrl(`data:image/jpeg;base64,${data.image_base64}`);
          setStreamingUrl(null);
          setRealtimeStreamUrl(null);
          setCurrentSessionId(null);
          setDownloadAvailable(false);
        }
      } catch (err) {
        console.error(err);
        alert('Error during prediction.');
      }
      setIsProcessing(false);
    } else if (inputMode === 'camera' && isStreamActive) {
      // For camera mode, we're already streaming with detection
      // The analysis is happening in real-time
      setPredictionResult({
        type: 'camera',
        isStreamActive: true,
        model: selectedModel,
        message: 'Real-time detection is active'
      });
    }
  };
  
  const getRiskLevel = (confidence: number, isDrowning: boolean) => {
    if (!isDrowning) return { level: 'Safe', color: 'text-green-600 bg-green-100', icon: CheckCircle };
    if (confidence > 0.9) return { level: 'Critical', color: 'text-red-600 bg-red-100', icon: AlertTriangle };
    if (confidence > 0.8) return { level: 'High Risk', color: 'text-orange-600 bg-orange-100', icon: AlertTriangle };
    return { level: 'Moderate Risk', color: 'text-yellow-600 bg-yellow-100', icon: AlertTriangle };
  };

  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2 flex items-center">
          <Brain className="w-8 h-8 mr-3" />
          AI Drowning Detection Testing
        </h2>
        <p className="text-purple-100">Test and validate drowning detection models with your own video and image data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Panel - Controls */}
        <div className="space-y-6">
          {/* Model Selection */}
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2" />
              Select Detection Model
            </h3>
            
            <div className="space-y-4">
              {models.map((model) => (
                <label key={model.id} className="block cursor-pointer">
                  <div className={`p-4 border-2 rounded-lg transition-all ${
                    selectedModel === model.id 
                      ? 'border-purple-500 bg-purple-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}>
                    <div className="flex items-start space-x-3">
                      <input
                        type="radio"
                        name="model"
                        value={model.id}
                        checked={selectedModel === model.id}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="mt-1"
                      />
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-900">{model.name}</h4>
                        <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                          <span>Accuracy: {model.accuracy}</span>
                          <span>Speed: {model.processingTime}</span>
                        </div>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {model.features.map((feature, index) => (
                            <span key={index} className="px-2 py-1 bg-gray-100 text-xs rounded">
                              {feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* File Upload */}
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              {inputMode === 'upload' ? <Upload className="w-5 h-5 mr-2" /> : <Camera className="w-5 h-5 mr-2" />}
              {inputMode === 'upload' ? 'Upload Media' : 'Live Camera Feed'}
            </h3>
            
            {/* Input Mode Selection */}
            <div className="flex space-x-4 mb-6">
              <button
                onClick={() => {
                  setInputMode('upload');
                  setUploadedFile(null);
                  setPreviewUrl(null);
                  setStreamingUrl(null);
                  setPredictionResult(null);
                }}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  inputMode === 'upload'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Upload className="w-4 h-4" />
                <span>Upload Media</span>
              </button>
              <button
                onClick={() => {
                  setInputMode('camera');
                  setUploadedFile(null);
                  setPreviewUrl(null);
                  setStreamingUrl(null);
                  setPredictionResult(null);
                }}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  inputMode === 'camera'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Camera className="w-4 h-4" />
                <span>Live Camera</span>
              </button>
            </div>

            <div className="space-y-4">
              {inputMode === 'upload' ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-purple-500 hover:bg-purple-50 transition-colors"
                >
                  <div className="flex flex-col items-center space-y-3">
                    <div className="flex space-x-2">
                      <FileVideo className="w-8 h-8 text-gray-400" />
                      <ImageIcon className="w-8 h-8 text-gray-400" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-gray-900">Upload Video or Image</p>
                      <p className="text-sm text-gray-500">Drag and drop or click to browse</p>
                      <p className="text-xs text-gray-400 mt-1">Supports MP4, AVI, MOV, JPG, PNG</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-gray-100 rounded-lg p-4 text-center space-y-3">
                    <Camera className="w-12 h-12 text-gray-400 mx-auto" />
                    <p className="text-sm text-gray-600">
                      {isStreamActive ? 'Camera is active and ready for analysis' : 'Click to access your camera'}
                    </p>
                    {webcamStatus && (
                      <div className="text-xs text-gray-500">
                        Status: {webcamStatus.active ? 'Active' : 'Inactive'}
                        {webcamStatus.model && ` | Model: ${webcamStatus.model}`}
                        <button
                          onClick={checkWebcamStatus}
                          className="ml-2 text-blue-500 hover:text-blue-700 underline"
                        >
                          Refresh
                        </button>
                      </div>
                    )}
                    {!isStreamActive ? (
                      <button
                        onClick={startCamera}
                        disabled={isProcessing}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg text-sm font-medium transition-colors"
                      >
                        {isProcessing ? 'Starting...' : 'Start Camera'}
                      </button>
                    ) : (
                      <button
                        onClick={stopCamera}
                        disabled={isProcessing}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-lg text-sm font-medium transition-colors"
                      >
                        {isProcessing ? 'Stopping...' : 'Stop Camera'}
                      </button>
                    )}
                  </div>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*,image/*"
                onChange={handleFileUpload}
                className="hidden"
              />

              {uploadedFile && (
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
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
                </div>
              )}

              <button
                onClick={handlePredict}
                disabled={(inputMode === 'upload' && !uploadedFile) || (inputMode === 'camera' && !isStreamActive) || isProcessing}
                className="w-full flex items-center justify-center space-x-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white py-3 px-4 rounded-lg font-semibold transition-colors"
              >
                {isProcessing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>{inputMode === 'upload' ? 'Run Prediction' : 'Analyze Live Feed'}</span>
                  </>
                )}
              </button>

              {inputMode === 'upload' && predictionResult && (
                <button
                   onClick={handleClearResults}
                   className="w-full mt-2 flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white py-3 px-4 rounded-lg font-semibold transition-colors"
                 >
                   <Square className="w-5 h-5" />
                   <span>Start Over</span>
                 </button>
              )}


            </div>
          </div>

          {/* Model Info */}
          {selectedModelData && (
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Model Information</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Accuracy:</span>
                  <span className="font-semibold">{selectedModelData.accuracy}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Processing Time:</span>
                  <span className="font-semibold">{selectedModelData.processingTime}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Model Type:</span>
                  <span className="font-semibold">Deep Learning YOLOv8l</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Training Data:</span>
                  <span className="font-semibold">{selectedModelData.trainingData}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Preview and Results */}
        <div className="space-y-6">
          {/* Media Preview */}
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold mb-4">Media Preview</h3>
            
            <div className="w-full h-96 bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
                              {inputMode === 'camera' && isStreamActive ? (
                  <img
                    src="http://127.0.0.1:8000/video_feed"
                    alt="Live Camera Feed"
                    className="max-w-full max-h-full object-contain"
                  />
                ) : inputMode === 'camera' ? (
                  <div className="text-center text-gray-500">
                    <Camera className="w-16 h-16 mx-auto mb-2" />
                    <p>Camera not started</p>
                  </div>
                ) : realtimeStreamUrl ? (
                  <img
                    src={realtimeStreamUrl}
                    alt="Real-time Video Processing"
                    className="max-w-full max-h-full object-contain"
                  />
                ) : streamingUrl ? (
                  <img
                    src={streamingUrl}
                    alt="Processed Video Stream"
                    className="max-w-full max-h-full object-contain"
                  />
                ) : previewUrl ? (
                  uploadedFile?.type.startsWith('video/') ? (
                    <video
                      src={previewUrl}
                      controls
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <img
                      src={previewUrl}
                      alt="Prediction Result"
                      className="max-w-full max-h-full object-contain"
                    />
                  )
                ) : (
                  // Fallback when no preview
                  <div className="text-center text-gray-400">
                    <ImageIcon className="w-16 h-16 mx-auto mb-2" />
                    <p>No media uploaded</p>
                  </div>
                )}              
            </div>
          </div>

          {/* Prediction Results */}
          {predictionResult && (
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
              
              <div className="space-y-4">
                {/* Risk Assessment */}
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold">Risk Assessment</h4>
                    <div className="flex items-center space-x-2">
                      <Clock className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-500">
                        {predictionResult.analysis?.processingTime || 'Real-time'}
                      </span>
                    </div>
                  </div>
                  
                  {(() => {
                    const analysis = predictionResult.analysis;
                    if (!analysis) {
                      return (
                        <div className="flex items-center space-x-3 p-3 rounded-lg bg-blue-100 text-blue-600">
                          <Monitor className="w-6 h-6" />
                          <div>
                            <div className="font-semibold">Live Detection Active</div>
                            <div className="text-sm">Real-time analysis running</div>
                          </div>
                        </div>
                      );
                    }
                    
                    const risk = getRiskLevel(analysis.confidence, analysis.isDrowning);
                    const RiskIcon = risk.icon;
                    return (
                      <div className={`flex items-center space-x-3 p-3 rounded-lg ${risk.color}`}>
                        <RiskIcon className="w-6 h-6" />
                        <div>
                          <div className="font-semibold">{analysis.riskLevel || risk.level}</div>
                          <div className="text-sm">
                            Confidence: {(analysis.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>

                {/* Recommendations */}
                {predictionResult.analysis?.recommendations && (
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-semibold mb-3">Recommendations</h4>
                    <div className="space-y-1">
                      {predictionResult.analysis.recommendations.map((rec: string, index: number) => (
                        <div key={index} className="flex items-center space-x-2 text-sm">
                          <CheckCircle className="w-4 h-4 text-blue-500" />
                          <span>{rec}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Detected Objects */}
                {predictionResult.analysis?.detectedObjects && predictionResult.analysis.detectedObjects.length > 0 && (
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-semibold mb-3">Detected Objects</h4>
                    <div className="space-y-2">
                      {predictionResult.analysis.detectedObjects.map((obj: any, index: number) => (
                        <div key={index} className="flex items-center justify-between text-sm">
                          <span className="capitalize">{obj.type.replace('_', ' ')}</span>
                          <span className="font-medium">{(obj.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Download Button for Processed Videos */}
                {downloadAvailable && (
                  <div className="mt-4 pt-4 border-t">
                    <button
                      onClick={handleDownloadVideo}
                      className="w-full flex items-center justify-center space-x-2 bg-green-600 hover:bg-green-700 text-white py-3 px-4 rounded-lg font-semibold transition-colors"
                    >
                      <FileVideo className="w-5 h-5" />
                      <span>Download Processed Video</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}