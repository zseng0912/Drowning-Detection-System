// React component for displaying individual camera feeds in the pool surveillance system
// Supports both above-water and underwater camera views with sound detection
import { useState, useRef, useEffect } from 'react';
import { Maximize2, AlertTriangle, Eye, EyeOff } from 'lucide-react';
import { useSoundDetection } from '../hooks/useSoundDetection';

interface CameraFeedProps {
  zone: string;
  type: 'above' | 'underwater';
  isActive: boolean;
  hasAlert?: boolean;
  onFullscreen: () => void;
  onSoundDetected?: (detected: boolean, zone: string, type: 'above' | 'underwater', audioLevel: number) => void;
}

export default function CameraFeed({ 
  zone, 
  type, 
  isActive, 
  hasAlert, 
  onFullscreen, 
  onSoundDetected 
}: CameraFeedProps) {
  // State management for camera feed visibility and video element tracking
  const [isVisible, setIsVisible] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentVideo, setCurrentVideo] = useState<HTMLVideoElement | null>(null);
  const { isSoundDetected, audioLevel } = useSoundDetection(currentVideo);

  // Update current video reference for sound detection when visibility changes
  useEffect(() => {
    if (isVisible) {
      setCurrentVideo(videoRef.current);
    } else {
      // Set to null when video is hidden to reset sound detection
      setCurrentVideo(null);
    }
  }, [videoRef.current, isVisible]);

  // Handle sound detection events and notify parent component
  useEffect(() => {
    if (onSoundDetected && currentVideo) {
      // Only call onSoundDetected when we have a video element
      onSoundDetected(isSoundDetected, zone, type, audioLevel);
    } else if (onSoundDetected && !currentVideo) {
      // When video is hidden, explicitly clear the alert
      onSoundDetected(false, zone, type, 0);
    }
  }, [isSoundDetected, audioLevel, zone, type, onSoundDetected, currentVideo]);

  // Control video playback based on surveillance system status
  useEffect(() => {
    if (videoRef.current) {
      if (isActive) {
        videoRef.current.play().catch(console.error);
      } else {
        videoRef.current.pause();
      }
    }
  }, [isActive]);

  // Main camera feed container with alert styling and status indicators
  return (
    <div className={`relative bg-gray-900 rounded-lg overflow-hidden border-2 transition-all duration-300 ${
      hasAlert ? 'border-red-500 animate-pulse' : 'border-gray-700'
    } ${isActive ? 'ring-2 ring-blue-500' : ''}`}>
      {/* Header Section - Zone info and controls */}
      <div className={`flex items-center justify-between p-3 ${
        type === 'above' ? 'bg-blue-600' : 'bg-teal-600'
      }`}>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
          <span className="text-white font-semibold">{zone} - {type === 'above' ? 'Above Water' : 'Underwater'}</span>
        </div>
        <div className="flex items-center space-x-2">
          {hasAlert && <AlertTriangle className="w-5 h-5 text-red-400" />}
          <button
            onClick={() => setIsVisible(!isVisible)}
            className="text-white hover:text-gray-200 transition-colors"
          >
            {isVisible ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Video Feed Container - Main video display area */}
      {isVisible && (
        <div className="relative aspect-video bg-gray-800">
          {/* System Inactive Overlay - Displayed when monitoring is disabled */}
          {!isActive && (
            <div className="absolute inset-0 bg-black bg-opacity-80 flex items-center justify-center z-20">
              <div className="text-center">
                <div className="text-red-400 text-2xl font-bold mb-2">SYSTEM INACTIVE</div>
                <div className="text-gray-300 text-sm">Monitoring Disabled</div>
                <div className="text-gray-400 text-xs mt-2">Start system to resume monitoring</div>
              </div>
            </div>
          )}
          {/* Video Element - Actual camera feed or placeholder */}
          {zone === 'Zone A' ? (
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              loop
              playsInline
              controls
              onError={(e) => {
                console.error('Video error:', e);
                console.error('Video src:', e.currentTarget.src);
              }}
              onLoadedData={() => {
                console.log('Video loaded successfully');
                setCurrentVideo(videoRef.current);
              }}
            >
              <source 
                 src={type === 'above' ? '/demo_video/ZoneA_Abovewater.mp4' : '/demo_video/ZoneA_Underwater.mp4'} 
                 type="video/mp4" 
               />
              Your browser does not support the video tag.
            </video>
          ) : zone === 'Zone B' ? (
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              loop
              playsInline
              controls
              onError={(e) => {
                console.error('Video error:', e);
                console.error('Video src:', e.currentTarget.src);
              }}
              onLoadedData={() => {
                console.log('Video loaded successfully');
                setCurrentVideo(videoRef.current);
              }}
            >
              <source 
                 src={type === 'above' ? '/demo_video/ZoneB_Abovewater.mp4' : '/demo_video/ZoneB_Underwater.mp4'} 
                 type="video/mp4" 
               />
              Your browser does not support the video tag.
            </video>
          ):(
            /* Placeholder Display - For zones without video feeds */
            <div className={`w-full h-full ${
              type === 'above' 
                ? 'bg-gradient-to-b from-blue-200 to-blue-400' 
                : 'bg-gradient-to-b from-teal-300 to-teal-600'
            } flex items-center justify-center`}>
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-2">{zone}</div>
                <div className="text-sm text-white opacity-75">
                  {type === 'above' ? 'Pool Surface View' : 'Underwater View'}
                </div>
                <div className="text-xs text-white opacity-50 mt-2">
                  Video feed not available
                </div>
              </div>
            </div>
          )}
          
          {/* Timestamp Overlay - Current time display */}
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
            {new Date().toLocaleTimeString()}
          </div>

          {/* Controls Overlay - Hover controls for fullscreen */}
          <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center opacity-0 hover:opacity-100">
            <div className="flex space-x-3">
              <button
                onClick={onFullscreen}
                className="p-3 bg-gray-800 hover:bg-gray-700 text-white rounded-full transition-colors"
              >
                <Maximize2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status Bar - Camera feed status and quality info */}
      <div className="flex items-center justify-between p-2 bg-gray-800 text-xs text-gray-300">
        <span>Status: {isActive ? 'Active' : 'Offline'}</span>
        <span>Quality: HD</span>
      </div>
    </div>
  );
}