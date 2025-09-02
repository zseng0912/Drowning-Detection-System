import React, { useState } from 'react';
import { Maximize2, Play, Square, AlertTriangle, Eye, EyeOff } from 'lucide-react';

interface CameraFeedProps {
  zone: string;
  type: 'above' | 'underwater';
  isActive: boolean;
  hasAlert?: boolean;
  onFullscreen: () => void;
  onToggleRecording: () => void;
  isRecording: boolean;
}

export default function CameraFeed({ 
  zone, 
  type, 
  isActive, 
  hasAlert, 
  onFullscreen, 
  onToggleRecording, 
  isRecording 
}: CameraFeedProps) {
  const [isVisible, setIsVisible] = useState(true);

  return (
    <div className={`relative bg-gray-900 rounded-lg overflow-hidden border-2 transition-all duration-300 ${
      hasAlert ? 'border-red-500 animate-pulse' : 'border-gray-700'
    } ${isActive ? 'ring-2 ring-blue-500' : ''}`}>
      {/* Header */}
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

      {/* Video Feed */}
      {isVisible && (
        <div className="relative aspect-video bg-gray-800">
          {/* Simulated camera feed */}
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
              {/* Simulated swimmer detection */}
              {zone === 'Zone A' && type === 'above' && (
                <div className="absolute top-16 left-12 w-8 h-8 bg-yellow-400 rounded-full animate-bounce border-2 border-white">
                  <div className="text-xs text-black font-bold text-center leading-8">S</div>
                </div>
              )}
            </div>
          </div>

          {/* Recording indicator */}
          {isRecording && (
            <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-600 px-2 py-1 rounded">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span className="text-white text-xs font-semibold">REC</span>
            </div>
          )}

          {/* Timestamp */}
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
            {new Date().toLocaleTimeString()}
          </div>

          {/* Controls overlay */}
          <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center opacity-0 hover:opacity-100">
            <div className="flex space-x-3">
              <button
                onClick={onToggleRecording}
                className={`p-3 rounded-full ${
                  isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-gray-800 hover:bg-gray-700'
                } text-white transition-colors`}
              >
                {isRecording ? <Square className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
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

      {/* Status bar */}
      <div className="flex items-center justify-between p-2 bg-gray-800 text-xs text-gray-300">
        <span>Status: {isActive ? 'Active' : 'Offline'}</span>
        <span>Quality: HD</span>
      </div>
    </div>
  );
}