// Custom React hook for real-time audio level detection and sound monitoring from video elements
import { useEffect, useRef, useState } from 'react';

// Configuration options for sound detection sensitivity and audio analysis
interface SoundDetectionOptions {
  threshold?: number; // Audio level threshold (0-1)
  smoothingTimeConstant?: number; // Smoothing for audio analysis
  fftSize?: number; // FFT size for frequency analysis
}

// Main hook function that analyzes audio from video element and returns sound detection state
export const useSoundDetection = (
  videoElement: HTMLVideoElement | null,
  options: SoundDetectionOptions = {}
) => {
  // Extract options with default values for audio analysis configuration
  const {
    threshold = 0.01, // Default threshold for sound detection
    smoothingTimeConstant = 0.8,
    fftSize = 256
  } = options;

  // State variables for sound detection results
  const [isSoundDetected, setIsSoundDetected] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  // Refs for Web Audio API components
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Main effect hook that sets up audio analysis when video element changes
  useEffect(() => {
    if (!videoElement) {
      // Reset sound detection state when video element is null (e.g., when camera is hidden)
      setIsSoundDetected(false);
      setAudioLevel(0);
      return;
    }

    // Initialize Web Audio API components for audio analysis
    const initializeAudioContext = async () => {
      try {
        // Only create audio context if it doesn't exist
        if (!audioContextRef.current) {
          audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        
        // Only create analyser if it doesn't exist
        if (!analyserRef.current) {
          analyserRef.current = audioContextRef.current.createAnalyser();
          analyserRef.current.fftSize = fftSize;
          analyserRef.current.smoothingTimeConstant = smoothingTimeConstant;
        }
        
        // Only create source if it doesn't exist (MediaElementAudioSourceNode can only be created once per element)
        if (!sourceRef.current) {
          sourceRef.current = audioContextRef.current.createMediaElementSource(videoElement);
          
          // Connect nodes
          sourceRef.current.connect(analyserRef.current);
          sourceRef.current.connect(audioContextRef.current.destination);
        }
        
        // Start analyzing
        analyzeAudio();
      } catch (error) {
        console.error('Error initializing audio context:', error);
      }
    };

    // Start continuous audio analysis using requestAnimationFrame
    const analyzeAudio = () => {
      if (!analyserRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      // Recursive function for real-time audio level calculation
      const analyze = () => {
        if (!analyserRef.current) return;
        
        analyserRef.current.getByteFrequencyData(dataArray);
        
        // Calculate average audio level
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const average = sum / bufferLength / 255; // Normalize to 0-1
        
        setAudioLevel(average);
        setIsSoundDetected(average > threshold);
        
        animationFrameRef.current = requestAnimationFrame(analyze);
      };
      
      analyze();
    };

    // Event handlers for video lifecycle management
    const handleCanPlay = () => {
      initializeAudioContext();
    };

    // Handle video play events and ensure audio context is active
    const handlePlay = () => {
      // Ensure audio context is running when video plays/replays
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
      // Always restart audio analysis on play to ensure it works during auto-replay
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      analyzeAudio();
    };

    // Handle video end events for looped playback
    const handleEnded = () => {
      // Video ended, but will auto-replay due to loop attribute
      // Keep audio context active for seamless transition
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
    };

    if (videoElement.readyState >= 3) {
      // Video is already ready
      initializeAudioContext();
    } else {
      videoElement.addEventListener('canplay', handleCanPlay);
    }

    // Add event listeners for video playback events
    videoElement.addEventListener('play', handlePlay);
    videoElement.addEventListener('ended', handleEnded);

    // Cleanup function to prevent memory leaks and remove event listeners
    return () => {
      // Cancel ongoing animation frame
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      // Close audio context
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      // Remove event listeners
      videoElement.removeEventListener('canplay', handleCanPlay);
      videoElement.removeEventListener('play', handlePlay);
      videoElement.removeEventListener('ended', handleEnded);
    };
  }, [videoElement, threshold, smoothingTimeConstant, fftSize]);

  // Return sound detection state and current audio level
  return {
    isSoundDetected,
    audioLevel
  };
};