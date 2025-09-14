// Main surveillance dashboard component for pool monitoring system
// Manages camera feeds, alerts, system controls, and emergency procedures
import { useState, useEffect, useCallback } from 'react';
import CameraFeed from './CameraFeed';
import AlertPanel from './AlertPanel';
import ControlPanel from './ControlPanel';
import StatusBar from './StatusBar';
import ConfirmationDialog from './ConfirmationDialog';

interface Alert {
  id: string;
  type: 'warning' | 'danger' | 'info';
  zone: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}

interface SurveillanceTabProps {
  isSystemActive: boolean;
  setIsSystemActive: (active: boolean) => void;
}

export default function SurveillanceTab({ isSystemActive, setIsSystemActive }: SurveillanceTabProps) {
  // State management for surveillance system
  const [soundAlerts, setSoundAlerts] = useState<{[key: string]: boolean}>({});
  const [showEmergencyDialog, setShowEmergencyDialog] = useState(false);
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      type: 'info',
      zone: 'Zone A - Above Water',
      message: 'Routine system check completed',
      timestamp: new Date(),
      acknowledged: true,
    },
    {
      id: '2',
      type: 'info',
      zone: 'Zone B - Underwater',
      message: 'Routine system check completed',
      timestamp: new Date(Date.now() - 300000),
      acknowledged: true,
    },
  ]);

  // System statistics and performance monitoring
  const [systemStats, setSystemStats] = useState({
    networkStatus: 'online' as const,
    storageUsed: 67,
    cpuUsage: 34,
    temperature: 42,
    activeCameras: 4,
    totalCameras: 4,
  });

  // Real-time system statistics simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemStats(prev => ({
        ...prev,
        cpuUsage: Math.floor(Math.random() * 20) + 30,
        temperature: Math.floor(Math.random() * 10) + 38,
        storageUsed: Math.min(100, prev.storageUsed + Math.random() - 0.4),
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // System overview metrics for dashboard display
  const systemOverview = {
    totalZones: 4, // We have 4 camera feeds (Zone A above, Zone A underwater, Zone B above, Zone B underwater)
    activeCameras: isSystemActive ? 4 : 0, // All cameras are active when system is active
    inactiveCameras: isSystemActive ? 0 : 4, // All cameras are inactive when system is inactive
    activeAlerts: alerts.filter(alert => !alert.acknowledged).length // Count unacknowledged alerts
  };



  // Handle fullscreen video display for camera feeds
  const handleFullscreen = (zone: string) => {
    console.log(`Opening fullscreen view for ${zone}`);
    
    // Find the video element for the specific zone
    const videoElements = document.querySelectorAll('video');
    let targetVideo: HTMLVideoElement | null = null;
    
    // Map zone identifiers to actual zones
    const zoneMap: {[key: string]: {zone: string, type: string}} = {
      'above-a': {zone: 'Zone A', type: 'above'},
      'above-b': {zone: 'Zone B', type: 'above'},
      'underwater-a': {zone: 'Zone A', type: 'underwater'},
      'underwater-b': {zone: 'Zone B', type: 'underwater'}
    };
    
    const zoneInfo = zoneMap[zone];
    if (!zoneInfo) return;
    
    // Find the correct video element by checking its source
    videoElements.forEach((video) => {
      const source = video.querySelector('source');
      if (source) {
        const src = source.src;
        const isCorrectZone = src.includes(zoneInfo.zone.replace(' ', ''));
        const isCorrectType = zoneInfo.type === 'above' ? 
          src.includes('Abovewater') : 
          src.includes('Underwater');
        
        if (isCorrectZone && isCorrectType) {
          targetVideo = video;
        }
      }
    });
    
    if (targetVideo) {
      // Request fullscreen
      if (targetVideo.requestFullscreen) {
        targetVideo.requestFullscreen().catch(err => {
          console.error('Error attempting to enable fullscreen:', err);
        });
      } else if ((targetVideo as any).webkitRequestFullscreen) {
        // Safari support
        (targetVideo as any).webkitRequestFullscreen();
      } else if ((targetVideo as any).msRequestFullscreen) {
        // IE/Edge support
        (targetVideo as any).msRequestFullscreen();
      }
    } else {
      console.warn('Could not find video element for zone:', zone);
    }
  };

  // Mark alert as acknowledged by operator
  const handleAcknowledgeAlert = (id: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === id ? { ...alert, acknowledged: true } : alert
      )
    );
  };

  // Remove alert from the system
  const handleDismissAlert = (id: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id));
  };

  // Toggle surveillance system on/off
  const handleToggleSystem = () => {
    setIsSystemActive(!isSystemActive);
  };

  // Initiate emergency stop procedure
  const handleEmergencyStop = () => {
    setShowEmergencyDialog(true);
  };

  // Execute emergency stop with system shutdown sequence
  const confirmEmergencyStop = async () => {
    setIsLoading(true);
    setShowEmergencyDialog(false);
    
    try {
      // Step 1: Stop all camera feeds
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Step 2: Clear all sound alerts
      setSoundAlerts({});
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Step 3: Update system stats to show shutdown
      setSystemStats(prev => ({
        ...prev,
        cpuUsage: 0,
        temperature: 25,
        activeCameras: 0,
      }));
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Step 4: Deactivate system
      setIsSystemActive(false);
      
      // Step 5: Add emergency alert
      const emergencyAlert: Alert = {
        id: Date.now().toString(),
        type: 'danger',
        zone: 'System',
        message: 'Emergency stop activated - All systems halted safely',
        timestamp: new Date(),
        acknowledged: false,
      };
      setAlerts(prev => [emergencyAlert, ...prev]);
      
    } catch (error) {
      console.error('Emergency stop failed:', error);
      const errorAlert: Alert = {
        id: Date.now().toString(),
        type: 'danger',
        zone: 'System',
        message: 'Emergency stop encountered an error - Manual intervention required',
        timestamp: new Date(),
        acknowledged: false,
      };
      setAlerts(prev => [errorAlert, ...prev]);
    } finally {
      setIsLoading(false);
    }
  };

  // Initiate system reset procedure
  const handleResetSystem = () => {
    setShowResetDialog(true);
  };

  // Execute system reset with initialization sequence
  const confirmResetSystem = async () => {
    setIsLoading(true);
    setShowResetDialog(false);
    
    try {
      // Step 1: Clear all alerts and sound alerts
      setAlerts([]);
      setSoundAlerts({});
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Step 2: Reset system statistics
      setSystemStats(prev => ({
        ...prev,
        cpuUsage: 15,
        temperature: 35,
        activeCameras: 0,
      }));
      await new Promise(resolve => setTimeout(resolve, 400));
      
      // Step 3: Initialize cameras
      setSystemStats(prev => ({
        ...prev,
        activeCameras: 2,
        cpuUsage: 20,
      }));
      await new Promise(resolve => setTimeout(resolve, 400));
      
      // Step 4: Complete camera initialization
      setSystemStats(prev => ({
        ...prev,
        activeCameras: 4,
        cpuUsage: 25,
        temperature: 38,
      }));
      await new Promise(resolve => setTimeout(resolve, 400));
      
      // Step 5: Activate system
      setIsSystemActive(true);
      
      // Step 6: Add system reset confirmation alert
      const resetAlert: Alert = {
        id: Date.now().toString(),
        type: 'info',
        zone: 'System',
        message: 'System reset completed successfully - All cameras online',
        timestamp: new Date(),
        acknowledged: false,
      };
      setAlerts(prev => [resetAlert, ...prev]);
      
    } catch (error) {
      console.error('System reset failed:', error);
      const errorAlert: Alert = {
        id: Date.now().toString(),
        type: 'danger',
        zone: 'System',
        message: 'System reset failed - Please check system status',
        timestamp: new Date(),
        acknowledged: false,
      };
      setAlerts(prev => [errorAlert, ...prev]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle drowning detection alerts from camera feeds
  const handleSoundDetected = useCallback((detected: boolean, zone: string, type: 'above' | 'underwater', audioLevel: number) => {
    const alertKey = `${zone}-${type}`;
    const alertId = `sound-${alertKey}-${Date.now()}`;
    
    if (detected && !soundAlerts[alertKey]) {
      // Sound detected - add alert
      setSoundAlerts(prev => ({ ...prev, [alertKey]: true }));
      
      const soundAlert: Alert = {
        id: alertId,
        type: 'warning',
        zone: `${zone} - ${type === 'above' ? 'Above Water' : 'Underwater'}`,
        message: `Drowning Detected !!!`,
        timestamp: new Date(),
        acknowledged: false,
      };
      
      setAlerts(prev => {
        // Remove any existing sound alerts for this camera
        const filteredAlerts = prev.filter(alert => !alert.id.startsWith(`sound-${alertKey}`));
        return [soundAlert, ...filteredAlerts];
      });
    } else if (!detected) {
      // Sound stopped or camera hidden - always remove alert regardless of previous state
      setSoundAlerts(prev => {
        const newState = { ...prev };
        delete newState[alertKey];
        return newState;
      });
      
      setAlerts(prev => prev.filter(alert => !alert.id.startsWith(`sound-${alertKey}`)));
    }
  }, []);

  // Main surveillance dashboard layout
  return (
    <div className="flex flex-col min-h-screen">
      {/* Main Content - Dashboard grid layout */}
      <div className="flex-1 p-6 space-y-6">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Camera Feeds Section - Multiple zone monitoring */}
          <div className="xl:col-span-2 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <CameraFeed
                zone="Zone A"
                type="above"
                isActive={isSystemActive}
                hasAlert={alerts.some(a => a.zone.includes('Zone A') && a.type === 'warning')}
                onFullscreen={() => handleFullscreen('above-a')}
                onSoundDetected={handleSoundDetected}
              />
              <CameraFeed
                zone="Zone B"
                type="above"
                isActive={isSystemActive}
                onFullscreen={() => handleFullscreen('above-b')}
                onSoundDetected={handleSoundDetected}
              />
              <CameraFeed
                zone="Zone A"
                type="underwater"
                isActive={isSystemActive}
                onFullscreen={() => handleFullscreen('underwater-a')}
                onSoundDetected={handleSoundDetected}
              />
              <CameraFeed
                zone="Zone B"
                type="underwater"
                isActive={isSystemActive}
                hasAlert={alerts.some(a => a.zone.includes('Zone B') && !a.acknowledged)}
                onFullscreen={() => handleFullscreen('underwater-b')}
                onSoundDetected={handleSoundDetected}
              />
            </div>
          </div>

          {/* Side Panel - Controls and alerts */}
          <div className="xl:col-span-1 space-y-6">
            <ControlPanel
              isSystemActive={isSystemActive}
              onToggleSystem={handleToggleSystem}
              onEmergencyStop={handleEmergencyStop}
              onResetSystem={handleResetSystem}
              systemOverview={systemOverview}
              isLoading={isLoading}
            />
            <AlertPanel
              alerts={alerts}
              onAcknowledge={handleAcknowledgeAlert}
              onDismiss={handleDismissAlert}
            />
          </div>
        </div>
      </div>

      {/* Status Bar - System performance indicators */}
      <StatusBar stats={systemStats} />
      
      {/* Confirmation Dialogs - Emergency and reset confirmations */}
      <ConfirmationDialog
        isOpen={showEmergencyDialog}
        title="Emergency Stop"
        message="Are you sure you want to activate emergency stop? This will immediately halt all system operations and cannot be undone."
        confirmText="Emergency Stop"
        cancelText="Cancel"
        onConfirm={confirmEmergencyStop}
        onCancel={() => setShowEmergencyDialog(false)}
        type="danger"
      />
      
      <ConfirmationDialog
        isOpen={showResetDialog}
        title="Reset System"
        message="Are you sure you want to reset the system? This will clear all alerts, restart all cameras, and reset system statistics."
        confirmText="Reset System"
        cancelText="Cancel"
        onConfirm={confirmResetSystem}
        onCancel={() => setShowResetDialog(false)}
        type="warning"
      />
    </div>
  );
}