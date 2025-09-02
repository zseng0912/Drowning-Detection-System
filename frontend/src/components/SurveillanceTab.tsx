import React, { useState, useEffect } from 'react';
import CameraFeed from './CameraFeed';
import AlertPanel from './AlertPanel';
import ControlPanel from './ControlPanel';
import StatusBar from './StatusBar';

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
  const [recordingStates, setRecordingStates] = useState({
    'above-a': false,
    'above-b': false,
    'underwater-a': false,
    'underwater-b': false,
  });
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      type: 'warning',
      zone: 'Zone A - Above Water',
      message: 'Motion detected in restricted area',
      timestamp: new Date(),
      acknowledged: false,
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

  const [systemStats, setSystemStats] = useState({
    networkStatus: 'online' as const,
    storageUsed: 67,
    cpuUsage: 34,
    temperature: 42,
    activeCameras: 4,
    totalCameras: 4,
  });

  // Simulate system stats updates
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

  const handleToggleRecording = (zone: string) => {
    setRecordingStates(prev => ({
      ...prev,
      [zone]: !prev[zone as keyof typeof prev],
    }));
  };

  const handleFullscreen = (zone: string) => {
    console.log(`Opening fullscreen view for ${zone}`);
    // Implement fullscreen functionality
  };

  const handleAcknowledgeAlert = (id: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === id ? { ...alert, acknowledged: true } : alert
      )
    );
  };

  const handleDismissAlert = (id: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id));
  };

  const handleToggleSystem = () => {
    setIsSystemActive(!isSystemActive);
  };

  const handleEmergencyStop = () => {
    setIsSystemActive(false);
    setRecordingStates({
      'above-a': false,
      'above-b': false,
      'underwater-a': false,
      'underwater-b': false,
    });
    // Add emergency alert
    const emergencyAlert: Alert = {
      id: Date.now().toString(),
      type: 'danger',
      zone: 'System',
      message: 'Emergency stop activated - All systems halted',
      timestamp: new Date(),
      acknowledged: false,
    };
    setAlerts(prev => [emergencyAlert, ...prev]);
  };

  const handleResetSystem = () => {
    setIsSystemActive(true);
    setAlerts([]);
    setSystemStats(prev => ({
      ...prev,
      cpuUsage: 25,
      temperature: 38,
    }));
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Camera Feeds */}
          <div className="xl:col-span-2 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <CameraFeed
                zone="Zone A"
                type="above"
                isActive={isSystemActive}
                hasAlert={alerts.some(a => a.zone.includes('Zone A') && a.type === 'warning')}
                onFullscreen={() => handleFullscreen('above-a')}
                onToggleRecording={() => handleToggleRecording('above-a')}
                isRecording={recordingStates['above-a']}
              />
              <CameraFeed
                zone="Zone B"
                type="above"
                isActive={isSystemActive}
                onFullscreen={() => handleFullscreen('above-b')}
                onToggleRecording={() => handleToggleRecording('above-b')}
                isRecording={recordingStates['above-b']}
              />
              <CameraFeed
                zone="Zone A"
                type="underwater"
                isActive={isSystemActive}
                onFullscreen={() => handleFullscreen('underwater-a')}
                onToggleRecording={() => handleToggleRecording('underwater-a')}
                isRecording={recordingStates['underwater-a']}
              />
              <CameraFeed
                zone="Zone B"
                type="underwater"
                isActive={isSystemActive}
                hasAlert={alerts.some(a => a.zone.includes('Zone B') && !a.acknowledged)}
                onFullscreen={() => handleFullscreen('underwater-b')}
                onToggleRecording={() => handleToggleRecording('underwater-b')}
                isRecording={recordingStates['underwater-b']}
              />
            </div>
          </div>

          {/* Side Panel */}
          <div className="xl:col-span-1 space-y-6">
            <ControlPanel
              isSystemActive={isSystemActive}
              onToggleSystem={handleToggleSystem}
              onEmergencyStop={handleEmergencyStop}
              onResetSystem={handleResetSystem}
            />
            <AlertPanel
              alerts={alerts}
              onAcknowledge={handleAcknowledgeAlert}
              onDismiss={handleDismissAlert}
            />
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar stats={systemStats} />
    </div>
  );
}