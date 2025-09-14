// System status bar component displaying real-time system metrics and camera status
import { Wifi, HardDrive, Cpu, Thermometer } from 'lucide-react';

// Interface defining system statistics structure
interface SystemStats {
  networkStatus: 'online' | 'offline';
  storageUsed: number;
  cpuUsage: number;
  temperature: number;
  activeCameras: 4;
  totalCameras: number;
}

// Props interface for the StatusBar component
interface StatusBarProps {
  stats: SystemStats;
}

// Main StatusBar component for displaying system health and operational metrics
export default function StatusBar({ stats }: StatusBarProps) {
  // Function to determine status color based on value thresholds
  const getStatusColor = (value: number, thresholds: { warning: number; danger: number }) => {
    if (value >= thresholds.danger) return 'text-red-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    // Main status bar container
    <div className="bg-white border-t border-gray-200 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left section: System metrics */}
        <div className="flex items-center space-x-8">
          {/* Network connectivity status indicator */}
          <div className="flex items-center space-x-2">
            <Wifi className={`w-4 h-4 ${
              stats.networkStatus === 'online' ? 'text-green-600' : 'text-red-600'
            }`} />
            <span className="text-sm text-gray-700">
              Network: {stats.networkStatus}
            </span>
          </div>

          {/* Storage usage percentage with color-coded status */}
          <div className="flex items-center space-x-2">
            <HardDrive className={`w-4 h-4 ${
              getStatusColor(stats.storageUsed, { warning: 80, danger: 90 })
            }`} />
            <span className="text-sm text-gray-700">
              Storage: {stats.storageUsed}%
            </span>
          </div>

          {/* CPU usage percentage with performance indicators */}
          <div className="flex items-center space-x-2">
            <Cpu className={`w-4 h-4 ${
              getStatusColor(stats.cpuUsage, { warning: 70, danger: 85 })
            }`} />
            <span className="text-sm text-gray-700">
              CPU: {stats.cpuUsage}%
            </span>
          </div>

          {/* System temperature monitoring with thermal alerts */}
          <div className="flex items-center space-x-2">
            <Thermometer className={`w-4 h-4 ${
              getStatusColor(stats.temperature, { warning: 60, danger: 70 })
            }`} />
            <span className="text-sm text-gray-700">
              Temp: {stats.temperature}°C
            </span>
          </div>
        </div>

        {/* Right section: Camera status and timestamp */}
        <div className="flex items-center space-x-4">
          {/* Active camera count display */}
          <span className="text-sm text-gray-700">
            Cameras: {stats.activeCameras}/{stats.totalCameras}
          </span>
          {/* Current system timestamp */}
          <span className="text-sm text-gray-500">
            {new Date().toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}