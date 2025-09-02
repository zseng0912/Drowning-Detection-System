import React from 'react';
import { Wifi, HardDrive, Cpu, Thermometer } from 'lucide-react';

interface SystemStats {
  networkStatus: 'online' | 'offline';
  storageUsed: number;
  cpuUsage: number;
  temperature: number;
  activeCameras: number;
  totalCameras: number;
}

interface StatusBarProps {
  stats: SystemStats;
}

export default function StatusBar({ stats }: StatusBarProps) {
  const getStatusColor = (value: number, thresholds: { warning: number; danger: number }) => {
    if (value >= thresholds.danger) return 'text-red-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="bg-white border-t border-gray-200 px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-8">
          {/* Network Status */}
          <div className="flex items-center space-x-2">
            <Wifi className={`w-4 h-4 ${
              stats.networkStatus === 'online' ? 'text-green-600' : 'text-red-600'
            }`} />
            <span className="text-sm text-gray-700">
              Network: {stats.networkStatus}
            </span>
          </div>

          {/* Storage Usage */}
          <div className="flex items-center space-x-2">
            <HardDrive className={`w-4 h-4 ${
              getStatusColor(stats.storageUsed, { warning: 80, danger: 90 })
            }`} />
            <span className="text-sm text-gray-700">
              Storage: {stats.storageUsed}%
            </span>
          </div>

          {/* CPU Usage */}
          <div className="flex items-center space-x-2">
            <Cpu className={`w-4 h-4 ${
              getStatusColor(stats.cpuUsage, { warning: 70, danger: 85 })
            }`} />
            <span className="text-sm text-gray-700">
              CPU: {stats.cpuUsage}%
            </span>
          </div>

          {/* Temperature */}
          <div className="flex items-center space-x-2">
            <Thermometer className={`w-4 h-4 ${
              getStatusColor(stats.temperature, { warning: 60, danger: 70 })
            }`} />
            <span className="text-sm text-gray-700">
              Temp: {stats.temperature}°C
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-700">
            Cameras: {stats.activeCameras}/{stats.totalCameras}
          </span>
          <span className="text-sm text-gray-500">
            {new Date().toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}