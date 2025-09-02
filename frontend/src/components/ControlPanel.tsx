import React from 'react';
import { RotateCcw, Settings, Shield } from 'lucide-react';

interface ControlPanelProps {
  isSystemActive: boolean;
  onToggleSystem: () => void;
  onEmergencyStop: () => void;
  onResetSystem: () => void;
}

export default function ControlPanel({ 
  isSystemActive, 
  onToggleSystem, 
  onEmergencyStop, 
  onResetSystem 
}: ControlPanelProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
        <Settings className="w-5 h-5 mr-2" />
        System Controls
      </h3>
      
      <div className="space-y-4">
        {/* System Status */}
        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className={`w-4 h-4 rounded-full ${
              isSystemActive ? 'bg-green-500' : 'bg-red-500'
            } animate-pulse`}></div>
            <span className="font-semibold text-gray-900">
              System Status: {isSystemActive ? 'Active' : 'Inactive'}
            </span>
          </div>
          <button
            onClick={onToggleSystem}
            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
              isSystemActive 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isSystemActive ? 'Stop System' : 'Start System'}
          </button>
        </div>

        {/* Control Buttons */}
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={onEmergencyStop}
            className="flex items-center justify-center space-x-2 p-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-colors"
          >
            <Shield className="w-5 h-5" />
            <span>Emergency Stop</span>
          </button>
          
          <button
            onClick={onResetSystem}
            className="flex items-center justify-center space-x-2 p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
          >
            <RotateCcw className="w-5 h-5" />
            <span>Reset System</span>
          </button>
        </div>

        {/* Quick Actions */}
        <div className="border-t pt-4">
          <h4 className="font-semibold text-gray-900 mb-3">System Overview</h4>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-blue-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">4</div>
              <div className="text-xs text-blue-700">Total Zones</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-600">4</div>
              <div className="text-xs text-green-700">Active Cameras</div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-red-600">0</div>
              <div className="text-xs text-red-700">Inactive Cameras</div>
            </div>
            <div className="bg-yellow-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-yellow-600">1</div>
              <div className="text-xs text-yellow-700">Active Alerts</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}