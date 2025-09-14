// System control panel component for pool surveillance management
// Provides system controls, status monitoring, and overview statistics
import { RotateCcw, Settings, Shield } from 'lucide-react';

// System overview statistics interface
interface SystemOverview {
  totalZones: number;
  activeCameras: number;
  inactiveCameras: number;
  activeAlerts: number;
}

// Control panel component props interface
interface ControlPanelProps {
  isSystemActive: boolean;
  onToggleSystem: () => void;
  onEmergencyStop: () => void;
  onResetSystem: () => void;
  systemOverview: SystemOverview;
  isLoading?: boolean;
}

// Main control panel component with system management capabilities
export default function ControlPanel({ 
  isSystemActive, 
  onToggleSystem, 
  onEmergencyStop, 
  onResetSystem,
  systemOverview,
  isLoading = false
}: ControlPanelProps) {
  // Main control panel layout with system controls and status
  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
      {/* Control Panel Header */}
      <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
        <Settings className="w-5 h-5 mr-2" />
        System Controls
      </h3>
      
      <div className="space-y-4">
        {/* System Status Display - Shows current system state and toggle */}
        <div className={`flex items-center justify-between p-4 rounded-lg transition-all duration-500 ${
          isLoading ? 'bg-blue-50 border-2 border-blue-200' :
          isSystemActive ? 'bg-green-50 border-2 border-green-200' : 'bg-red-50 border-2 border-red-200'
        }`}>
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className={`w-4 h-4 rounded-full transition-all duration-300 ${
                isLoading ? 'bg-blue-500 animate-pulse' :
                isSystemActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}></div>
              {isLoading && (
                <div className="absolute inset-0 w-4 h-4 rounded-full bg-blue-400 animate-ping"></div>
              )}
            </div>
            <div>
              <span className="font-semibold text-gray-900">
                System Status: {isLoading ? 'Processing...' : isSystemActive ? 'Active' : 'Inactive'}
              </span>
              {isLoading && (
                <div className="text-xs text-blue-600 mt-1">
                  Please wait while system processes your request
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onToggleSystem}
            disabled={isLoading}
            className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 transform ${
              isLoading 
                ? 'bg-gray-400 cursor-not-allowed scale-95' :
              isSystemActive 
                ? 'bg-red-600 hover:bg-red-700 text-white hover:scale-105' 
                : 'bg-green-600 hover:bg-green-700 text-white hover:scale-105'
            }`}
          >
            {isLoading ? 'Processing...' : isSystemActive ? 'Stop System' : 'Start System'}
          </button>
        </div>

        {/* Emergency and Reset Control Buttons */}
        <div className="grid grid-cols-2 gap-4">
          {/* Emergency Stop Button - Immediate system shutdown */}
          <button
            onClick={onEmergencyStop}
            disabled={isLoading}
            className={`flex items-center justify-center space-x-2 p-4 rounded-lg font-semibold transition-all duration-300 transform ${
              isLoading 
                ? 'bg-gray-400 cursor-not-allowed scale-95' 
                : 'bg-red-600 hover:bg-red-700 text-white hover:scale-105 hover:shadow-lg'
            }`}
          >
            <Shield className={`w-5 h-5 transition-transform duration-300 ${
              isLoading ? 'animate-pulse' : 'group-hover:scale-110'
            }`} />
            <span>{isLoading ? 'Processing...' : 'Emergency Stop'}</span>
          </button>
          
          {/* System Reset Button - Restart system components */}
          <button
            onClick={onResetSystem}
            disabled={isLoading}
            className={`flex items-center justify-center space-x-2 p-4 rounded-lg font-semibold transition-all duration-300 transform ${
              isLoading 
                ? 'bg-gray-400 cursor-not-allowed scale-95' 
                : 'bg-blue-600 hover:bg-blue-700 text-white hover:scale-105 hover:shadow-lg'
            }`}
          >
            <RotateCcw className={`w-5 h-5 transition-transform duration-300 ${
              isLoading ? 'animate-spin' : 'group-hover:scale-110'
            }`} />
            <span>{isLoading ? 'Resetting...' : 'Reset System'}</span>
          </button>
        </div>

        {/* System Overview Statistics Section */}
        <div className="border-t pt-4">
          <h4 className="font-semibold text-gray-900 mb-3">System Overview</h4>
          {/* Statistics Grid - Real-time system metrics */}
          <div className="grid grid-cols-2 gap-3">
            {/* Total Zones Counter */}
            <div className="bg-blue-50 p-3 rounded-lg text-center transition-all duration-300 hover:bg-blue-100 hover:scale-105 hover:shadow-md">
              <div className={`text-2xl font-bold text-blue-600 transition-all duration-300 ${
                isLoading ? 'animate-pulse' : ''
              }`}>{systemOverview.totalZones}</div>
              <div className="text-xs text-blue-700">Total Zones</div>
            </div>
            {/* Active Cameras Counter */}
            <div className={`bg-green-50 p-3 rounded-lg text-center transition-all duration-300 hover:scale-105 hover:shadow-md ${
              isSystemActive ? 'hover:bg-green-100' : 'hover:bg-gray-100'
            }`}>
              <div className={`text-2xl font-bold transition-all duration-500 ${
                isSystemActive ? 'text-green-600' : 'text-gray-400'
              } ${isLoading ? 'animate-pulse' : ''}`}>
                {systemOverview.activeCameras}
              </div>
              <div className={`text-xs transition-colors duration-300 ${
                isSystemActive ? 'text-green-700' : 'text-gray-500'
              }`}>Active Cameras</div>
            </div>
            {/* Inactive Cameras Counter */}
            <div className={`bg-red-50 p-3 rounded-lg text-center transition-all duration-300 hover:scale-105 hover:shadow-md ${
              !isSystemActive ? 'hover:bg-red-100' : 'hover:bg-gray-100'
            }`}>
              <div className={`text-2xl font-bold transition-all duration-500 ${
                !isSystemActive ? 'text-red-600' : 'text-gray-400'
              } ${isLoading ? 'animate-pulse' : ''}`}>
                {systemOverview.inactiveCameras}
              </div>
              <div className={`text-xs transition-colors duration-300 ${
                !isSystemActive ? 'text-red-700' : 'text-gray-500'
              }`}>Inactive Cameras</div>
            </div>
            {/* Active Alerts Counter */}
            <div className={`bg-yellow-50 p-3 rounded-lg text-center transition-all duration-300 hover:scale-105 hover:shadow-md ${
              systemOverview.activeAlerts > 0 ? 'hover:bg-yellow-100 animate-pulse' : 'hover:bg-gray-100'
            }`}>
              <div className={`text-2xl font-bold transition-all duration-300 ${
                systemOverview.activeAlerts > 0 ? 'text-yellow-600' : 'text-gray-400'
              } ${isLoading ? 'animate-pulse' : ''}`}>
                {systemOverview.activeAlerts}
              </div>
              <div className={`text-xs transition-colors duration-300 ${
                systemOverview.activeAlerts > 0 ? 'text-yellow-700' : 'text-gray-500'
              }`}>Active Alerts</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}