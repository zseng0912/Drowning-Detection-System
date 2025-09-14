// Alert panel component for displaying and managing system alerts in the pool surveillance system
import { AlertTriangle, Bell, X } from 'lucide-react';

// Interface defining the structure of an alert object
interface Alert {
  id: string;
  type: 'warning' | 'danger' | 'info';
  zone: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}

// Props interface for the AlertPanel component
interface AlertPanelProps {
  alerts: Alert[];
  onAcknowledge: (id: string) => void;
  onDismiss: (id: string) => void;
}

// Main AlertPanel component for displaying active alerts with acknowledgment and dismissal functionality
export default function AlertPanel({ alerts, onAcknowledge, onDismiss }: AlertPanelProps) {
  // Function to determine alert styling based on alert type
  const getAlertColor = (type: string) => {
    switch (type) {
      case 'danger': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-yellow-500 bg-yellow-50';
      case 'info': return 'border-blue-500 bg-blue-50';
      default: return 'border-gray-500 bg-gray-50';
    }
  };

  // Function to get appropriate icon for each alert type
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'danger': return <AlertTriangle className="w-5 h-5 text-red-600" />;
      case 'warning': return <Bell className="w-5 h-5 text-yellow-600" />;
      case 'info': return <Bell className="w-5 h-5 text-blue-600" />;
      default: return <Bell className="w-5 h-5 text-gray-600" />;
    }
  };

  return (
    // Main alert panel container with scrollable content
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 max-h-96 overflow-y-auto">
      {/* Header section displaying alert count */}
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <Bell className="w-5 h-5 mr-2" />
          Active Alerts ({alerts.filter(a => !a.acknowledged).length})
        </h3>
      </div>
      
      {/* Alert list section */}
      <div className="divide-y divide-gray-200">
        {/* Display message when no alerts are present */}
        {alerts.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            No active alerts
          </div>
        ) : (
          // Map through alerts and render each alert item
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 border-l-4 ${getAlertColor(alert.type)} ${
                alert.acknowledged ? 'opacity-60' : ''
              }`}
            >
              {/* Alert content layout */}
              <div className="flex items-start justify-between">
                {/* Alert icon and content section */}
                <div className="flex items-start space-x-3">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-sm text-gray-900">{alert.zone}</span>
                      <span className="text-xs text-gray-500">
                        {alert.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 mt-1">{alert.message}</p>
                    {/* Acknowledge button for unacknowledged alerts */}
                    {!alert.acknowledged && (
                      <button
                        onClick={() => onAcknowledge(alert.id)}
                        className="mt-2 text-xs bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition-colors"
                      >
                        Acknowledge
                      </button>
                    )}
                  </div>
                </div>
                {/* Dismiss button for removing alerts */}
                <button
                  onClick={() => onDismiss(alert.id)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}