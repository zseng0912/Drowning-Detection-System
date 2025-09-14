// User profile dropdown component displaying user information, status, and quick actions
import React, { useState } from 'react';
import { User, Clock, Shield, Award, LogOut, Settings, Bell } from 'lucide-react';

// Props interface for the UserProfile component
interface UserProfileProps {
  user: any;
  onLogout: () => void;
}

// Main UserProfile component with dropdown functionality for user information and actions
export default function UserProfile({ user, onLogout }: UserProfileProps) {
  // State for controlling dropdown visibility
  const [showDropdown, setShowDropdown] = useState(false);

  // Helper function to get status indicator color based on duty status
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'on-duty': return 'bg-green-500';
      case 'off-duty': return 'bg-gray-500';
      case 'break': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  // Helper function to get readable status text
  const getStatusText = (status: string) => {
    switch (status) {
      case 'on-duty': return 'On Duty';
      case 'off-duty': return 'Off Duty';
      case 'break': return 'On Break';
      default: return 'Unknown';
    }
  };

  return (
    // Main container with relative positioning for dropdown
    <div className="relative">
      {/* Profile button trigger with avatar and basic info */}
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="flex items-center space-x-3 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg px-4 py-2 transition-all"
      >
        {/* Avatar with status indicator */}
        <div className="relative">
          <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center text-white font-semibold">
            {user.avatar}
          </div>
          <div className={`absolute -bottom-1 -right-1 w-4 h-4 ${getStatusColor(user.status)} rounded-full border-2 border-white`}></div>
        </div>
        {/* User name and role display */}
        <div className="text-left">
          <div className="text-white font-semibold text-sm">{user.name}</div>
          <div className="text-blue-200 text-xs">{user.role}</div>
        </div>
      </button>

      {/* Dropdown menu with user details and actions */}
      {showDropdown && (
        <>
          {/* Backdrop overlay to close dropdown */}
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setShowDropdown(false)}
          ></div>
          {/* Main dropdown container */}
          <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-lg shadow-xl border border-gray-200 z-20">
            {/* Profile header section with detailed user info */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-lg">
                    {user.avatar}
                  </div>
                  <div className={`absolute -bottom-1 -right-1 w-4 h-4 ${getStatusColor(user.status)} rounded-full border-2 border-white`}></div>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{user.name}</h3>
                  <p className="text-sm text-gray-600">{user.role}</p>
                  <div className="flex items-center space-x-2 mt-1">
                    <div className={`w-2 h-2 ${getStatusColor(user.status)} rounded-full`}></div>
                    <span className="text-xs text-gray-500">{getStatusText(user.status)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Current shift information section */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center space-x-2 mb-2">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">Current Shift</span>
              </div>
              <p className="text-sm text-gray-600 ml-6">{user.shift}</p>
            </div>

            {/* User certifications display section */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center space-x-2 mb-3">
                <Award className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">Certifications</span>
              </div>
              <div className="flex flex-wrap gap-2 ml-6">
                {user.certifications.map((cert: string, index: number) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                  >
                    {cert}
                  </span>
                ))}
              </div>
            </div>

            {/* Quick action buttons section */}
            <div className="p-4 space-y-2">
              <button className="w-full flex items-center space-x-3 p-2 text-left hover:bg-gray-50 rounded-lg transition-colors">
                <Settings className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-700">Settings</span>
              </button>
              <button
                onClick={onLogout}
                className="w-full flex items-center space-x-3 p-2 text-left hover:bg-red-50 rounded-lg transition-colors text-red-600"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm">Logout</span>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}