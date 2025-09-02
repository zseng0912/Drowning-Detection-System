import React from 'react';
import { Monitor, Brain, Image, MapPin } from 'lucide-react';

interface TabNavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export default function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  const tabs = [
    { id: 'surveillance', label: 'Live Surveillance', icon: Monitor },
    { id: 'detection', label: 'Drowning Detection', icon: Brain },
    { id: 'enhancement', label: 'Image Enhancement', icon: Image },
    { id: 'emergency', label: 'Emergency Map', icon: MapPin },
  ];

  return (
    <div className="bg-white border-b border-gray-200 shadow-sm">
      <div className="px-6">
        <nav className="flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}