import React, { useState, useEffect } from 'react';
import LoginForm from './components/LoginForm';
import UserProfile from './components/UserProfile';
import TabNavigation from './components/TabNavigation';
import SurveillanceTab from './components/SurveillanceTab';
import DrowningDetection from './components/DrowningDetection';
import ImageEnhancement from './components/ImageEnhancement';
import EmergencyMap from './components/EmergencyMap';
import { Waves, Shield } from 'lucide-react';
import { getUser, isAuthenticated, api } from './services/api';

function App() {
  const [user, setUser] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('surveillance'); // Will be set after login
  const [isSystemActive, setIsSystemActive] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isLoading, setIsLoading] = useState(true);

  // Clear authentication on app startup to force login
  useEffect(() => {
    const clearAuthAndLoad = async () => {
      try {
        // Always clear stored authentication data on startup
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
        setUser(null);
      } catch (error) {
        console.error('Auth clear error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    clearAuthAndLoad();
  }, []);

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleLogin = (userData: any) => {
    setUser(userData);
    setActiveTab('surveillance'); // Set default tab after successful login
  };

  const handleLogout = async () => {
    try {
      await api.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      // Don't set activeTab here - let it be handled by login
    }
  };

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-teal-800 to-blue-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  // Show login form if user is not authenticated
  if (!user) {
    return <LoginForm onLogin={handleLogin} />;
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'surveillance':
        return (
          <SurveillanceTab 
            isSystemActive={isSystemActive} 
            setIsSystemActive={setIsSystemActive} 
          />
        );
      case 'detection':
        return <DrowningDetection />;
      case 'enhancement':
        return <ImageEnhancement />;
      case 'emergency':
        return <EmergencyMap />;
      default:
        return (
          <SurveillanceTab 
            isSystemActive={isSystemActive} 
            setIsSystemActive={setIsSystemActive} 
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-slate-200 flex flex-col">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-900 to-teal-800 text-white p-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-white bg-opacity-20 rounded-lg">
              <Waves className="w-8 h-8 text-blue-200" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Pool Surveillance Dashboard</h1>
              <p className="text-blue-200 text-sm">Advanced Water Safety Monitoring & Emergency Response</p>
            </div>
            <div className="flex items-center space-x-4 ml-8">
              <div className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-green-400" />
                <span className="text-sm">System Secure</span>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                isSystemActive 
                  ? 'bg-green-500 text-white' 
                  : 'bg-red-500 text-white'
              }`}>
                {isSystemActive ? 'ACTIVE' : 'INACTIVE'}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <UserProfile user={user} onLogout={handleLogout} />
            <div className="text-right">
              <div className="text-white text-lg font-bold">
                {currentTime.toLocaleDateString('en-US', { 
                  weekday: 'short', 
                  year: 'numeric', 
                  month: 'short', 
                  day: 'numeric' 
                })}
              </div>
              <div className="text-blue-200 text-base font-semibold">
                {currentTime.toLocaleTimeString('en-US', { 
                  hour12: false,
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                })}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Tab Content */}
      <div className="flex-1">
        {renderTabContent()}
      </div>
    </div>
  );
}

export default App;