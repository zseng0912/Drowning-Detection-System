import React, { useState } from 'react';
import LoginForm from './components/LoginForm';
import UserProfile from './components/UserProfile';
import TabNavigation from './components/TabNavigation';
import SurveillanceTab from './components/SurveillanceTab';
import DrowningDetection from './components/DrowningDetection';
import ImageEnhancement from './components/ImageEnhancement';
import EmergencyMap from './components/EmergencyMap';
import { Waves, Shield } from 'lucide-react';

function App() {
  const [user, setUser] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('surveillance');
  const [isSystemActive, setIsSystemActive] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update time every second
  React.useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleLogin = (userData: any) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-indigo-900 flex items-center justify-center">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-6">
            <Waves className="w-12 h-12 text-blue-400 mr-3" />
            <h1 className="text-4xl font-bold text-white">AquaGuard AI</h1>
          </div>
          <p className="text-blue-200 text-lg mb-8">Advanced Drowning Detection & Water Safety System</p>
          <LoginForm onLogin={handleLogin} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-indigo-900">
      <header className="bg-black bg-opacity-20 backdrop-blur-sm border-b border-blue-700">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Waves className="w-8 h-8 text-blue-400" />
              <h1 className="text-2xl font-bold text-white">AquaGuard AI</h1>
            </div>
            <div className="flex items-center space-x-4">
              <UserProfile user={user} onLogout={handleLogout} />
              <div className="flex flex-col items-end space-y-2">
                <div className="flex items-center space-x-4">
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
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
        
        <div className="mt-8">
          {activeTab === 'surveillance' && (
            <SurveillanceTab 
              isSystemActive={isSystemActive}
              onToggleSystem={() => setIsSystemActive(!isSystemActive)}
            />
          )}
          {activeTab === 'detection' && <DrowningDetection />}
          {activeTab === 'enhancement' && <ImageEnhancement />}
          {activeTab === 'emergency' && <EmergencyMap />}
        </div>
      </main>
    </div>
  );
}

export default App;