import React from 'react';
import { useDashboard } from '../context/DashboardContext';

export default function Settings() {
  const { settings, updateSettings } = useDashboard();

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <header>
        <h2 className="text-4xl font-extrabold tracking-tighter text-on-surface">Preferences</h2>
        <p className="text-on-surface-variant mt-2 text-sm">Manage dashboard display and behavior</p>
      </header>

      <div className="bg-surface-container-low rounded-[2rem] border border-outline-variant/10 p-8 shadow-xl space-y-8">
        <div>
          <h3 className="text-base font-bold text-primary-fixed border-b border-outline-variant/10 pb-4 mb-6">Appearance</h3>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-bold text-on-surface">Dark Mode</h4>
              <p className="text-xs text-on-surface-variant mt-1">Switch between Ethereal Dark and Light modes</p>
            </div>
            <button 
              onClick={() => updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' })}
              className={`w-12 h-6 rounded-full relative transition-colors ${settings.theme === 'dark' ? 'bg-primary' : 'bg-surface-container-highest'}`}
            >
              <div className={`absolute top-1 w-4 h-4 rounded-full bg-on-primary transition-all ${settings.theme === 'dark' ? 'left-7' : 'left-1'}`} />
            </button>
          </div>
        </div>

        <div>
          <h3 className="text-base font-bold text-primary-fixed border-b border-outline-variant/10 pb-4 mb-6">Alerts</h3>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-bold text-on-surface">Push Notifications</h4>
              <p className="text-xs text-on-surface-variant mt-1">Receive alerts on extraction errors or low confidence scores</p>
            </div>
            <button 
              onClick={() => updateSettings({ notifications: !settings.notifications })}
              className={`w-12 h-6 rounded-full relative transition-colors ${settings.notifications ? 'bg-tertiary' : 'bg-surface-container-highest'}`}
            >
              <div className={`absolute top-1 w-4 h-4 rounded-full bg-background transition-all ${settings.notifications ? 'left-7' : 'left-1'}`} />
            </button>
          </div>
        </div>
      </div>
      
    </div>
  );
}
