import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { UploadCloud, History, Settings, Plus, Table } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';

export default function Sidebar() {
  const navigate = useNavigate();
  const { setCurrentExtraction } = useDashboard();
  
  const handleNewExtraction = () => {
    setCurrentExtraction(null); // clear any previous result
    navigate('/');
  };

  const navItems = [
    { name: 'Upload', path: '/', icon: UploadCloud },
    { name: 'History', path: '/history', icon: History },
    { name: 'Data Table', path: '/datatable', icon: Table },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="h-screen w-64 fixed left-0 top-0 border-r border-outline-variant/15 bg-surface-container-low flex flex-col py-6 z-50">
      <div className="px-6 mb-10">
        <h1 className="text-xl font-bold tracking-tighter text-primary">ET Gen AI</h1>
        <p className="text-xs text-on-surface-variant/60 font-medium">Hackathon Edition</p>
      </div>

      <nav className="flex-1 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg mx-2 transition-colors ${
                isActive
                  ? 'bg-surface-container-high text-primary'
                  : 'text-on-surface-variant hover:bg-surface-container'
              }`
            }
          >
            <item.icon size={20} />
            <span className="font-medium text-sm">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="px-4 mt-auto">
        <button onClick={handleNewExtraction} className="w-full py-3 bg-gradient-to-br from-primary to-primary-container text-on-primary font-bold rounded-xl text-sm shadow-lg shadow-primary-container/20 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity">
          <Plus size={16} />
          New Extraction
        </button>

        <div className="mt-8 flex items-center gap-3 px-2">
          <div className="w-10 h-10 rounded-full bg-surface-container-highest flex items-center justify-center border border-outline-variant/20 overflow-hidden">
            <img 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDR3psC9jkg95qWefZeLOdX4mmNAcBWWD29YQEfhZvYEEk53Onatdm5Qhac2E7XiskWbpC1x-0Zq8-_0j-D8U54tBNmV-o6D8PXL8eAmbORoas-o54SC-4lIn-T0DPkG4XqhUXyoiaBC6XiTJp24ebvek_q4bX--b4P_iJsIha8la_VXb5WtFlTphD8pOb8bPygwXkHpzdBKqYXUJilts7IWLoNWe3ExTANQdyfaI38zhe6i37s-IOm7qobJMf7gVeESH4gZl9fatU"
              alt="User profile"
              className="w-full h-full object-cover"
            />
          </div>
          <div>
            <p className="text-xs font-bold text-on-surface">Alex Mercer</p>
            <p className="text-[10px] text-on-surface-variant">Admin Access</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
