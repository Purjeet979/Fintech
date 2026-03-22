import React, { createContext, useContext, useState, useEffect } from 'react';

const DashboardContext = createContext();

const initialHistory = [
  { id: '1', docId: 'Tractor_Inv_9921.pdf', dealer: 'Mahindra Agri-Solutions Pvt Ltd.', model: 'Arjun Novo 605 DI-i', hp: '57 HP', cost: '₹ 8,95,000', confidence: 98, date: '2026-03-22T09:42:00', signature: true, stamp: true, provider: 'Qwen2-VL', time: '1.42s' },
  { id: '2', docId: 'Quotation_Swaraj_855.pdf', dealer: 'Swaraj Dealership', model: 'Swaraj 855 FE', hp: '52 HP', cost: '₹ 7,50,000', confidence: 91, date: '2026-03-22T08:15:00', signature: true, stamp: false, provider: 'IBM Granite', time: '2.1s' }
];

export const DashboardProvider = ({ children }) => {
  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem('dash_history');
    return saved ? JSON.parse(saved) : initialHistory;
  });

  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('dash_settings');
    return saved ? JSON.parse(saved) : { theme: 'dark', notifications: true };
  });

  const [currentExtraction, setCurrentExtraction] = useState(null);

  useEffect(() => {
    localStorage.setItem('dash_history', JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    localStorage.setItem('dash_settings', JSON.stringify(settings));
    if (settings.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [settings]);

  const addExtraction = (data) => {
    // Only used for mock direct insertion if needed, replaced by manual confirm generally
  };

  const addToHistory = (data) => {
    if (!history.find(h => h.id === data.id)) {
      setHistory(prev => [data, ...prev]);
    }
  };

  const deleteExtraction = (id) => {
    setHistory(prev => prev.filter(h => h.id !== id));
  };

  const updateSettings = (newSettings) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };

  return (
    <DashboardContext.Provider value={{
      history, currentExtraction, setCurrentExtraction, addExtraction, addToHistory, deleteExtraction, settings, updateSettings
    }}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => useContext(DashboardContext);
