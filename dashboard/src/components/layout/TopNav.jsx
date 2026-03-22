import React from 'react';
import { Search, Bell, HelpCircle } from 'lucide-react';
import { NavLink } from 'react-router-dom';

export default function TopNav() {
  return (
    <header className="fixed top-0 w-full z-40 border-b border-primary/10 bg-background/60 backdrop-blur-xl flex justify-between items-center h-20 px-8 pl-[18rem]">
      <div className="flex items-center gap-8">
        <span className="text-2xl font-black bg-gradient-to-br from-primary to-primary-container bg-clip-text text-transparent">
          Ethereal Engine
        </span>
      </div>

      <div className="flex items-center gap-6">
        <div className="relative flex items-center">
          <input
            type="text"
            placeholder="Search extractions..."
            className="bg-surface-container-lowest border-none rounded-full px-4 py-2 flex items-center h-10 text-xs w-64 focus:ring-2 focus:ring-tertiary-container placeholder:text-outline-variant text-on-surface"
          />
          <Search size={16} className="absolute right-3 text-on-surface-variant" />
        </div>

        <div className="flex items-center gap-4 text-on-surface-variant">
          <Bell size={20} className="cursor-pointer hover:text-primary transition-colors" />
          <HelpCircle size={20} className="cursor-pointer hover:text-primary transition-colors" />
        </div>

        <button className="bg-primary/10 text-primary border border-primary/20 px-5 py-2 rounded-lg text-xs font-bold hover:bg-primary/20 transition-all">
          Extract Data
        </button>
      </div>
    </header>
  );
}
