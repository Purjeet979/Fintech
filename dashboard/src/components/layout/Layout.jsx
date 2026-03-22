import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopNav from './TopNav';

export default function Layout() {
  return (
    <div className="font-body selection:bg-primary-container selection:text-on-primary-container bg-background min-h-screen flex flex-col">
      <Sidebar />
      <TopNav />
      <main className="ml-64 pt-28 p-10 flex-1 h-full overflow-y-auto w-[calc(100%-16rem)]">
        <Outlet />
      </main>
    </div>
  );
}
