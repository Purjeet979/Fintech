import React, { useState } from 'react';
import { useDashboard } from '../context/DashboardContext';
import { Search, Trash2, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function History() {
  const { history, deleteExtraction } = useDashboard();
  const [filter, setFilter] = useState('');

  const filtered = history.filter(item => 
    item.docId.toLowerCase().includes(filter.toLowerCase()) || 
    item.dealer.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h2 className="text-4xl font-extrabold tracking-tighter text-on-surface">Extraction History</h2>
          <p className="text-on-surface-variant mt-2 text-sm">Review past documents and verified data</p>
        </div>
        <div className="relative">
          <input
            type="text"
            placeholder="Search by ID or dealer..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-surface-container-lowest border-none rounded-full px-4 py-2 w-64 text-sm focus:ring-2 focus:ring-tertiary-container outline-none text-on-surface h-10"
          />
          <Search size={16} className="absolute right-4 top-3 text-on-surface-variant pointer-events-none" />
        </div>
      </header>

      {/* Accuracy Chart */}
      {history.length > 0 && (
        <div className="bg-surface-container-low rounded-[2rem] p-8 mt-5 mb-8 border border-outline-variant/10 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-tertiary/10 rounded-lg">
              <TrendingUp className="text-tertiary" size={20} />
            </div>
            <h3 className="text-lg font-bold text-on-surface">Extraction Accuracy Over Time</h3>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={[...history].reverse()} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" vertical={false} opacity={0.3} />
                <XAxis dataKey="docId" stroke="#718096" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(str) => str.substring(0, 10) + '...'} />
                <YAxis stroke="#718096" fontSize={12} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a202c', borderColor: '#2d3748', borderRadius: '0.5rem' }}
                  itemStyle={{ color: '#63b3ed' }}
                  formatter={(value) => [`${value}%`, 'Confidence']}
                />
                <Line type="monotone" dataKey="confidence" stroke="#4cd6ff" strokeWidth={3} dot={{ fill: '#4cd6ff', strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: '#b8c3ff' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="bg-surface-container-low rounded-[2rem] border border-outline-variant/10 overflow-hidden shadow-xl p-2 mt-8">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-outline-variant/15 text-xs uppercase tracking-wider text-on-surface-variant font-bold">
              <th className="p-4 px-6 rounded-tl-xl">Document</th>
              <th className="p-4">Dealer</th>
              <th className="p-4">Confidence</th>
              <th className="p-4 px-6 text-right rounded-tr-xl">Date</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-outline-variant/5 text-sm">
            {filtered.map(item => (
              <tr key={item.id} className="hover:bg-surface-container/50 transition-colors cursor-pointer text-on-surface group">
                <td className="p-4 px-6 font-medium text-primary-fixed">{item.docId}</td>
                <td className="p-4 opacity-90">{item.dealer}</td>
                <td className="p-4">
                  <span className={`px-2 py-1 rounded inline-flex font-bold text-xs ${item.confidence > 90 ? 'bg-tertiary/10 text-tertiary' : 'bg-error/10 text-error'}`}>
                    {item.confidence}%
                  </span>
                </td>
                <td className="p-4 px-6 text-right text-on-surface-variant flex gap-4 justify-end items-center h-full">
                  <span>{new Date(item.date).toLocaleDateString()}</span>
                  <button onClick={(e) => { e.stopPropagation(); deleteExtraction(item.id); }} className="text-error hover:text-error-hover transition-colors opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-error/10" title="Delete record">
                     <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan="4" className="p-16 text-center text-on-surface-variant">No records found matching "{filter}".</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
