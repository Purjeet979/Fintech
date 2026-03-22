import React, { useState } from 'react';
import { useDashboard } from '../context/DashboardContext';
import { Search, Download, Trash2, Edit2, Check, X } from 'lucide-react';

export default function DataTable() {
  const { history, deleteExtraction, updateExtraction } = useDashboard();
  const [filter, setFilter] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState({});

  const filtered = history.filter(item => 
    item.docId.toLowerCase().includes(filter.toLowerCase()) || 
    (item.dealer && item.dealer.toLowerCase().includes(filter.toLowerCase())) ||
    (item.model && item.model.toLowerCase().includes(filter.toLowerCase()))
  );

  const handleEditClick = (item) => {
    setEditingId(item.id);
    setEditForm({
      dealer: item.dealer || '',
      model: item.model || '',
      hp: item.hp || '',
      cost: item.cost || ''
    });
  };

  const handleSave = (id) => {
    updateExtraction(id, editForm);
    setEditingId(null);
  };

  const handleCancel = () => {
    setEditingId(null);
    setEditForm({});
  };

  const handleChange = (e, field) => {
    setEditForm(prev => ({ ...prev, [field]: e.target.value }));
  };

  const exportCSV = () => {
    if (filtered.length === 0) return;
    
    const headers = ['Document', 'Dealer', 'Model', 'HP', 'Cost', 'Confidence', 'Date'];
    const csvRows = [headers.join(',')];
    
    filtered.forEach(item => {
      const row = [
        `"${item.docId || ''}"`,
        `"${item.dealer || ''}"`,
        `"${item.model || ''}"`,
        `"${item.hp || ''}"`,
        `"${item.cost || ''}"`,
        `${item.confidence || 0}`,
        `"${new Date(item.date).toLocaleDateString()}"`
      ];
      csvRows.push(row.join(','));
    });
    
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'extractions_data.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-10">
      <header className="flex justify-between items-center">
        <div>
          <h2 className="text-4xl font-extrabold tracking-tighter text-on-surface">Data Table</h2>
          <p className="text-on-surface-variant mt-2 text-sm">Comprehensive view & edit of all extracted fields</p>
        </div>
        <div className="flex gap-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search data..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="bg-surface-container-lowest border-none rounded-full px-4 py-2 w-64 text-sm focus:ring-2 focus:ring-tertiary-container outline-none text-on-surface h-10"
            />
            <Search size={16} className="absolute right-4 top-3 text-on-surface-variant pointer-events-none" />
          </div>
          <button 
            onClick={exportCSV}
            className="flex items-center gap-2 bg-surface-container-high hover:bg-surface-container-highest transition-colors rounded-full px-4 py-2 text-sm font-bold text-on-surface h-10"
          >
            <Download size={16} />
            Export CSV
          </button>
        </div>
      </header>

      <div className="bg-surface-container-low rounded-[2rem] border border-outline-variant/10 overflow-hidden shadow-xl p-2 mt-8">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse whitespace-nowrap">
            <thead>
              <tr className="border-b border-outline-variant/15 text-xs uppercase tracking-wider text-on-surface-variant font-bold">
                <th className="p-4 px-6 rounded-tl-xl">Document</th>
                <th className="p-4">Dealer Name</th>
                <th className="p-4">Model Name</th>
                <th className="p-4">HP (Horse Power)</th>
                <th className="p-4">Asset Cost</th>
                <th className="p-4 text-center">Confidence</th>
                <th className="p-4 px-6 text-right rounded-tr-xl">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/5 text-sm">
              {filtered.map(item => {
                const isEditing = editingId === item.id;
                
                return (
                  <tr key={item.id} className={`transition-colors text-on-surface group ${isEditing ? 'bg-surface-container-high' : 'hover:bg-surface-container/50'}`}>
                    <td className="p-4 px-6 font-medium text-primary-fixed">{item.docId}</td>
                    
                    <td className="p-4">
                      {isEditing ? (
                        <input type="text" value={editForm.dealer} onChange={e => handleChange(e, 'dealer')} className="bg-surface-container-lowest border border-outline-variant/30 rounded px-2 py-1 w-full max-w-[200px] text-sm focus:outline-none focus:ring-1 focus:ring-primary" />
                      ) : (
                        <div className="truncate max-w-[200px] opacity-90" title={item.dealer}>{item.dealer || 'N/A'}</div>
                      )}
                    </td>
                    
                    <td className="p-4">
                      {isEditing ? (
                        <input type="text" value={editForm.model} onChange={e => handleChange(e, 'model')} className="bg-surface-container-lowest border border-outline-variant/30 rounded px-2 py-1 w-full max-w-[150px] text-sm focus:outline-none focus:ring-1 focus:ring-primary" />
                      ) : (
                        <div className="truncate max-w-[150px] opacity-90" title={item.model}>{item.model || 'N/A'}</div>
                      )}
                    </td>
                    
                    <td className="p-4">
                      {isEditing ? (
                        <input type="text" value={editForm.hp} onChange={e => handleChange(e, 'hp')} className="bg-surface-container-lowest border border-outline-variant/30 rounded px-2 py-1 w-full max-w-[80px] text-sm focus:outline-none focus:ring-1 focus:ring-primary" />
                      ) : (
                        <div className="opacity-90">{item.hp || 'N/A'}</div>
                      )}
                    </td>
                    
                    <td className="p-4">
                      {isEditing ? (
                        <input type="text" value={editForm.cost} onChange={e => handleChange(e, 'cost')} className="bg-surface-container-lowest border border-outline-variant/30 rounded px-2 py-1 w-full max-w-[120px] text-sm focus:outline-none focus:ring-1 focus:ring-primary" />
                      ) : (
                        <div className="opacity-90 font-mono text-tertiary">{item.cost || 'N/A'}</div>
                      )}
                    </td>
                    
                    <td className="p-4 text-center">
                      <span className={`px-2 py-1 rounded inline-flex font-bold text-xs ${item.confidence > 90 ? 'bg-tertiary/10 text-tertiary' : item.confidence > 75 ? 'bg-primary/10 text-primary' : 'bg-error/10 text-error'}`}>
                        {item.confidence}%
                      </span>
                    </td>
                    
                    <td className="p-4 px-6 text-right flex gap-2 justify-end items-center h-full">
                      {isEditing ? (
                        <>
                          <button onClick={() => handleSave(item.id)} className="text-secondary hover:text-secondary-hover bg-secondary/10 p-1.5 rounded transition-colors" title="Save changes">
                            <Check size={16} />
                          </button>
                          <button onClick={handleCancel} className="text-on-surface-variant hover:text-on-surface bg-surface-container-highest p-1.5 rounded transition-colors" title="Cancel edit">
                            <X size={16} />
                          </button>
                        </>
                      ) : (
                        <>
                          <button onClick={() => handleEditClick(item)} className="text-primary hover:text-primary-hover transition-colors opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-primary/10" title="Edit record">
                             <Edit2 size={16} />
                          </button>
                          <button onClick={() => deleteExtraction(item.id)} className="text-error hover:text-error-hover transition-colors opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-error/10" title="Delete record">
                             <Trash2 size={16} />
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan="7" className="p-16 text-center text-on-surface-variant">No records found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
