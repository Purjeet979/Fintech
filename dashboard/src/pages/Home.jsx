import React, { useState, useRef, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useDashboard } from '../context/DashboardContext';
import { UploadCloud, ChevronDown, Sparkles, CheckCircle, Edit3, FileImage, Table } from 'lucide-react';

export default function Home() {
  const { currentExtraction, setCurrentExtraction, addToHistory } = useDashboard();
  const [isExtracting, setIsExtracting] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [mode, setMode] = useState('fast');
  const [isEditing, setIsEditing] = useState(false);
  const fileInputRef = useRef(null);
  const [loadingStep, setLoadingStep] = useState(0);

  useEffect(() => {
    if (!isExtracting) {
      setLoadingStep(0);
      return;
    }
    const steps = [0, 800, 2000, 3500];
    const timers = steps.map((delay, idx) => setTimeout(() => setLoadingStep(idx), delay));
    return () => timers.forEach(clearTimeout);
  }, [isExtracting]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleExtract = async () => {
    if (!selectedFile) {
      alert("Please upload a file first");
      return;
    }

    setIsExtracting(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', mode);

    try {
      const response = await fetch('/api/extract', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error("Extraction failed");
      }

      const result = await response.json();
      
      setCurrentExtraction({
        id: Date.now().toString(),
        docId: selectedFile.name,
        dealer: result.fields?.dealer_name || 'Unknown',
        model: result.fields?.model_name || 'Unknown',
        hp: result.fields?.horse_power ? `${result.fields.horse_power} HP` : 'N/A',
        cost: result.fields?.asset_cost ? `$ / ₹ ${result.fields.asset_cost}` : 'N/A',
        confidence: Math.round((result.confidence || 0) * 100),
        date: new Date().toISOString(),
        signature: result.fields?.signature?.present || false,
        stamp: result.fields?.stamp?.present || false,
        provider: mode === 'fast' ? 'Fast Mode' : mode.toUpperCase(),
        time: `${result.processing_time_sec || 0}s`
      });

    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto flex flex-col lg:flex-row gap-10">
      <div className="flex-1 space-y-10">
        <header>
          <h2 className="text-4xl lg:text-[2.5rem] font-extrabold tracking-tighter text-on-surface">
            Tractor Loan <span className="text-tertiary">Quotation AI</span>
          </h2>
          <p className="text-on-surface-variant mt-2 max-w-lg">
            Advanced neural extraction for dealership invoices. Upload your documents to begin automated data structuring.
          </p>
        </header>

        <section className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-primary to-tertiary rounded-3xl blur opacity-10 group-hover:opacity-25 transition duration-1000"></div>
          <div className="relative bg-surface-container rounded-3xl border-2 border-dashed border-outline-variant/30 p-12 flex flex-col items-center justify-center text-center backdrop-blur-md overflow-hidden min-h-[300px]">
             <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".pdf,.png,.jpg,.jpeg" />
             <div className="w-24 h-24 bg-surface-container-high rounded-full flex items-center justify-center mb-6 border border-outline-variant/20 shadow-2xl relative z-10">
               {selectedFile ? <FileImage size={40} className="text-tertiary" /> : <UploadCloud size={40} className="text-primary" />}
             </div>
             <h3 className="text-xl font-bold text-on-surface mb-2 relative z-10">
               {selectedFile ? selectedFile.name : 'Drop Tractor Quotation here'}
             </h3>
             <p className="text-sm text-on-surface-variant mb-8 relative z-10">
               {selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` : 'Supports PDF, PNG, JPG (Max 25MB)'}
             </p>
             <button 
               onClick={() => fileInputRef.current?.click()}
               className="bg-surface-container-highest px-8 py-3 rounded-xl border border-primary/20 font-bold text-sm text-primary hover:bg-primary hover:text-on-primary transition-all relative z-10">
               {selectedFile ? 'Change File' : 'Browse Files'}
             </button>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <label className="text-[0.6875rem] font-bold uppercase tracking-widest text-on-surface-variant ml-1">AI Model Selection</label>
            <div className="relative">
              <select 
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full bg-surface-container-lowest border-none rounded-xl px-5 py-4 text-sm appearance-none focus:ring-2 focus:ring-tertiary transition-all cursor-pointer text-on-surface">
                <option value="fast">Fast Mode / No VLM</option>
                <option value="granite">IBM Granite (Balanced)</option>
                <option value="qwen" >Qwen2-VL (High Precision)</option>
              </select>
              <ChevronDown size={16} className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-outline" />
            </div>
          </div>
          <div className="flex items-end">
             <button 
               onClick={handleExtract}
               disabled={isExtracting}
               className="w-full bg-gradient-to-br from-[#b8c3ff] to-[#2e5bff] py-4 rounded-xl font-black text-on-primary tracking-wide shadow-xl shadow-primary/10 hover:shadow-primary/30 transition-all flex items-center justify-center gap-3 disabled:opacity-50 group"
             >
               {isExtracting ? (
                 <div className="w-5 h-5 border-2 border-on-primary border-t-transparent rounded-full animate-spin"></div>
               ) : (
                 <Sparkles size={20} className="group-hover:rotate-12 transition-transform" />
               )}
               {isExtracting ? 'Extracting...' : 'Extract Data'}
             </button>
          </div>
        </section>
      </div>

      <div className="w-full lg:w-[420px] space-y-8">
        <div className="relative bg-surface-container-low rounded-[2rem] p-8 border border-outline-variant/10 shadow-2xl min-h-[600px] overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-tertiary/5 blur-3xl rounded-full -mr-10 -mt-10"></div>
          
          {isExtracting ? (
            <div className="h-full flex flex-col pt-4 relative z-10">
              <div className="flex items-center gap-4 mb-10 text-tertiary">
                <div className="relative w-7 h-7">
                  <div className="w-7 h-7 border-[3px] border-tertiary border-t-transparent rounded-full animate-spin"></div>
                </div>
                <h3 className="text-[1.35rem] font-bold text-on-surface">Processing document...</h3>
              </div>
              
              <div className="flex flex-col gap-6 pl-3 relative">
                <div className="absolute left-[15px] top-4 bottom-4 w-0.5 bg-outline-variant/20 -z-10"></div>
                
                {[
                  "OCR & text extraction",
                  "Layout analysis",
                  "Field extraction",
                  "Confidence scoring"
                ].map((stepText, idx) => {
                  const isActive = loadingStep >= idx;
                  const isCurrent = loadingStep === idx;
                  return (
                    <div key={idx} className={`flex items-center gap-5 transition-all duration-700 ${isActive ? 'opacity-100 translate-x-0' : 'opacity-30 -translate-x-2'}`}>
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center transition-colors duration-500 bg-surface-container-low outline outline-4 outline-surface-container-low`}>
                         <div className={`w-2.5 h-2.5 rounded-full transition-all duration-500 delay-150 ${isActive ? 'bg-tertiary scale-100 shadow-[0_0_10px_rgba(76,214,255,0.8)]' : 'bg-outline-variant scale-75'}`}></div>
                      </div>
                      <span className={`text-[15px] transition-colors duration-500 ${isActive ? (isCurrent ? 'text-on-surface font-semibold' : 'text-on-surface-variant font-medium') : 'text-on-surface-variant/40'}`}>
                        {stepText === "Confidence scoring" ? "Confidence & Consensus" : stepText}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <>
              <div className="flex justify-between items-start mb-10 relative z-10">
                <div>
                  <span className="text-[10px] bg-tertiary/10 text-tertiary px-2 py-1 rounded-full font-bold uppercase tracking-tighter mb-2 inline-block">
                    {currentExtraction ? 'Analysis Complete' : 'Awaiting Document'}
                  </span>
                  <h4 className="text-xl font-extrabold text-on-surface">Extracted Fields</h4>
                </div>
                {currentExtraction && (
                  <div className="text-right">
                    <div className="text-[2.5rem] font-bold text-tertiary leading-none tracking-tighter">
                      {currentExtraction.confidence}<span className="text-lg">%</span>
                    </div>
                    <span className="text-[10px] font-bold text-on-surface-variant uppercase">Confidence</span>
                  </div>
                )}
              </div>

              {currentExtraction ? (
                <div className="space-y-10 pb-20 relative z-10">
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold uppercase tracking-[0.05em] text-on-surface-variant">Dealer Name</label>
                    {isEditing ? (
                      <input className="w-full bg-surface-container border border-outline-variant/30 rounded p-1 text-primary-fixed text-lg font-medium focus:outline-none focus:border-tertiary" value={currentExtraction.dealer} onChange={e => setCurrentExtraction({...currentExtraction, dealer: e.target.value})} />
                    ) : (
                      <p className="text-primary-fixed text-lg font-medium leading-tight">{currentExtraction.dealer}</p>
                    )}
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold uppercase tracking-[0.05em] text-on-surface-variant">Model Name</label>
                    {isEditing ? (
                      <input className="w-full bg-surface-container border border-outline-variant/30 rounded p-1 text-primary-fixed text-lg font-medium focus:outline-none focus:border-tertiary" value={currentExtraction.model} onChange={e => setCurrentExtraction({...currentExtraction, model: e.target.value})} />
                    ) : (
                      <p className="text-primary-fixed text-lg font-medium leading-tight">{currentExtraction.model}</p>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold uppercase tracking-[0.05em] text-on-surface-variant">Horse Power (HP)</label>
                      {isEditing ? (
                        <input className="w-full bg-surface-container border border-outline-variant/30 rounded p-1 text-primary-fixed text-lg font-medium focus:outline-none focus:border-tertiary" value={currentExtraction.hp} onChange={e => setCurrentExtraction({...currentExtraction, hp: e.target.value})} />
                      ) : (
                        <p className="text-primary-fixed text-lg font-medium">{currentExtraction.hp}</p>
                      )}
                    </div>
                    <div className="space-y-1 text-right">
                      <label className="text-[10px] font-bold uppercase tracking-[0.05em] text-on-surface-variant">Asset Cost</label>
                      {isEditing ? (
                        <input className="w-full bg-surface-container border border-outline-variant/30 rounded p-1 text-primary-fixed text-lg font-bold text-right focus:outline-none focus:border-tertiary" value={currentExtraction.cost} onChange={e => setCurrentExtraction({...currentExtraction, cost: e.target.value})} />
                      ) : (
                        <p className="text-primary-fixed text-lg font-bold">{currentExtraction.cost}</p>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 bg-surface-container-lowest/50 p-4 rounded-2xl border border-outline-variant/10">
                     <div className="flex items-center gap-2">
                       <div className={`w-2 h-2 rounded-full ${currentExtraction.signature ? 'bg-tertiary animate-pulse shadow-[0_0_8px_rgba(76,214,255,0.8)]' : 'bg-outline-variant'}`}></div>
                       <span className="text-xs font-bold text-on-surface tracking-wide">Signature</span>
                     </div>
                     <div className="flex items-center gap-2 justify-end">
                       <div className={`w-2 h-2 rounded-full ${currentExtraction.stamp ? 'bg-tertiary animate-pulse shadow-[0_0_8px_rgba(76,214,255,0.8)]' : 'bg-outline-variant'}`}></div>
                       <span className="text-xs font-bold text-on-surface tracking-wide">Stamp Found</span>
                     </div>
                  </div>
                  
                  <div className="absolute bottom-8 left-8 right-8 flex gap-3">
                     <button onClick={() => setIsEditing(!isEditing)} className="flex-1 bg-surface-container-highest border border-outline-variant/20 py-3 rounded-xl text-xs font-bold hover:bg-surface-variant transition-colors flex items-center justify-center gap-2">
                       <Edit3 size={14} /> {isEditing ? 'Save Edits' : 'Edit'}
                     </button>
                     <button onClick={() => { addToHistory(currentExtraction); setCurrentExtraction(null); setIsEditing(false); }} className="flex-1 bg-primary text-on-primary py-3 rounded-xl text-xs font-bold hover:bg-primary-fixed-dim transition-colors flex items-center justify-center gap-2">
                       <CheckCircle size={14} /> Confirm
                     </button>
                  </div>
                  
                  <div className="pt-2 text-center">
                    <NavLink to="/datatable" className="text-[10px] font-bold text-tertiary flex items-center justify-center gap-1 hover:underline">
                      <Table size={12} /> View all stored data in Tabular Format
                    </NavLink>
                  </div>
                  
                </div>
              ) : (
                <div className="flex items-center justify-center h-48 border-t border-outline-variant/10 text-on-surface-variant text-sm mt-8 relative z-10">
                    No extraction data yet. Run "Extract Data".
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
