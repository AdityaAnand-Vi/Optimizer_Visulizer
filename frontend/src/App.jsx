import React, { useState } from 'react';
import PartAPlayground from './components/PartAPlayground';
import PartBDashboard from './components/PartBDashboard';

export default function App() {
  const [activeTab, setActiveTab] = useState('playground');
  const [appMode, setAppMode] = useState('Simple mode');

  return (
    <div className="min-h-screen bg-[#10141A] text-[#E7EAEE]">
      {/* Header */}
      <header className="border-b border-[#2E3742] bg-[#13171F] px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <span className="text-2xl">⚡</span>
          <div>
            <h1 className="text-xl font-bold tracking-wide text-white">Optimizer Visualizer & Benchmark Suite</h1>
            <p className="text-xs text-[#8B96A3]">Pure NumPy Full-Stack Interactive Telemetry Panel</p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-[#1A2029] border border-[#2E3742] p-1 rounded">
          <button
            onClick={() => setAppMode('Simple mode')}
            className={`px-3 py-1 text-xs font-semibold rounded ${appMode === 'Simple mode' ? 'bg-[#E3A23B] text-[#10141A]' : 'text-[#8B96A3]'}`}
          >
            Simple Mode
          </button>
          <button
            onClick={() => setAppMode('Advanced mode')}
            className={`px-3 py-1 text-xs font-semibold rounded ${appMode === 'Advanced mode' ? 'bg-[#E3A23B] text-[#10141A]' : 'text-[#8B96A3]'}`}
          >
            Advanced Mode
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="border-b border-[#2E3742] bg-[#13171F] px-6 flex gap-6 text-sm font-semibold">
        <button
          onClick={() => setActiveTab('playground')}
          className={`py-3 border-b-2 ${activeTab === 'playground' ? 'border-[#E3A23B] text-[#E3A23B]' : 'border-transparent text-[#8B96A3] hover:text-white'}`}
        >
          ⚡ Part A: 2D Loss Surface Playground
        </button>
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`py-3 border-b-2 ${activeTab === 'dashboard' ? 'border-[#E3A23B] text-[#E3A23B]' : 'border-transparent text-[#8B96A3] hover:text-white'}`}
        >
          🧠 Part B: Neural Network Benchmark
        </button>
      </div>

      {/* Main Content */}
      <main className="p-6">
        {activeTab === 'playground' ? (
          <PartAPlayground appMode={appMode} />
        ) : (
          <PartBDashboard appMode={appMode} />
        )}
      </main>
    </div>
  );
}
