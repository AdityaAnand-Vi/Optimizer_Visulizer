import React, { useState, useEffect, useRef } from 'react';
import { Activity, Layers, BookOpen, HelpCircle, Sparkles } from 'lucide-react';
import PartAPlayground from './components/PartAPlayground';
import PartBDashboard from './components/PartBDashboard';
import LabReportAnswers from './components/LabReportAnswers';
import HowToUseModal from './components/HowToUseModal';

/* ── Aurora particle canvas ────────────────────────────────── */
function AuroraCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    /* Orb definitions */
    const orbs = [
      { x: 0.15, y: 0.2,  r: 0.45, color: [99, 102, 241],  speed: 0.00015, phase: 0 },
      { x: 0.75, y: 0.6,  r: 0.5,  color: [139, 92, 246],  speed: 0.00011, phase: 2 },
      { x: 0.5,  y: 0.9,  r: 0.4,  color: [6, 182, 212],   speed: 0.00019, phase: 4 },
      { x: 0.85, y: 0.15, r: 0.3,  color: [236, 72, 153],  speed: 0.00013, phase: 1 },
    ];

    /* Floating particles */
    const particles = Array.from({ length: 45 }, () => ({
      x: Math.random(),
      y: Math.random(),
      size: Math.random() * 1.5 + 0.3,
      opacity: Math.random() * 0.35 + 0.08,
      speed: Math.random() * 0.00008 + 0.00002,
      phase: Math.random() * Math.PI * 2,
    }));

    let raf;
    const draw = (t) => {
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      /* Orbs */
      orbs.forEach(o => {
        const drift = Math.sin(t * o.speed + o.phase);
        const cx = (o.x + drift * 0.08) * W;
        const cy = (o.y + Math.cos(t * o.speed + o.phase) * 0.06) * H;
        const rad = o.r * Math.min(W, H);

        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
        grad.addColorStop(0, `rgba(${o.color}, 0.10)`);
        grad.addColorStop(0.5, `rgba(${o.color}, 0.04)`);
        grad.addColorStop(1, `rgba(${o.color}, 0)`);

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, rad, 0, Math.PI * 2);
        ctx.fill();
      });

      /* Particles */
      particles.forEach(p => {
        const drift = Math.sin(t * p.speed + p.phase);
        const cx = (p.x + drift * 0.04) * W;
        const cy = ((p.y - (t * p.speed * 0.05) % 1 + 1) % 1) * H;
        const alpha = p.opacity * (0.6 + 0.4 * Math.sin(t * p.speed * 3 + p.phase));

        ctx.fillStyle = `rgba(165, 180, 252, ${alpha})`;
        ctx.beginPath();
        ctx.arc(cx, cy, p.size, 0, Math.PI * 2);
        ctx.fill();
      });

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      id="aurora-canvas"
      style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0 }}
    />
  );
}

/* ── Main App ──────────────────────────────────────────────── */
export default function App() {
  const [activeTab, setActiveTab] = useState('playground');
  const [appMode,  setAppMode]  = useState('Beginner mode');
  const [showHelpModal, setShowHelpModal] = useState(false);

  useEffect(() => {
    const handleOpenHelp = () => setShowHelpModal(true);
    window.addEventListener('openHelp', handleOpenHelp);
    return () => window.removeEventListener('openHelp', handleOpenHelp);
  }, []);

  return (
    <>
      <AuroraCanvas />

      <div className="app-root" style={{ opacity: 1, transform: 'none' }}>
        {/* ── Header ── */}
        <header className="site-header">
          <div className="logo-wrap">
            <div className="logo-badge">
              <Activity className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <div className="logo-text">Optimizer Lab</div>
              <div className="logo-sub">Pure NumPy · From SGD to AdamW</div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              className="btn-secondary"
              style={{ padding: '6px 12px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={() => setShowHelpModal(true)}
            >
              <HelpCircle className="w-3.5 h-3.5 text-indigo-400" /> Guide &amp; Hotkeys
            </button>

            <div className="live-badge">
              <div className="live-dot" />
              LIVE
            </div>

            <div className="mode-toggle">
              <button
                className={`mode-btn ${appMode.includes('Beginner') || appMode.includes('Simple') ? 'active' : ''}`}
                onClick={() => setAppMode('Beginner mode')}
              >
                Beginner
              </button>
              <button
                className={`mode-btn ${appMode.includes('Advanced') ? 'active' : ''}`}
                onClick={() => setAppMode('Advanced mode')}
              >
                Advanced
              </button>
            </div>
          </div>
        </header>

        {/* ── Navigation Tabs ── */}
        <nav className="tab-nav">
          {[
            { id: 'playground', label: 'Part A — 2D Playground', icon: Activity },
            { id: 'dashboard',  label: 'Part B — Neural Benchmark', icon: Layers },
            { id: 'theory',     label: 'Part C — Theory & Reflection Hub', icon: BookOpen },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`tab-btn ${activeTab === id ? 'active' : ''}`}
              onClick={() => setActiveTab(id)}
              style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </nav>

        {/* ── Main Viewport ── */}
        <main className="app-main">
          {activeTab === 'playground' && <PartAPlayground appMode={appMode} />}
          {activeTab === 'dashboard'  && <PartBDashboard  appMode={appMode} />}
          {activeTab === 'theory'     && <LabReportAnswers />}
        </main>
      </div>

      {/* Universal Guide Modal */}
      <HowToUseModal isOpen={showHelpModal} onClose={() => setShowHelpModal(false)} />
    </>
  );
}
