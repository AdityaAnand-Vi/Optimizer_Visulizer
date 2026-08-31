import React, { useState, useEffect, useRef, useCallback } from 'react';
import { runBenchmarkTrain } from '../api';

const OPT_COLORS = {
  SGD:         { hex: '#FF6B6B', rgb: '255,107,107' },
  SGDMomentum: { hex: '#FFB454', rgb: '255,180,84'  },
  Momentum:    { hex: '#FFB454', rgb: '255,180,84'  },
  NAG:         { hex: '#3DD9FF', rgb: '61,217,255'  },
  AdaGrad:     { hex: '#49D99A', rgb: '73,217,154'  },
  RMSProp:     { hex: '#2DD4BF', rgb: '45,212,191'  },
  Adam:        { hex: '#7C6CFF', rgb: '124,108,255' },
  AdamW:       { hex: '#F472B6', rgb: '244,114,182' },
};

const ALL_OPTS  = ['SGD', 'SGDMomentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW'];
const DISPLAY_NAMES = {
  SGD: 'SGD',
  SGDMomentum: 'SGD Momentum',
  NAG: 'NAG',
  AdaGrad: 'AdaGrad',
  RMSProp: 'RMSProp',
  Adam: 'Adam',
  AdamW: 'AdamW'
};
const LR_OPTS   = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];

function drawChart(canvas, series, { isLog = false, isPercent = false, epochs = 100, hoverEpoch = null, hoverOpt = null } = {}) {
  if (!canvas || !series) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const P = { t: 28, r: 14, b: 20, l: 14 };

  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let g = 1; g < 5; g++) {
    const y = P.t + (g/5)*(H-P.t-P.b);
    ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W-P.r, y); ctx.stroke();
  }

  let allVals = [];
  Object.values(series).forEach(a => { if (a) allVals = allVals.concat(a.filter(v => v != null && !isNaN(v))); });
  if (!allVals.length) return;

  let minV = Math.min(...allVals), maxV = Math.max(...allVals);
  if (isPercent) { minV = Math.min(50, minV); maxV = 100.5; }

  const tx = ep => P.l + ((ep-1) / Math.max(1, epochs-1)) * (W-P.l-P.r);
  const ty = v => {
    if (!v || isNaN(v)) return H-P.b;
    if (isLog) {
      const lv = Math.log10(Math.max(1e-7, v));
      const n  = (lv - Math.log10(Math.max(1e-7, minV))) / (Math.log10(Math.max(1e-6, maxV)) - Math.log10(Math.max(1e-7, minV)) || 1);
      return H-P.b - Math.max(0,Math.min(1,n))*(H-P.t-P.b);
    }
    const n = (v - minV) / (maxV - minV || 1);
    return H-P.b - Math.max(0,Math.min(1,n))*(H-P.t-P.b);
  };

  Object.keys(series).forEach(opt => {
    const vals = series[opt];
    if (!vals?.length) return;
    const { hex } = OPT_COLORS[opt] || { hex: '#fff' };
    const isHovered = hoverOpt === opt;
    const isMuted = hoverOpt && !isHovered;
    
    ctx.strokeStyle = hex;
    ctx.lineWidth = isHovered ? 2.5 : (isMuted ? 1.0 : 1.8);
    ctx.globalAlpha = isMuted ? 0.3 : (isHovered ? 1.0 : 0.8);
    ctx.shadowColor = hex;
    ctx.shadowBlur = isMuted ? 0 : 4;
    
    ctx.beginPath(); let s = false;
    vals.forEach((v, i) => {
      if (v != null && !isNaN(v)) {
        const x = tx(i+1), y = ty(v);
        if (!s) { ctx.moveTo(x, y); s = true; } else ctx.lineTo(x, y);
      }
    });
    ctx.stroke(); ctx.shadowBlur = 0;

    const last = vals[vals.length-1];
    if (last != null) {
      ctx.fillStyle = hex; ctx.shadowColor = hex; ctx.shadowBlur = isMuted ? 0 : 10;
      ctx.beginPath(); ctx.arc(tx(vals.length), ty(last), isHovered ? 4.5 : 3.5, 0, 2*Math.PI); ctx.fill();
      ctx.shadowBlur = 0;
    }
    ctx.globalAlpha = 1.0;
  });

  if (hoverEpoch != null && hoverEpoch > 0 && hoverEpoch <= epochs) {
    const hx = tx(hoverEpoch);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(hx, P.t);
    ctx.lineTo(hx, H - P.b);
    ctx.stroke();
    ctx.setLineDash([]);
    
    Object.keys(series).forEach(opt => {
      const vals = series[opt];
      if (!vals?.length) return;
      const v = vals[hoverEpoch - 1];
      if (v != null && !isNaN(v)) {
        const { hex } = OPT_COLORS[opt] || { hex: '#fff' };
        const isHovered = hoverOpt === opt;
        const isMuted = hoverOpt && !isHovered;
        ctx.fillStyle = hex;
        ctx.globalAlpha = isMuted ? 0.3 : 1.0;
        ctx.beginPath();
        ctx.arc(hx, ty(v), isHovered ? 5 : 3, 0, 2*Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    });
  }
}

export default function PartBDashboard({ appMode }) {
  const isBeginner = appMode.includes('Beginner') || appMode.includes('Simple');
  const isSimple = isBeginner;

  const [opts,      setOpts]      = useState(['SGD','SGDMomentum','Adam','AdamW']);
  const [lr,        setLr]        = useState(0.01);
  const [epochs,    setEpochs]    = useState(100);
  const [batch,     setBatch]     = useState('32');
  const [beta,      setBeta]      = useState(0.9);
  const [beta1,     setBeta1]     = useState(0.9);
  const [beta2,     setBeta2]     = useState(0.999);
  const [wd,        setWd]        = useState(0.001);
  const [training,  setTraining]  = useState(false);
  const [pct,       setPct]       = useState(0);
  const [results,   setResults]   = useState(null);
  const [modal,     setModal]     = useState(false);
  const [error,     setError]     = useState(null);  // null | 'backend_offline' | 'unknown'
  const [hoverEpoch,setHoverEpoch]= useState(null);
  const [hoverOpt,  setHoverOpt]  = useState(null);
  
  const [playbackEpoch, setPlaybackEpoch] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const trainLossRef = useRef(null);
  const testLossRef  = useRef(null);
  const testAccRef   = useRef(null);
  const effLrRef     = useRef(null);

  const toggle = name => setOpts(p => p.includes(name) ? p.filter(o => o !== name) : [...p, name]);

  const handleTrain = async () => {
    if (!opts.length) return;
    setTraining(true); setPct(0); setError(null); setIsPlaying(false);
    const ticker = setInterval(() => setPct(p => Math.min(90, p + 2)), 800);
    try {
      const bs = batch === 'Full Batch' ? 455 : parseInt(batch);
      const res = await runBenchmarkTrain({ optimizers: opts, lr, epochs, batch_size: bs, beta, beta1, beta2, weight_decay: wd });
      setResults(res);
      setPlaybackEpoch(0);
      setIsPlaying(true);
    } catch (err) {
      console.error(err);
      // Detect network/backend errors vs other errors
      const isNetErr = err instanceof TypeError && err.message.toLowerCase().includes('fetch');
      setError(isNetErr ? 'backend_offline' : 'unknown');
    } finally {
      clearInterval(ticker);
      setPct(100);
      setTimeout(() => setTraining(false), 500);
    }
  };

  useEffect(() => {
    if (!results || !isPlaying || playbackEpoch >= epochs) {
      if (playbackEpoch >= epochs && isPlaying) setIsPlaying(false);
      return;
    }
    const delay = Math.max(10, 30 / playbackSpeed);
    const timer = setTimeout(() => {
      setPlaybackEpoch(p => Math.min(epochs, p + 1));
    }, delay);
    return () => clearTimeout(timer);
  }, [results, isPlaying, playbackEpoch, epochs, playbackSpeed]);

  useEffect(() => {
    if (!results) return;
    const r = results.results;
    const tl={}, vl={}, ta={}, el={};
    Object.keys(r).forEach(o => { 
      tl[o]=r[o].train_loss.slice(0, playbackEpoch); 
      vl[o]=r[o].test_loss.slice(0, playbackEpoch); 
      ta[o]=r[o].test_acc.slice(0, playbackEpoch); 
      if(r[o].eff_lr) el[o]=r[o].eff_lr.slice(0, playbackEpoch); 
    });
    drawChart(trainLossRef.current, tl, { isLog: true,  epochs, hoverEpoch, hoverOpt });
    drawChart(testLossRef.current,  vl, { isLog: true,  epochs, hoverEpoch, hoverOpt });
    drawChart(testAccRef.current,   ta, { isPercent: true, epochs, hoverEpoch, hoverOpt });
    if (!isSimple) drawChart(effLrRef.current, el, { isLog: true, epochs, hoverEpoch, hoverOpt });
  }, [results, isSimple, epochs, hoverEpoch, hoverOpt, playbackEpoch]);

  let bestAcc = null, minLoss = null, minEpoch = null;
  let accTies = [], lossTies = [], epochTies = [], observations = [];

  if (results) {
    bestAcc = Math.max(...results.summary_table.map(r => r.final_test_acc));
    minLoss = Math.min(...results.summary_table.map(r => r.final_test_loss));
    const epochs_valid = results.summary_table.map(r => r.convergence_epoch).filter(e => typeof e === 'number' && e > 0);
    minEpoch = epochs_valid.length > 0 ? Math.min(...epochs_valid) : null;
    
    accTies = results.summary_table.filter(r => r.final_test_acc === bestAcc).map(r => DISPLAY_NAMES[r.optimizer]);
    lossTies = results.summary_table.filter(r => r.final_test_loss === minLoss).map(r => DISPLAY_NAMES[r.optimizer]);
    epochTies = minEpoch ? results.summary_table.filter(r => r.convergence_epoch === minEpoch).map(r => DISPLAY_NAMES[r.optimizer]) : [];
    
    if (accTies.length === 1) observations.push(`${accTies[0]} achieved the highest test accuracy.`);
    else if (accTies.length > 1) observations.push(`${accTies.length} optimizers tied for the highest test accuracy.`);
    
    if (lossTies.length === 1) observations.push(`${lossTies[0]} achieved the lowest test loss.`);
    else if (lossTies.length > 1) observations.push(`${lossTies.length} optimizers tied for the lowest test loss.`);
    
    if (epochTies.length === 1) observations.push(`${epochTies[0]} converged the fastest (epoch ${minEpoch}).`);
    
    const overfit = results.summary_table.filter(r => (r.final_train_acc - r.final_test_acc) > 5).map(r => DISPLAY_NAMES[r.optimizer]);
    if (overfit.length > 0) observations.push(`${overfit.join(', ')} showed signs of overfitting (train accuracy > test accuracy by 5%+).`);
  }

  const handleMouseMove = (e) => {
    if (!results) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const W = rect.width;
    const P_l = 14, P_r = 14;
    let ep = 1 + Math.round((x - P_l) * Math.max(1, epochs - 1) / (W - P_l - P_r));
    ep = Math.max(1, Math.min(epochs, ep));
    setHoverEpoch(ep);
  };

  const META = [
    { icon: '⬡', label: 'Features',       value: '30',       color: '#8B5CF6' },
    { icon: '📊', label: 'Train Samples',  value: '455',      color: '#6366F1' },
    { icon: '🔬', label: 'Test Samples',   value: '114',      color: '#06B6D4' },
    { icon: '✂', label: 'Train/Test Split', value: '80/20 %', color: '#10B981' },
  ];

  let statusText = 'Configure your optimizers and run the benchmark.';
  if (error) statusText = error === 'backend_offline' ? 'Backend offline — start the Python server first.' : 'Benchmark failed — check console for details.';
  else if (training) statusText = `Training ${opts.length} optimizers...`;
  else if (results) statusText = `Benchmark complete · ${results.summary_table.length} optimizers`;

  return (
    <div className="playground-layout">

      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className="playground-sidebar sidebar">
        <div>
          <div className="section-header">
            <span className="section-title">OPTIMIZERS</span>
            <div className="section-line" />
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {ALL_OPTS.map(name => {
              const active = opts.includes(name);
              const { hex, rgb } = OPT_COLORS[name];
              return (
                <button
                  key={name}
                  className={`opt-pill ${active ? 'active' : ''}`}
                  style={active ? { color: hex, borderColor: `rgba(${rgb},0.5)`, background: `rgba(${rgb},0.12)`, boxShadow: `0 0 10px rgba(${rgb},0.2)` } : {}}
                  onClick={() => toggle(name)}
                >
                  <span style={{ fontSize: 8 }}>{active ? '●' : '○'}</span>
                  {DISPLAY_NAMES[name]}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <div className="section-header">
            <span className="section-title">TRAINING</span>
            <div className="section-line" />
          </div>
          
          <div style={{ marginBottom: 12 }}>
            <div className="param-row"><span className="param-label">Learning Rate</span></div>
            <select className="hud-select" value={lr} onChange={e => setLr(parseFloat(e.target.value))} style={{ marginBottom: 4 }}>
              {LR_OPTS.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
            <div style={{ fontSize: 10, color: 'var(--text-sec)', lineHeight: 1.3 }}>Controls the size of each optimization update.</div>
          </div>

          {[
            { label: 'Epochs', val: epochs, set: setEpochs, type: 'number', help: 'One complete pass through the training dataset.' },
          ].map(({ label, val, set, help }) => (
            <div key={label} style={{ marginBottom: 12 }}>
              <div className="param-row">
                <span className="param-label">{label}</span>
                <span className="param-value">{val}</span>
              </div>
              <input className="hud-input" type="number" value={val} min={10} max={500} step={10}
                onChange={e => set(parseInt(e.target.value)||10)} style={{ marginBottom: 4 }} />
              <div style={{ fontSize: 10, color: 'var(--text-sec)', lineHeight: 1.3 }}>{help}</div>
            </div>
          ))}

          <div style={{ marginBottom: 12 }}>
            <div className="param-row"><span className="param-label">Batch Size</span></div>
            <select className="hud-select" value={batch} onChange={e => setBatch(e.target.value)} style={{ marginBottom: 4 }}>
              {['32','64','128','Full Batch'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
            <div style={{ fontSize: 10, color: 'var(--text-sec)', lineHeight: 1.3 }}>Number of training examples processed before updating the model.</div>
          </div>
        </div>

        {!isSimple && (
          <div>
            <div className="section-header">
              <span className="section-title">HYPERPARAMETERS</span>
              <div className="section-line" />
            </div>
            {[
              { label: 'β',  val: beta,  set: setBeta,  min:0, max:0.999, step:0.01  },
              { label: 'β₁', val: beta1, set: setBeta1, min:0, max:0.999, step:0.01  },
              { label: 'β₂', val: beta2, set: setBeta2, min:0, max:0.999, step:0.001 },
              { label: 'λ',  val: wd,    set: setWd,    min:0, max:0.1,   step:0.001 },
            ].map(({ label, val, set, min, max, step }) => (
              <div key={label} style={{ marginBottom: 12 }}>
                <div className="param-row">
                  <span className="param-label">{label}</span>
                  <span className="param-value">{val}</span>
                </div>
                <input className="hud-range" type="range" min={min} max={max} step={step} value={val}
                  onChange={e => set(parseFloat(e.target.value))} />
              </div>
            ))}
          </div>
        )}

        <div>
          <div className="section-header">
            <span className="section-title">RUN</span>
            <div className="section-line" />
          </div>
          {training && (
            <div style={{ marginBottom: 12 }}>
              <div className="param-row">
                <span className="param-label">Training…</span>
                <span className="param-value">{pct}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          )}

          <button className="btn-primary" style={{ width: '100%', padding: '11px 0' }}
            onClick={handleTrain} disabled={training}>
            {training ? '⏳  Training…' : '🚀  Run Benchmark'}
          </button>
        </div>

        <div style={{ marginTop: 16 }}>
          <div className="section-header">
            <span className="section-title">HELP</span>
            <div className="section-line" />
          </div>
          <button className="btn-secondary" style={{ width: '100%', padding: '9px 0' }}
            onClick={() => setModal(true)}>
            📖  Guide
          </button>
        </div>
      </aside>

      {/* ── Main ──────────────────────────────────── */}
      <section className="playground-content">
      
        <div style={{ marginBottom: 20 }}>
          <h2 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 4px 0', letterSpacing: '-0.5px' }}>Which optimizer learns best?</h2>
          <p style={{ margin: 0, color: 'var(--text-sec)', fontSize: 14 }}>Compare convergence, validation loss, and final performance under the same training conditions.</p>
        </div>

        {/* Metadata cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
          {META.map(({ icon, label, value, color }) => (
            <div key={label} className="metric-card">
              <div style={{ fontSize: 20, marginBottom: 8 }}>{icon}</div>
              <div className="metric-value" style={{ background: `linear-gradient(135deg, #F1F5F9, ${color})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {value}
              </div>
              <div className="metric-label">{label}</div>
              {/* Color accent bar */}
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${color}, transparent)`, opacity: 0.5 }} />
            </div>
          ))}
        </div>

        {/* Status strip */}
        <div className="telemetry-strip" style={{ alignItems: 'center' }}>
          <div className="live-dot" style={{
            background: error ? '#EF4444' : training ? '#F59E0B' : results ? '#10B981' : '#6366F1',
            boxShadow: `0 0 6px ${error ? '#EF4444' : training ? '#F59E0B' : results ? '#10B981' : '#6366F1'}`,
          }} />
          <span className="tele-label">STATUS</span>
          <span className="tele-value">{statusText.toUpperCase()}</span>
          {results && bestAcc && !error && (
            <>
              <span className="tele-sep">|</span>
              <span className="tele-label">BEST TEST ACC</span>
              <span className="tele-value" style={{ color: '#34D399', textShadow: '0 0 8px rgba(52,211,153,0.5)' }}>
                {bestAcc.toFixed(2)}%
              </span>
            </>
          )}
          <div style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: 'var(--text-sec)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ padding: '2px 6px', background: 'rgba(255,255,255,0.05)', borderRadius: 4, color: '#A78BFA' }}>Controlled comparison</span>
            Same dataset · Same learning rate · Same epochs · Same batch size
          </span>
        </div>

        {/* Offline error banner */}
        {error === 'backend_offline' && (
          <div style={{
            marginBottom: 12, padding: '14px 18px',
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.35)',
            borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'flex-start', gap: 12,
          }}>
            <span style={{ fontSize: 20 }}>⚠️</span>
            <div>
              <div style={{ fontWeight: 700, color: '#FCA5A5', marginBottom: 4 }}>Backend server is not running</div>
              <div style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.6 }}>
                Part B requires the Python FastAPI backend to train the neural network.
                Part A works offline because it uses a built-in JavaScript fallback.<br />
                <strong style={{ color: '#FCD34D' }}>To start the backend:</strong> open a terminal in the{' '}
                <code style={{ fontFamily: 'JetBrains Mono', background: 'rgba(0,0,0,0.4)', padding: '1px 5px', borderRadius: 3 }}>backend/</code>{' '}
                folder and run:
                <br />
                <code style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: '#6EE7B7', background: 'rgba(0,0,0,0.4)', padding: '4px 8px', borderRadius: 3, display: 'inline-block', marginTop: 6 }}>
                  uvicorn main:app --reload --port 8000
                </code>
              </div>
            </div>
          </div>
        )}
        {error === 'unknown' && (
          <div style={{
            marginBottom: 12, padding: '12px 16px',
            background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 'var(--radius-md)', fontSize: 12, color: '#FCA5A5',
          }}>
            ❌ Benchmark failed. Check the browser console for details.
          </div>
        )}

        {results && (
          <div className="glass-card" style={{ padding: '8px 14px', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 12 }}>
             <button 
                onClick={() => {
                  if (playbackEpoch >= epochs) { setPlaybackEpoch(0); setIsPlaying(true); }
                  else setIsPlaying(!isPlaying);
                }} 
                className="btn-secondary" style={{ padding: '4px 12px' }}>
                {playbackEpoch >= epochs ? '↻ Replay' : (isPlaying ? 'Ⅱ Pause' : '▶ Play')}
             </button>
             <button 
                onClick={() => { setIsPlaying(false); setPlaybackEpoch(p => Math.min(epochs, p + 1)); }} 
                disabled={isPlaying || playbackEpoch >= epochs} 
                className="btn-secondary" style={{ padding: '4px 12px' }}>
                → Step
             </button>
             {/* Skip to end — reveals the summary table immediately */}
             <button
                onClick={() => { setIsPlaying(false); setPlaybackEpoch(epochs); }}
                disabled={playbackEpoch >= epochs}
                className="btn-secondary" style={{ padding: '4px 12px' }}>
                ⏭ Skip to End
             </button>
             
             <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10 }}>
               <input type="range" min={0} max={epochs} value={playbackEpoch} 
                  onChange={(e) => { setIsPlaying(false); setPlaybackEpoch(parseInt(e.target.value)); }} 
                  style={{ flex: 1 }} className="hud-range" />
               <span style={{ fontSize: 12, color: 'var(--text-sec)', fontFamily: 'JetBrains Mono', minWidth: 60, textAlign: 'right' }}>
                 {playbackEpoch} / {epochs}
               </span>
             </div>

             <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
               <span style={{ fontSize: 10, color: 'var(--text-sec)', textTransform: 'uppercase' }}>Speed</span>
               <select className="hud-select" value={playbackSpeed} onChange={e => setPlaybackSpeed(parseFloat(e.target.value))} style={{ width: 60, padding: '2px 4px' }}>
                 <option value={0.5}>0.5×</option>
                 <option value={1}>1×</option>
                 <option value={2}>2×</option>
                 <option value={4}>4×</option>
               </select>
             </div>
          </div>
        )}

        {/* Charts */}
        <div style={{ display: 'grid', gridTemplateColumns: isSimple ? '1fr 1fr 1fr' : '1fr 1fr', gap: 10 }}>
          {[
            { ref: trainLossRef, title: 'Training Loss (BCE)', desc: 'How well the model fits the training data.' },
            { ref: testLossRef,  title: 'Test Loss (BCE)', desc: 'How well the model performs on held-out test data.' },
            { ref: testAccRef,   title: 'Test Accuracy (%)', desc: 'How often the model predicts the correct class.' },
            ...(!isSimple ? [{ ref: effLrRef, title: 'Effective LR — ηₑff', desc: 'Actual per-parameter step size.' }] : []),
          ].map(({ ref, title, desc }) => (
            <div key={title} className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ padding: '10px 14px 8px', borderBottom: '1px solid var(--border)' }}>
                <div className="section-header" style={{ marginBottom: 2 }}>
                  <span className="section-title">{title}</span>
                  <div className="section-line" />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-sec)', minHeight: 16 }}>{desc}</div>
              </div>
              <div style={{ position: 'relative' }} onMouseMove={handleMouseMove} onMouseLeave={() => setHoverEpoch(null)}>
                <canvas ref={ref} width={860} height={380}
                  style={{ display: 'block', width: '100%', height: 'auto', background: 'rgba(3,5,13,0.5)', cursor: 'crosshair' }} />
                <div style={{ position: 'absolute', bottom: 6, left: 10, display: 'flex', flexWrap: 'wrap', gap: 6, pointerEvents: 'none' }}>
                  {opts.map(name => (
                    <span key={name} style={{ fontFamily: 'JetBrains Mono', fontSize: 9,
                      color: OPT_COLORS[name]?.hex,
                      opacity: hoverOpt && hoverOpt !== name ? 0.3 : 1,
                      textShadow: hoverOpt === name ? `0 0 6px ${OPT_COLORS[name]?.hex}` : 'none',
                      transition: 'opacity 0.2s' }}>
                      ● {DISPLAY_NAMES[name]}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Summary table */}
        {results?.summary_table && playbackEpoch >= epochs && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 10 }}>
              <div className="glass-card" style={{ padding: '12px 16px' }}>
                <div style={{ fontSize: 11, color: 'var(--text-sec)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>Best Test Accuracy</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#34D399' }}>
                  {bestAcc?.toFixed(2)}% <span style={{ fontSize: 12, color: 'var(--text-sec)', fontWeight: 400 }}>· {accTies.length === 1 ? accTies[0] : `${accTies.length} tied`}</span>
                </div>
              </div>
              <div className="glass-card" style={{ padding: '12px 16px' }}>
                <div style={{ fontSize: 11, color: 'var(--text-sec)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>Fastest Convergence</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#60A5FA' }}>
                  {minEpoch ? `${minEpoch} epochs` : 'N/A'} <span style={{ fontSize: 12, color: 'var(--text-sec)', fontWeight: 400 }}>· {epochTies.length === 1 ? epochTies[0] : `${epochTies.length} tied`}</span>
                </div>
              </div>
              <div className="glass-card" style={{ padding: '12px 16px' }}>
                <div style={{ fontSize: 11, color: 'var(--text-sec)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>Lowest Test Loss</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#F472B6' }}>
                  {minLoss?.toFixed(4)} <span style={{ fontSize: 12, color: 'var(--text-sec)', fontWeight: 400 }}>· {lossTies.length === 1 ? lossTies[0] : `${lossTies.length} tied`}</span>
                </div>
              </div>
            </div>

            <div className="glass-card" style={{ padding: 14 }}>
              <div className="section-header">
                <span className="section-title">Benchmark Results</span>
                <div className="section-line" />
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      {['Optimizer','Train Loss','Test Loss','Train Acc %','Test Acc %','Conv. Epoch','Status'].map(h => (
                        <th key={h}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.summary_table.map(row => {
                      const isBest = row.final_test_acc === bestAcc;
                      const { hex } = OPT_COLORS[row.optimizer] || {};
                      const isHovered = hoverOpt === row.optimizer;
                      const status = (row.convergence_epoch === 'Diverged' || isNaN(row.final_test_loss) || row.final_test_loss > 100)
                        ? <span style={{ color: '#EF4444' }}>Unstable</span>
                        : <span style={{ color: '#10B981' }}>Stable</span>;

                      return (
                        <tr key={row.optimizer}
                            onMouseEnter={() => setHoverOpt(row.optimizer)}
                            onMouseLeave={() => setHoverOpt(null)}
                            style={{ background: isHovered ? 'rgba(255,255,255,0.05)' : 'transparent', cursor: 'default' }}
                        >
                          <td className="opt-name" style={{ color: hex }}>
                            {isBest ? '★ ' : ''}{DISPLAY_NAMES[row.optimizer]}
                          </td>
                        <td>{row.final_train_loss.toFixed(4)}</td>
                        <td>{row.final_test_loss.toFixed(4)}</td>
                        <td>{row.final_train_acc.toFixed(2)}%</td>
                        <td className={isBest ? 'best' : ''}>{row.final_test_acc.toFixed(2)}%</td>
                        <td>{row.convergence_epoch}</td>
                        <td>{status}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="glass-card" style={{ padding: 14, marginTop: 10, borderLeft: '4px solid #8B5CF6' }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#E2E8F0' }}>What did we learn?</div>
              <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--text-sec)', fontSize: 13, lineHeight: 1.6 }}>
                {observations.map((obs, idx) => <li key={idx}>{obs}</li>)}
              </ul>
            </div>
          </>
        )}
      </section>

      {/* ── Modal ─────────────────────────────────── */}
      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(false)}>
          <div className="modal-panel" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ fontWeight: 700, fontSize: 15 }}>📖 How to Use — Part B</div>
              <button className="btn-secondary" style={{ padding: '5px 12px', fontSize: 12 }} onClick={() => setModal(false)}>
                ✕ Close
              </button>
            </div>
            <div className="modal-body">
              {[
                { title: 'Dataset', body: 'Breast Cancer Wisconsin — 569 samples, 30 features, binary target. Standardized with StandardScaler (80/20 stratified split).' },
                { title: 'Architecture', body: 'Input(30) → Dense(16) → ReLU → Dense(8) → ReLU → Dense(1) → Sigmoid. Pure NumPy, no autograd framework.' },
                { title: 'Effective LR', body: 'For adaptive methods: ηₑff = η / (√v̂ₜ + ε). Tracks actual per-parameter step size after variance scaling.' },
                { title: 'Convergence Epoch', body: 'First epoch where validation loss stays within 1% of its final value for all subsequent epochs.' },
                { title: 'Workflow', body: 'Select optimizers → configure hyperparameters → click Run Benchmark → inspect charts → compare in summary table.' },
              ].map(({ title, body }) => (
                <div key={title} className="modal-info-card">
                  <div className="modal-info-title">{title}</div>
                  {body}
                </div>
              ))}
            </div>
            <div style={{ padding: '0 24px 20px', textAlign: 'right' }}>
              <button className="btn-primary" style={{ padding: '10px 24px' }} onClick={() => setModal(false)}>
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
