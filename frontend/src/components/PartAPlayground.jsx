import React, { useState, useEffect, useRef } from 'react';
import { fetchTrajectoryData } from '../api';

const OPTIMIZER_COLORS = {
  SGD: '#ef4444',
  SGDMomentum: '#f97316',
  Momentum: '#f97316',
  NAG: '#eab308',
  AdaGrad: '#22c55e',
  RMSProp: '#3b82f6',
  Adam: '#a855f7',
  AdamW: '#b45309',
};

const ALL_OPTIMIZERS = ['SGD', 'SGDMomentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW'];

const SURFACES = {
  'L1: c = 10': { c: 10.0, label: 'L1: c = 10' },
  'L2: c = 50 (default)': { c: 50.0, label: 'L2: c = 50 (default)' },
  'L3: c = 100': { c: 100.0, label: 'L3: c = 100' },
  'L4: c = 1000': { c: 1000.0, label: 'L4: c = 1000' },
};

const LOG_LR_OPTIONS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];

const OPTIMIZER_EXPLANATIONS = {
  SGD: "Stochastic Gradient Descent updates parameters directly along steepest descent. High condition numbers cause violent cross-valley oscillations.",
  SGDMomentum: "Momentum accumulates past velocity to accelerate progress down consistent slopes and dampen orthogonal bouncing.",
  NAG: "Nesterov Accelerated Gradient computes gradients at a look-ahead position (theta - beta*v), applying predictive braking before overshooting.",
  AdaGrad: "AdaGrad tracks accumulated squared gradients to adaptively shrink step sizes on frequently updated parameters.",
  RMSProp: "RMSProp replaces AdaGrad's unbounded sum with an exponential moving average, maintaining active learning rates.",
  Adam: "Adam combines momentum (1st moment m) and RMSProp variance tracking (2nd moment v) with bias-correction factors.",
  AdamW: "AdamW decouples L2 weight decay from adaptive gradient scaling, preserving true scale-invariant regularization.",
};

export default function PartAPlayground({ appMode }) {
  const isSimple = appMode === 'Simple mode';

  // Control states
  const [selectedSurface, setSelectedSurface] = useState('L2: c = 50 (default)');
  const [selectedOpts, setSelectedOpts] = useState(['SGD', 'SGDMomentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW']);
  const [lr, setLr] = useState(0.01);
  const [x0, setX0] = useState(8.0);
  const [y0, setY0] = useState(8.0);
  const [beta, setBeta] = useState(0.9);
  const [beta1, setBeta1] = useState(0.9);
  const [beta2, setBeta2] = useState(0.999);
  const [weightDecay, setWeightDecay] = useState(0.001);
  const [showGradArrow, setShowGradArrow] = useState(true);

  // Playback states
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(50);
  const [data, setData] = useState(null);
  const [explainOpt, setExplainOpt] = useState('NAG');
  const [explainOpen, setExplainOpen] = useState(false);

  const contourCanvasRef = useRef(null);
  const lossCanvasRef = useRef(null);

  // Fetch trajectory data from FastAPI backend
  useEffect(() => {
    async function loadTrajectory() {
      try {
        const res = await fetchTrajectoryData({
          surface_name: selectedSurface,
          optimizers: selectedOpts,
          x0,
          y0,
          lr,
          beta,
          beta1,
          beta2,
          weight_decay: weightDecay,
          max_steps: 500,
        });
        setData(res);
        setCurrentStep(0);
      } catch (err) {
        console.error("Failed to load trajectory:", err);
      }
    }
    if (selectedOpts.length > 0) {
      loadTrajectory();
    }
  }, [selectedSurface, selectedOpts, lr, x0, y0, beta, beta1, beta2, weightDecay]);

  // Animation Loop Execution
  useEffect(() => {
    let timer = null;
    if (isPlaying && currentStep < 500) {
      const stepsPerFrame = Math.max(1, Math.round(100.0 / speedMs));
      const delay = speedMs * stepsPerFrame;
      timer = setTimeout(() => {
        setCurrentStep(prev => Math.min(500, prev + stepsPerFrame));
      }, delay);
    } else if (currentStep >= 500) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, speedMs]);

  // Draw View 1: 2D Contour Map
  useEffect(() => {
    if (!data || !contourCanvasRef.current) return;
    const canvas = contourCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const max_x = data.max_x || 10.0;
    const max_y = data.max_y || 10.0;
    const c_val = SURFACES[selectedSurface]?.c || 50.0;

    // Helper coordinate mapping
    const toCanvasX = x => ((x + max_x) / (2 * max_x)) * width;
    const toCanvasY = y => height - ((y + max_y) / (2 * max_y)) * height;

    // Render Contour Gradient Background
    const imgData = ctx.createImageData(width, height);
    for (let py = 0; py < height; py++) {
      const y = max_y - (py / height) * (2 * max_y);
      for (let px = 0; px < width; px++) {
        const x = -max_x + (px / width) * (2 * max_x);
        const loss = x * x + c_val * y * y;
        const normLoss = Math.min(1.0, Math.log1p(loss) / 8.0);
        
        // Viridis-like colormap simulation
        const r = Math.floor(255 * (0.2 + 0.8 * normLoss));
        const g = Math.floor(255 * (0.1 + 0.7 * (1 - normLoss)));
        const b = Math.floor(255 * (0.4 + 0.5 * Math.sin(normLoss * Math.PI)));
        
        const idx = (py * width + px) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 220;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw Grid Lines
    ctx.strokeStyle = '#2E3742';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.stroke();

    // Draw Minimum Star (0,0)
    const minX = toCanvasX(0);
    const minY = toCanvasY(0);
    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.arc(minX, minY, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Draw Trajectories for active optimizers
    selectedOpts.forEach(name => {
      const traj = data.trajectories[name];
      if (!traj || traj.length === 0) return;

      const color = OPTIMIZER_COLORS[name] || '#ffffff';
      const validTraj = traj.slice(0, currentStep + 1);

      // Fading Trail (up to last 30 steps)
      const trailLen = 30;
      const recentTraj = validTraj.slice(Math.max(0, validTraj.length - trailLen));

      if (recentTraj.length > 1) {
        for (let i = 0; i < recentTraj.length - 1; i++) {
          const pt1 = recentTraj[i];
          const pt2 = recentTraj[i + 1];
          const alpha = 0.15 + 0.8 * (i / Math.max(1, recentTraj.length - 1));

          ctx.strokeStyle = color;
          ctx.globalAlpha = alpha;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(toCanvasX(pt1[0]), toCanvasY(pt1[1]));
          ctx.lineTo(toCanvasX(pt2[0]), toCanvasY(pt2[1]));
          ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
      }

      // Draw current position dot
      const currPos = validTraj[validTraj.length - 1];
      if (currPos && !isNaN(currPos[0]) && !isNaN(currPos[1])) {
        const cx = toCanvasX(currPos[0]);
        const cy = toCanvasY(currPos[1]);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx, cy, 5.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // Draw gradient direction arrow if enabled
        if (showGradArrow) {
          const gx = -2.0 * currPos[0];
          const gy = -2.0 * c_val * currPos[1];
          const gNorm = Math.hypot(gx, gy);

          if (gNorm > 1e-6) {
            const arrowLen = 25;
            const dx = (gx / gNorm) * arrowLen;
            const dy = -(gy / gNorm) * arrowLen;

            ctx.strokeStyle = color;
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + dx, cy + dy);
            ctx.stroke();
          }
        }
      }
    });
  }, [data, currentStep, selectedSurface, selectedOpts, showGradArrow]);

  // Draw View 2: Synchronized Loss Reduction Plot
  useEffect(() => {
    if (!data || !lossCanvasRef.current) return;
    const canvas = lossCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#1A2029';
    ctx.fillRect(0, 0, width, height);

    // Gridlines
    ctx.strokeStyle = '#2E3742';
    ctx.lineWidth = 0.5;
    for (let y = 0.2; y < 1.0; y += 0.2) {
      ctx.beginPath();
      ctx.moveTo(40, y * height);
      ctx.lineTo(width - 10, y * height);
      ctx.stroke();
    }

    const maxSteps = 500;
    const toCanvasX = step => 40 + (step / maxSteps) * (width - 50);
    const toCanvasY = loss => {
      if (loss == null || isNaN(loss) || loss <= 0) return height - 20;
      const logLoss = Math.log10(Math.max(1e-6, loss));
      const minLog = -6;
      const maxLog = 4;
      const norm = (logLoss - minLog) / (maxLog - minLog);
      return height - 20 - Math.max(0, Math.min(1, norm)) * (height - 40);
    };

    // Draw loss curves
    selectedOpts.forEach(name => {
      const losses = data.losses[name];
      if (!losses) return;

      const color = OPTIMIZER_COLORS[name] || '#ffffff';
      const validLosses = losses.slice(0, currentStep + 1);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.0;
      ctx.beginPath();

      let started = false;
      validLosses.forEach((lossVal, step) => {
        if (lossVal != null && !isNaN(lossVal)) {
          const x = toCanvasX(step);
          const y = toCanvasY(lossVal);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
      });
      ctx.stroke();

      // Current step marker dot
      if (validLosses.length > 0) {
        const lastLoss = validLosses[validLosses.length - 1];
        if (lastLoss != null && !isNaN(lastLoss)) {
          const cx = toCanvasX(currentStep);
          const cy = toCanvasY(lastLoss);
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    });
  }, [data, currentStep, selectedOpts]);

  // Telemetry strip values
  const firstOpt = selectedOpts[0];
  const firstTraj = data && data.trajectories[firstOpt] ? data.trajectories[firstOpt] : null;
  const firstLosses = data && data.losses[firstOpt] ? data.losses[firstOpt] : null;
  const posAtStep = firstTraj && firstTraj[currentStep] ? firstTraj[currentStep] : [0, 0];
  const lossAtStep = firstLosses && firstLosses[currentStep] != null ? firstLosses[currentStep] : null;

  const c_val = SURFACES[selectedSurface]?.c || 50.0;
  const gradNorm = posAtStep ? Math.hypot(2.0 * posAtStep[0], 2.0 * c_val * posAtStep[1]) : 0;

  const toggleOpt = name => {
    if (selectedOpts.includes(name)) {
      setSelectedOpts(selectedOpts.filter(o => o !== name));
    } else {
      setSelectedOpts([...selectedOpts, name]);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Sidebar Controls */}
      <aside className="col-span-3 bezel-card p-4 space-y-4">
        <div className="label-smallcaps">Part A Settings</div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Loss Surface</label>
          <select
            value={selectedSurface}
            onChange={e => setSelectedSurface(e.target.value)}
            className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1.5 text-xs text-white"
          >
            {Object.keys(SURFACES).map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Optimizers to Overlay</label>
          <div className="flex flex-wrap gap-1.5">
            {ALL_OPTIMIZERS.map(opt => {
              const active = selectedOpts.includes(opt);
              return (
                <button
                  key={opt}
                  onClick={() => toggleOpt(opt)}
                  style={{ backgroundColor: active ? OPTIMIZER_COLORS[opt] : '#1A2029', color: active ? '#10141A' : '#8B96A3' }}
                  className="px-2 py-0.5 text-[11px] font-bold rounded border border-[#2E3742] transition"
                >
                  {opt}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Learning Rate (\eta): {lr}</label>
          <select
            value={lr}
            onChange={e => setLr(parseFloat(e.target.value))}
            className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1.5 text-xs text-white font-mono-num"
          >
            {LOG_LR_OPTIONS.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>

        {!isSimple && (
          <div className="space-y-3 pt-2 border-t border-[#2E3742]">
            <div className="label-smallcaps">Hyperparameters</div>
            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\beta (Momentum / RMSProp): {beta}</label>
              <input
                type="range"
                min="0.0"
                max="0.999"
                step="0.01"
                value={beta}
                onChange={e => setBeta(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\beta_1 (Adam / AdamW): {beta1}</label>
              <input
                type="range"
                min="0.0"
                max="0.999"
                step="0.01"
                value={beta1}
                onChange={e => setBeta1(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\beta_2 (Adam / AdamW): {beta2}</label>
              <input
                type="range"
                min="0.0"
                max="0.999"
                step="0.001"
                value={beta2}
                onChange={e => setBeta2(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\lambda (AdamW Weight Decay): {weightDecay}</label>
              <input
                type="range"
                min="0.0"
                max="0.1"
                step="0.001"
                value={weightDecay}
                onChange={e => setWeightDecay(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#2E3742]">
          <div>
            <label className="text-xs text-[#8B96A3] block mb-1">Start x_0</label>
            <input
              type="number"
              value={x0}
              onChange={e => setX0(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1 text-xs font-mono-num"
            />
          </div>
          <div>
            <label className="text-xs text-[#8B96A3] block mb-1">Start y_0</label>
            <input
              type="number"
              value={y0}
              onChange={e => setY0(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1 text-xs font-mono-num"
            />
          </div>
        </div>

        <div className="pt-2 border-t border-[#2E3742]">
          <div className="label-smallcaps mb-2">Animation Controls</div>
          <div className="grid grid-cols-4 gap-1 mb-2">
            <button
              onClick={() => setIsPlaying(true)}
              className="btn-primary py-1 text-xs"
            >
              ▶
            </button>
            <button
              onClick={() => setIsPlaying(false)}
              className="btn-secondary py-1 text-xs"
            >
              ⏸
            </button>
            <button
              onClick={() => { setIsPlaying(false); setCurrentStep(prev => Math.min(500, prev + 1)); }}
              className="btn-secondary py-1 text-xs"
            >
              ⏭
            </button>
            <button
              onClick={() => { setIsPlaying(false); setCurrentStep(0); }}
              className="btn-secondary py-1 text-xs"
            >
              ↺
            </button>
          </div>

          <div>
            <label className="text-xs text-[#8B96A3] block mb-1">Step: {currentStep} / 500</label>
            <input
              type="range"
              min="0"
              max="500"
              value={currentStep}
              onChange={e => setCurrentStep(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="mt-2">
            <label className="text-xs text-[#8B96A3] block mb-1">Speed (ms/step): {speedMs}</label>
            <input
              type="range"
              min="10"
              max="500"
              step="10"
              value={speedMs}
              onChange={e => setSpeedMs(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </aside>

      {/* Main Visualizer Area */}
      <section className="col-span-9 space-y-4">
        {/* Telemetry Readout Strip */}
        <div className="bg-[#13171F] border border-[#2E3742] rounded p-3 flex items-center justify-between font-mono-num text-xs">
          <div className="flex items-center gap-4">
            <div>
              <span className="text-[#8B96A3] font-semibold tracking-wider">ITER:</span>{' '}
              <span className="text-[#E3A23B] font-bold">{currentStep}/500</span>
            </div>
            <span className="text-[#2E3742]">|</span>
            {firstOpt && (
              <>
                <div>
                  <span className="text-[#8B96A3] font-semibold tracking-wider">{firstOpt} LOSS:</span>{' '}
                  <span className="text-[#E3A23B] font-bold">
                    {lossAtStep != null ? lossAtStep.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <span className="text-[#2E3742]">|</span>
                <div>
                  <span className="text-[#8B96A3] font-semibold tracking-wider">{firstOpt} GRAD NORM:</span>{' '}
                  <span className="text-[#E3A23B] font-bold">{gradNorm.toFixed(4)}</span>
                </div>
              </>
            )}
          </div>

          <label className="flex items-center gap-2 text-xs text-[#8B96A3]">
            <input
              type="checkbox"
              checked={showGradArrow}
              onChange={e => setShowGradArrow(e.target.checked)}
            />
            Show Gradient Direction (Downhill Arrow)
          </label>
        </div>

        {/* Synchronized Dual View */}
        <div className="grid grid-cols-2 gap-4 bezel-card p-4">
          <div>
            <h3 className="text-xs font-bold text-[#8B96A3] uppercase tracking-wider mb-2">
              View 1: Contour Trajectory Plot ({selectedSurface})
            </h3>
            <canvas
              ref={contourCanvasRef}
              width={380}
              height={320}
              className="border border-[#2E3742] rounded w-full bg-[#10141A]"
            />
          </div>

          <div>
            <h3 className="text-xs font-bold text-[#8B96A3] uppercase tracking-wider mb-2">
              View 2: Loss Reduction vs Iteration (Log Scale)
            </h3>
            <canvas
              ref={lossCanvasRef}
              width={380}
              height={320}
              className="border border-[#2E3742] rounded w-full bg-[#10141A]"
            />
          </div>
        </div>

        {/* Explain-As-You-Go Panel */}
        <div className="bezel-card p-4 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-[#E3A23B] flex items-center gap-2">
              💡 Explain-As-You-Go (Algorithm Mechanics)
            </h3>
            <button
              onClick={() => setExplainOpen(!explainOpen)}
              className="text-xs text-[#8B96A3] hover:text-white"
            >
              {explainOpen ? '▲ Collapse' : '▼ Expand'}
            </button>
          </div>

          {explainOpen && (
            <div className="space-y-3 pt-2 border-t border-[#2E3742]">
              <div className="flex gap-2">
                {Object.keys(OPTIMIZER_EXPLANATIONS).map(opt => (
                  <button
                    key={opt}
                    onClick={() => setExplainOpt(opt)}
                    className={`px-2 py-1 text-xs rounded border ${explainOpt === opt ? 'border-[#E3A23B] text-[#E3A23B] bg-[#13171F]' : 'border-[#2E3742] text-[#8B96A3]'}`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
              <p className="text-xs text-[#E7EAEE] bg-[#13171F] p-3 rounded border border-[#2E3742] leading-relaxed">
                {OPTIMIZER_EXPLANATIONS[explainOpt]}
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
