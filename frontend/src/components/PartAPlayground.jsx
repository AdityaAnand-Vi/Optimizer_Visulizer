import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  RefreshCw,
  SkipForward,
  HelpCircle,
  Activity,
  Sparkles,
  Compass,
  BookOpen,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Sliders,
  Layers,
  Table,
  Info,
  Check,
  X,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { fetchTrajectoryData } from '../api';
import { fmtNumber, fmtPos, fmtLoss, fmtDelta } from '../utils/formatters';
import ConditioningExplorer from './ConditioningExplorer';
import ExplainAsYouGo from './ExplainAsYouGo';

/* ── Optimizer colour palette ────────────────────────────── */
const OPT_COLORS = {
  SGD:         { hex: '#FF6B6B', rgb: '255,107,107', dash: [] },
  SGDMomentum: { hex: '#FFB454', rgb: '255,180,84',  dash: [4, 4] },
  Momentum:    { hex: '#FFB454', rgb: '255,180,84',  dash: [4, 4] },
  NAG:         { hex: '#3DD9FF', rgb: '61,217,255',  dash: [2, 2] },
  AdaGrad:     { hex: '#49D99A', rgb: '73,217,154',  dash: [] },
  RMSProp:     { hex: '#2DD4BF', rgb: '45,212,191',  dash: [4, 4] },
  Adam:        { hex: '#7C6CFF', rgb: '124,108,255', dash: [] },
  AdamW:       { hex: '#F472B6', rgb: '244,114,182', dash: [8, 4] },
};

const ALL_OPTS = ['SGD', 'Momentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW'];

const SURFACES = {
  'L1: c = 10':           { c: 10, name: 'L₁: x² + 10y²', kappa: 10 },
  'L2: c = 50 (default)': { c: 50, name: 'L₂: x² + 50y²', kappa: 50 },
  'L3: c = 100':          { c: 100, name: 'L₃: x² + 100y²', kappa: 100 },
  'L4: c = 1000':         { c: 1000, name: 'L₄: x² + 1000y²', kappa: 1000 },
};

const LR_OPTS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];

const STEP_PRESETS = [
  { label: '100',             value: 100 },
  { label: '500 (default)',   value: 500 },
  { label: '1,000',          value: 1000 },
  { label: '2,000',          value: 2000 },
  { label: '5,000',          value: 5000 },
  { label: 'Until convergence', value: -1 },
];

const SAFETY_LIMIT = 100_000;



const OPT_TRAITS = {
  SGD:         { speed: 'Slow in ravines', adaptivity: 'None (Fixed)', weightDecay: 'None' },
  Momentum:    { speed: 'Fast acceleration', adaptivity: 'None (Velocity)', weightDecay: 'None' },
  SGDMomentum: { speed: 'Fast acceleration', adaptivity: 'None (Velocity)', weightDecay: 'None' },
  NAG:         { speed: 'Fast (Braking)', adaptivity: 'None (Look-ahead)', weightDecay: 'None' },
  AdaGrad:     { speed: 'Decaying (Stalls)', adaptivity: 'Coordinate-wise (Σg²)', weightDecay: 'None' },
  RMSProp:     { speed: 'Consistent Fast', adaptivity: 'Coordinate-wise (EMA)', weightDecay: 'None' },
  Adam:        { speed: 'Fastest', adaptivity: 'Dual Moments (m + v)', weightDecay: 'Folded L2' },
  AdamW:       { speed: 'Fastest', adaptivity: 'Dual Moments (m + v)', weightDecay: 'Decoupled (λ·θ)' },
};

/* ── Symbol Hover Tooltips Dictionary ───────────────────── */
const SYMBOL_DICT = {
  'η': { title: 'Learning Rate (η / Eta)', desc: 'Controls step size downhill. Too large causes divergence; too small makes training slow.' },
  'β': { title: 'Momentum Factor (β / Beta)', desc: 'Fraction of past velocity remembered (0.9 = 90% memory). Accelerates in steady directions.' },
  'β₁': { title: 'First Moment Decay (β₁)', desc: 'Adam memory factor for gradient direction (default 0.9).' },
  'β₂': { title: 'Second Moment Decay (β₂)', desc: 'Adam memory factor for gradient variance magnitude (default 0.999).' },
  'λ': { title: 'Weight Decay (λ / Lambda)', desc: 'Regularization shrinking parameter weights toward zero independently of gradient steps (AdamW).' },
  '∇L': { title: 'Gradient (∇L)', desc: 'Vector of steepest uphill slope. Optimizers step in opposite direction (−∇L) to reduce loss.' },
  '‖∇L‖': { title: 'Gradient Norm (‖∇L‖)', desc: 'Overall steepness of slope at current position. Reaches ~0 at the minimum.' },
  'θ': { title: 'Parameters (θ / Theta)', desc: 'Coordinates of the model. In 2D playground, θ = (x, y).' },
};

function SymbolTooltip({ sym, children }) {
  const info = SYMBOL_DICT[sym];
  if (!info) return children || sym;
  return (
    <span className="symbol-tooltip">
      {children || sym}
      <span className="tooltip-card">
        <span className="tooltip-title">{info.title}</span>
        <span>{info.desc}</span>
      </span>
    </span>
  );
}

/* ── Compact Coach Strip ─────────────────────────────────── */
function CompactCoachStrip({ onOpenGuide, onDismiss }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 14px',
        background: 'rgba(99, 102, 241, 0.08)',
        border: '1px solid rgba(99, 102, 241, 0.22)',
        borderRadius: 'var(--radius-md)',
        fontSize: 12,
        color: 'var(--text-sec)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 700, color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: 5 }}>
          <Sparkles className="w-3.5 h-3.5" /> Quick Guide:
        </span>
        <span>① Select optimizers</span>
        <span style={{ color: 'var(--text-muted)' }}>→</span>
        <span>② Press <strong>Run</strong> to watch descent</span>
        <span style={{ color: 'var(--text-muted)' }}>→</span>
        <span>③ Adjust learning rate <SymbolTooltip sym="η">η</SymbolTooltip></span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <button
          className="btn-secondary"
          style={{ padding: '3px 10px', fontSize: 11, color: '#A5B4FC', borderColor: 'rgba(99,102,241,0.3)' }}
          onClick={onOpenGuide}
        >
          Detailed Guide <ArrowRight className="w-3 h-3 ml-1" />
        </button>
        <button
          onClick={onDismiss}
          style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 13 }}
          title="Dismiss Coach"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

/* ── Contextual Experiment Cards ──────────────────────────── */
const EXPERIMENTS = [
  {
    id: 'nag-diverge',
    title: 'Can you make NAG diverge?',
    desc: 'High momentum peeking ahead on a steep ravine causes overshoots.',
    surface: 'L3: c = 100',
    lr: 0.05,
    opts: ['NAG', 'Momentum', 'Adam'],
  },
  {
    id: 'fastest-min',
    title: 'Which reaches minimum fastest?',
    desc: 'Compare convergence speed on standard convex surface.',
    surface: 'L2: c = 50 (default)',
    lr: 0.01,
    opts: ['SGD', 'Momentum', 'Adam'],
  },
  {
    id: 'ravine-sgd',
    title: 'Why does SGD struggle in ravines?',
    desc: 'Watch SGD bounce side-to-side across steep walls.',
    surface: 'L2: c = 50 (default)',
    lr: 0.01,
    opts: ['SGD', 'Momentum', 'Adam'],
  },
  {
    id: 'tiny-lr',
    title: 'What happens with a tiny learning rate?',
    desc: 'Observe slow crawling when step size is too small.',
    surface: 'L2: c = 50 (default)',
    lr: 0.0001,
    opts: ['SGD', 'Adam', 'AdamW'],
  },
];

function TryExperimentCards({ onApplyExperiment }) {
  const [open, setOpen] = useState(true);
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '12px 16px' }}>
      <div
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', marginBottom: open ? 10 : 0 }}
        onClick={() => setOpen(p => !p)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Compass className="w-4 h-4 text-indigo-400" />
          <span style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Try an Experiment — Guided Scenarios
          </span>
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{open ? '▲ Collapse' : '▼ Expand'}</span>
      </div>
      {open && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 10 }}>
          {EXPERIMENTS.map(exp => (
            <div
              key={exp.id}
              style={{
                background: 'var(--surface-elev)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 12px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: 8,
              }}
            >
              <div>
                <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-primary)', marginBottom: 3 }}>
                  {exp.title}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-sec)', lineHeight: 1.4 }}>
                  {exp.desc}
                </div>
              </div>
              <button
                className="btn-secondary"
                style={{ fontSize: 11, padding: '5px 10px', justifyContent: 'space-between', width: '100%', borderColor: 'rgba(99,102,241,0.3)' }}
                onClick={() => onApplyExperiment(exp)}
              >
                <span>Load Scenario</span>
                <ArrowRight className="w-3 h-3 text-indigo-400" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Step Math Block ──────────────────────────────────────── */
function StepMathBlock({ name, cur, prev, safeStep, data, surface, lr, beta, wd }) {
  const [showFormula, setShowFormula] = useState(false);
  const [showNumbers, setShowNumbers] = useState(false);

  const { hex } = OPT_COLORS[name] || { hex: '#7C6CFF' };
  const grad = data?.gradients?.[name]?.[safeStep] || [0, 0];
  const delta = data?.deltas?.[name]?.[safeStep] || [0, 0];
  const lossArr = data?.losses?.[name] || [];
  const curLoss = lossArr[safeStep];
  const prevLoss = lossArr[Math.max(0, safeStep - 1)];
  const dL = (curLoss != null && prevLoss != null) ? curLoss - prevLoss : null;
  const status = data?.statuses?.[name]?.[safeStep] || 'RUNNING';

  const dx = delta[0], dy = delta[1];
  const deltaMag = Math.hypot(dx, dy);
  const gx = grad[0], gy = grad[1];
  const gn = Math.hypot(gx, gy);

  const getWhyExplanation = () => {
    if (status === 'DIVERGED') return 'Step was too large for steep curvature, exploding parameters to infinity.';
    switch (name) {
      case 'SGD': return `Took a step of size ${fmtNumber(deltaMag)} directly opposite slope [${fmtNumber(gx)}, ${fmtNumber(gy)}].`;
      case 'Momentum': case 'SGDMomentum': return `Accelerated step to ${fmtNumber(deltaMag)} by combining slope with 90% velocity memory.`;
      case 'NAG': return `Peeked ahead before taking step of size ${fmtNumber(deltaMag)}, applying predictive braking.`;
      case 'AdaGrad': return `Scaled step to ${fmtNumber(deltaMag)} inversely by accumulated squared gradients √(G+ε).`;
      case 'RMSProp': return `Scaled step to ${fmtNumber(deltaMag)} using exponential moving average of squared gradients.`;
      case 'Adam': return `Adapted step to ${fmtNumber(deltaMag)} using 1st moment direction (m) and 2nd moment variance (v).`;
      case 'AdamW': return `Adapted step to ${fmtNumber(deltaMag)} with dual moments while shrinking weights uniformly by λ=${wd}.`;
      default: return `Took a step of size ${fmtNumber(deltaMag)} based on slope.`;
    }
  };

  return (
    <div className="step-math-block" style={{ borderLeft: `3px solid ${hex}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span className="step-math-opt" style={{ color: hex }}>{name} {status === 'DIVERGED' ? '(DIVERGED)' : ''}</span>
        {dL != null && (
          <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: dL <= 0 ? '#49D99A' : '#FF667D', fontWeight: 600 }}>
            Loss {dL <= 0 ? '↓' : '↑'} {fmtLoss(Math.abs(dL))}
          </span>
        )}
      </div>

      <div style={{ fontSize: 12, color: 'var(--text-primary)', marginBottom: 6 }}>
        <strong>Moved:</strong> {fmtPos(prev[0], prev[1])} → {fmtPos(cur[0], cur[1])}
      </div>

      <div style={{ fontSize: 11.5, color: 'var(--text-sec)', marginBottom: 8, lineHeight: 1.45, background: 'var(--bg-dark)', padding: '6px 8px', borderRadius: 4 }}>
        {getWhyExplanation()}
      </div>

      <div style={{ display: 'flex', gap: 6 }}>
        <button
          className="btn-secondary"
          style={{ flex: 1, padding: '3px 6px', fontSize: 10.5 }}
          onClick={() => setShowFormula(p => !p)}
        >
          {showFormula ? 'Hide Math' : 'Show Math'}
        </button>
        <button
          className="btn-secondary"
          style={{ flex: 1, padding: '3px 6px', fontSize: 10.5 }}
          onClick={() => setShowNumbers(p => !p)}
        >
          {showNumbers ? 'Hide Numbers' : 'Calculation'}
        </button>
      </div>

      {showFormula && (
        <div className="opt-info-formula" style={{ width: '100%', margin: '6px 0 0', fontSize: 10.5, whiteSpace: 'pre-wrap' }}>
          {name === 'SGD' && 'θ ← θ − η·∇L'}
          {(name === 'Momentum' || name === 'SGDMomentum') && 'v ← β·v + (1−β)·∇L\nθ ← θ − η·v'}
          {name === 'NAG' && 'v ← β·v + (1−β)·∇L(θ − βv)\nθ ← θ − η·v'}
          {name === 'AdaGrad' && 'G += ∇L²\nθ ← θ − (η/√(G+ε))·∇L'}
          {name === 'RMSProp' && 'v ← β·v + (1−β)·∇L²\nθ ← θ − (η/√(v+ε))·∇L'}
          {name === 'Adam' && 'm, v EMA + bias correction\nθ ← θ − η·m̂/(√v̂+ε)'}
          {name === 'AdamW' && 'θ ← θ − η·[ m̂/(√v̂+ε) + λ·θ ]'}
        </div>
      )}

      {showNumbers && (
        <div className="step-math-eq" style={{ marginTop: 6, fontSize: 10.5, background: 'var(--bg-dark)', padding: '6px 8px', borderRadius: 4 }}>
          <div>∇L = [{fmtNumber(gx)}, {fmtNumber(gy)}]</div>
          <div>‖∇L‖ = {fmtNumber(gn)}</div>
          <div>{fmtDelta(dx, dy)}</div>
          <div style={{ color: 'var(--text-primary)', marginTop: 4 }}>Updated θ = {fmtPos(cur[0], cur[1])}</div>
        </div>
      )}
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────── */
export default function PartAPlayground({ appMode }) {
  const isBeginner = appMode.includes('Beginner') || appMode.includes('Simple');

  const [showCoach, setShowCoach] = useState(true);
  const [mode,      setMode]      = useState('compare'); // 'explore' | 'compare'
  const [surface,   setSurface]   = useState('L2: c = 50 (default)');
  const [opts,      setOpts]      = useState(['SGD', 'Adam', 'AdamW']);
  const [lr,        setLr]        = useState(0.01);
  const [x0,        setX0]        = useState(8.0);
  const [y0,        setY0]        = useState(8.0);
  const [beta,      setBeta]      = useState(0.9);
  const [beta1,     setBeta1]     = useState(0.9);
  const [beta2,     setBeta2]     = useState(0.999);
  const [wd,        setWd]        = useState(0.001);
  const [maxSteps,  setMaxSteps]  = useState(500);      // -1 = Until convergence
  const [customSteps, setCustomSteps] = useState('');
  const [gradArrow, setGradArrow] = useState(true);
  const [showLabels,setShowLabels]= useState(true);

  const [step,      setStep]      = useState(0);
  const [playing,   setPlaying]   = useState(false);
  const [speed,     setSpeed]     = useState(60);
  const [data,      setData]      = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [hoverCoord,setHoverCoord]= useState(null);
  const [hoverStep, setHoverStep] = useState(null);

  const contourRef = useRef(null);
  const lossRef    = useRef(null);

  /* Keyboard Shortcuts */
  useEffect(() => {
    const handleKeyDown = (e) => {
      const tag = e.target.tagName.toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea') return;

      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying(p => !p);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        setPlaying(false);
        setStep(p => Math.min(effectiveMax, p + 1));
      } else if (e.code === 'KeyR') {
        e.preventDefault();
        setPlaying(false);
        setStep(0);
      } else if (['Digit1','Digit2','Digit3','Digit4','Digit5','Digit6','Digit7'].includes(e.code)) {
        const idx = parseInt(e.code.replace('Digit','')) - 1;
        if (idx >= 0 && idx < ALL_OPTS.length) {
          const optName = ALL_OPTS[idx];
          setOpts(prev => prev.includes(optName) ? (prev.length > 1 ? prev.filter(o => o !== optName) : prev) : [...prev, optName]);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  /* Derive effective backend max_steps */
  const backendMax = maxSteps === -1 ? SAFETY_LIMIT : maxSteps;

  /* Derived max step from returned data (may be shorter due to divergence / convergence) */
  const effectiveMax = useMemo(() => {
    if (!data) return backendMax;
    return Math.max(...opts.map(n => (data.trajectories?.[n]?.length ?? 1) - 1), 0);
  }, [data, opts, backendMax]);

  /* Fetch Trajectories */
  useEffect(() => {
    if (!opts.length) return;
    setLoading(true);
    setPlaying(false);
    setStep(0);
    fetchTrajectoryData({
      surface_name: surface,
      optimizers: opts,
      x0, y0, lr,
      beta, beta1, beta2, weight_decay: wd,
      max_steps: backendMax,
    })
      .then(r => { setData(r); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [surface, opts, lr, x0, y0, beta, beta1, beta2, wd, maxSteps]);

  /* Animation Frame */
  useEffect(() => {
    if (!playing || step >= effectiveMax) {
      if (step >= effectiveMax) setPlaying(false);
      return;
    }
    const spf = Math.max(1, Math.round(80 / speed));
    const t = setTimeout(() => setStep(p => Math.min(500, p + spf)), speed * spf);
    return () => clearTimeout(t);
  }, [playing, step, speed]);

  const toggleOpt = useCallback((name) => {
    if (mode === 'explore') {
      // Explore: single-optimizer focus
      setOpts([name]);
    } else {
      // Compare: unlimited selection, must keep at least 1
      setOpts(p =>
        p.includes(name)
          ? p.length > 1 ? p.filter(o => o !== name) : p
          : [...p, name]
      );
    }
  }, [mode]);

  /* Experiment result summary (computed once data arrives) */
  const experimentSummary = useMemo(() => {
    if (!data) return null;
    return opts.map(name => {
      const finalStatus = data.final_statuses?.[name] || 'RUNNING';
      const lossArr = data.losses?.[name] || [];
      const steps = lossArr.length - 1;

      let icon, label;
      if (finalStatus === 'CONVERGED') {
        icon = '✓'; label = `Converged · ${steps} steps`;
      } else if (finalStatus === 'DIVERGED') {
        icon = '⚠'; label = `Diverged · step ${steps}`;
      } else if (maxSteps === -1 && steps >= SAFETY_LIMIT - 1) {
        icon = '⏹'; label = `Stopped at safety limit (${steps.toLocaleString()} steps)`;
      } else {
        icon = '·'; label = `Ran ${steps} steps`;
      }
      return { name, finalStatus, steps, icon, label };
    });
  }, [data, opts, maxSteps]);

  const allDone = experimentSummary?.every(r =>
    r.finalStatus === 'CONVERGED' || r.finalStatus === 'DIVERGED'
  );

  const handleResetRun = () => {
    setPlaying(false);
    setStep(0);
  };

  const handleResetAll = () => {
    setPlaying(false);
    setStep(0);
    setSurface('L2: c = 50 (default)');
    setOpts(['SGD', 'Adam', 'AdamW']);
    setLr(0.01);
    setX0(8.0);
    setY0(8.0);
    setBeta(0.9);
    setBeta1(0.9);
    setBeta2(0.999);
    setWd(0.001);
    setMaxSteps(500);
    setCustomSteps('');
  };

  const handleApplyExperiment = (exp) => {
    setSurface(exp.surface);
    setLr(exp.lr);
    setOpts(exp.opts);
    setPlaying(false);
    setStep(0);
  };

  /* Canvas Mouse Interaction */
  const handleCanvasClick = (e) => {
    if (!contourRef.current || !data) return;
    const cv = contourRef.current;
    const rect = cv.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const mx = data.max_x || 10;
    const my = data.max_y || 10;
    const newX = Number((-mx + (cx / rect.width) * 2 * mx).toFixed(2));
    const newY = Number((my - (cy / rect.height) * 2 * my).toFixed(2));
    setX0(newX);
    setY0(newY);
    setPlaying(false);
    setStep(0);
  };

  const handleCanvasMouseMove = (e) => {
    if (!contourRef.current || !data) return;
    const cv = contourRef.current;
    const rect = cv.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const mx = data.max_x || 10;
    const my = data.max_y || 10;
    const hx = Number((-mx + (cx / rect.width) * 2 * mx).toFixed(2));
    const hy = Number((my - (cy / rect.height) * 2 * my).toFixed(2));
    const cVal = SURFACES[surface]?.c || 50;
    const loss = hx * hx + cVal * hy * hy;
    setHoverCoord({ x: hx, y: hy, loss });
  };

  /* Draw 2D Contour Map */
  useEffect(() => {
    if (!data || !contourRef.current) return;
    const cv = contourRef.current, ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);

    const c = SURFACES[surface]?.c || 50;
    let mx = data.max_x || 10, my = data.max_y || 10;

    const tx = x => ((x + mx) / (2 * mx)) * W;
    const ty = y => H - ((y + my) / (2 * my)) * H;

    /* Subtle Dark Neutral Topographic Terrain */
    const img = ctx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      const wy = my - (py / H) * 2 * my;
      for (let px = 0; px < W; px++) {
        const wx = -mx + (px / W) * 2 * mx;
        const t = Math.min(1, Math.log1p(wx * wx + c * wy * wy) / 10.5);
        const i = (py * W + px) * 4;
        // Neutral Slate / Navy Elevation
        img.data[i]     = Math.floor(10 + 16 * t); // R
        img.data[i + 1] = Math.floor(14 + 20 * t); // G
        img.data[i + 2] = Math.floor(24 + 30 * t); // B
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    /* Elevation Isolines */
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 0.8;
    [2, 10, 30, 80, 200, 500, 1200, 3000].forEach(lvl => {
      ctx.beginPath();
      for (let px = 0; px < W - 1; px += 2) {
        const wx = -mx + (px / W) * 2 * mx;
        const rem = lvl - wx * wx;
        if (rem >= 0) {
          const wy = Math.sqrt(rem / c);
          ctx.moveTo(tx(wx), ty(wy));  ctx.lineTo(tx(wx) + 1.2, ty(wy));
          ctx.moveTo(tx(wx), ty(-wy)); ctx.lineTo(tx(wx) + 1.2, ty(-wy));
        }
      }
      ctx.stroke();
    });

    /* Axes */
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(tx(0), 0); ctx.lineTo(tx(0), H);
    ctx.moveTo(0, ty(0)); ctx.lineTo(W, ty(0));
    ctx.stroke(); ctx.setLineDash([]);

    /* Global Minimum (Gold Star at origin) */
    const minX = tx(0), minY = ty(0);
    ctx.fillStyle = '#F59E0B';
    ctx.shadowColor = 'rgba(245, 158, 11, 0.6)';
    ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(minX, minY, 5, 0, 2 * Math.PI); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('★', minX, minY);
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';

    /* Start Marker (x0, y0) */
    const sx = tx(x0), sy = ty(y0);
    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 1.5; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#38BDF8';
    ctx.beginPath(); ctx.arc(sx, sy, 3, 0, 2 * Math.PI); ctx.fill();

    /* Trajectories */
    opts.forEach(name => {
      const traj = data.trajectories[name];
      if (!traj?.length) return;
      const { hex, rgb } = OPT_COLORS[name] || { hex: '#FFF', rgb: '255,255,255' };
      const slice = traj.slice(0, step + 1);

      if (slice.length > 1) {
        ctx.strokeStyle = hex;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (let i = 0; i < slice.length; i++) {
          const px = tx(slice[i][0]);
          const py = ty(slice[i][1]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }

      /* Active head point */
      const cur = slice[slice.length - 1];
      if (!cur) return;
      const safeStep = Math.min(step, slice.length - 1);
      const isDiv = data.statuses?.[name]?.[safeStep] === 'DIVERGED';

      let cx = tx(cur[0]), cy = ty(cur[1]);
      if (isDiv) {
        cx = Math.max(16, Math.min(W - 16, cx));
        cy = Math.max(16, Math.min(H - 16, cy));
        ctx.fillStyle = '#EF4444';
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2 * Math.PI); ctx.fill();
        ctx.font = 'bold 9px Inter';
        ctx.fillStyle = '#F87171';
        ctx.fillText(`${name} ↗ OFF-SCREEN`, cx + 9, cy + 4);
        return;
      }

      ctx.fillStyle = hex;
      ctx.shadowColor = hex; ctx.shadowBlur = 6;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2 * Math.PI); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2 * Math.PI); ctx.stroke();

      if (showLabels) {
        ctx.fillStyle = hex;
        ctx.font = '600 10px JetBrains Mono';
        ctx.fillText(name, cx + 8, cy - 4);
      }
    });

    /* Hover step link marker */
    if (hoverStep != null && hoverStep <= step) {
      opts.forEach(name => {
        const traj = data.trajectories[name];
        if (traj && traj[hoverStep]) {
          const hpt = traj[hoverStep];
          const hx = tx(hpt[0]), hy = ty(hpt[1]);
          ctx.strokeStyle = '#FDE047'; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(hx, hy, 9, 0, 2 * Math.PI); ctx.stroke();
        }
      });
    }

  }, [data, step, opts, surface, x0, y0, showLabels, hoverStep]);

  /* Draw Loss vs Iteration Chart — x-axis adapts to effectiveMax */
  useEffect(() => {
    if (!data || !lossRef.current) return;
    const cv = lossRef.current, ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const P = { t: 24, r: 16, b: 24, l: 42 };
    ctx.clearRect(0, 0, W, H);

    /* Grid lines */
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let g = 1; g < 5; g++) {
      const y = P.t + (g / 5) * (H - P.t - P.b);
      ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W - P.r, y); ctx.stroke();
    }

    const toX = s => P.l + (s / effectiveMax) * (W - P.l - P.r);
    const toY = v => {
      const lv = Math.log10(Math.max(1e-6, v));
      const norm = (lv - (-6)) / (7 - (-6));
      return H - P.b - Math.max(0, Math.min(1, norm)) * (H - P.t - P.b);
    };

    opts.forEach(name => {
      const losses = data.losses[name];
      if (!losses?.length) return;
      const { hex } = OPT_COLORS[name] || { hex: '#FFF' };
      const slice = losses.slice(0, step + 1);

      ctx.strokeStyle = hex; ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i < slice.length; i++) {
        const v = slice[i];
        if (v != null && !isNaN(v)) {
          const x = toX(i), y = toY(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      const last = slice[slice.length - 1];
      if (last != null && !isNaN(last)) {
        ctx.fillStyle = hex;
        ctx.beginPath(); ctx.arc(toX(slice.length - 1), toY(last), 4, 0, 2 * Math.PI); ctx.fill();
      }
    });

    /* Hover step crosshair on loss chart */
    if (hoverStep != null) {
      const hx = toX(hoverStep);
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, P.t); ctx.lineTo(hx, H - P.b);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(hoverStep.toLocaleString(), hx + 3, P.t + 11);
    }

    /* Axis labels — x-axis adapts to experiment length */
    const xLabel = maxSteps === -1 ? `${effectiveMax} steps` : effectiveMax.toLocaleString();
    ctx.fillStyle = 'rgba(160,170,190,0.6)';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('Loss', 6, P.t + 4);
    ctx.fillText(xLabel, W - P.r - (xLabel.length * 5.5), H - 6);
    ctx.fillText('0', P.l, H - 6);

  }, [data, step, opts, effectiveMax, hoverStep, maxSteps]);

  /* Telemetry of primary optimizer */
  const leadOpt = opts[0] || 'Adam';
  const traj0 = data?.trajectories?.[leadOpt];
  const lossArr0 = data?.losses?.[leadOpt];
  const status0 = data?.statuses?.[leadOpt]?.[Math.min(step, (data?.statuses?.[leadOpt]?.length || 1) - 1)] || 'RUNNING';
  const pos0 = traj0?.[Math.min(step, (traj0?.length || 1) - 1)] || [x0, y0];
  const curLoss0 = lossArr0?.[Math.min(step, (lossArr0?.length || 1) - 1)] ?? 0;
  const initLoss = lossArr0?.[0] ?? curLoss0;
  const lossReduction = initLoss > 0 ? ((initLoss - curLoss0) / initLoss * 100).toFixed(1) : '0';
  const dist0 = Math.hypot(pos0[0], pos0[1]).toFixed(2);

  /* Dynamic story sentence */
  const getStory = () => {
    if (status0 === 'DIVERGED') return `${leadOpt} diverged! Step size was too aggressive for steep valley walls.`;
    if (status0 === 'CONVERGED' || curLoss0 < 1e-4) return `${leadOpt} reached the global minimum at (0, 0) in ${step} steps.`;
    if (step === 0) return `Ready to run. Trajectory begins at (${x0.toFixed(1)}, ${y0.toFixed(1)}).`;
    return `${leadOpt} is making steady progress toward the origin (distance: ${dist0}).`;
  };

  /* Dynamic convergence comparison sentence */
  const getComparisonInsight = () => {
    if (opts.length < 2 || !data) return 'Tracking optimizer convergence rates in real time.';
    const optLosses = opts.map(o => ({
      name: o,
      loss: data.losses[o]?.[Math.min(step, (data.losses[o]?.length || 1) - 1)] ?? 9999
    })).sort((a, b) => a.loss - b.loss);
    return `${optLosses[0].name} is currently leading with lowest loss (${fmtLoss(optLosses[0].loss)}).`;
  };

  return (
    <div className="playground-layout">
      {/* ── LEFT PANEL: Sidebar Controls ── */}
      <aside className="playground-sidebar hud-card mobile-order-controls">
        {/* Primary Action Button */}
        <div>
          <button
            className="btn-primary"
            style={{
              width: '100%',
              padding: '13px 0',
              fontSize: 14,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              boxShadow: playing ? '0 0 16px rgba(99,102,241,0.5)' : 'none',
            }}
            onClick={() => setPlaying(p => !p)}
          >
            {playing ? (
              <>
                <Pause className="w-4 h-4" /> Pause Experiment
              </>
            ) : (
              <>
                <Play className="w-4 h-4" /> Run Experiment
              </>
            )}
          </button>

          {/* Secondary Controls Row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginTop: 8 }}>
            <button
              className="btn-secondary"
              style={{ padding: '7px 0', fontSize: 11 }}
              onClick={() => { setPlaying(false); setStep(p => Math.min(500, p + 1)); }}
              title="Next Step (ArrowRight)"
            >
              <SkipForward className="w-3 h-3" /> Step
            </button>
            <button
              className="btn-secondary"
              style={{ padding: '7px 0', fontSize: 11 }}
              onClick={handleResetRun}
              title="Reset to Step 0 (R)"
            >
              <RotateCcw className="w-3 h-3" /> Reset Run
            </button>
            <button
              className="btn-secondary"
              style={{ padding: '7px 0', fontSize: 11 }}
              onClick={handleResetAll}
              title="Restore all default settings"
            >
              <RefreshCw className="w-3 h-3" /> Reset All
            </button>
          </div>
        </div>

        {/* ── Experiment Length Control ── */}
        <div>
          <div className="section-header">
            <span className="section-title">Experiment Length</span>
            <div className="section-line" />
          </div>
          <select
            className="hud-select"
            value={maxSteps}
            onChange={e => {
              const val = parseInt(e.target.value);
              setMaxSteps(val);
              setPlaying(false);
              setStep(0);
            }}
          >
            {STEP_PRESETS.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
            <option value={0}>Custom…</option>
          </select>

          {/* Custom steps input */}
          {maxSteps === 0 && (
            <div style={{ display: 'flex', gap: 6, marginTop: 6, alignItems: 'center' }}>
              <input
                className="hud-input"
                type="number"
                min={1}
                max={100000}
                placeholder="e.g. 3000"
                value={customSteps}
                onChange={e => setCustomSteps(e.target.value)}
                style={{ flex: 1 }}
              />
              <button
                className="btn-secondary"
                style={{ padding: '5px 10px', fontSize: 11, whiteSpace: 'nowrap' }}
                onClick={() => {
                  const v = parseInt(customSteps);
                  if (v > 0 && v <= 100000) {
                    setMaxSteps(v);
                    setPlaying(false);
                    setStep(0);
                  }
                }}
              >
                Apply
              </button>
            </div>
          )}

          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 5 }}>
            {maxSteps === -1
              ? 'Runs until convergence or divergence (safety limit: 100k)'
              : `${backendMax.toLocaleString()} steps · applies to all optimizers equally`}
          </div>
        </div>

        {/* Optimizers Selector */}
        <div>
          <div className="section-header">
            <span className="section-title">Optimizers</span>
            <div className="section-line" />
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
            Toggle any combination · hotkeys 1–7 · at least 1 must stay active
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {ALL_OPTS.map(name => {
              const active = opts.includes(name);
              const { hex, rgb } = OPT_COLORS[name] || { hex: '#FFF', rgb: '255,255,255' };
              return (
                <button
                  key={name}
                  className={`opt-pill ${active ? 'active' : ''}`}
                  style={active ? { borderColor: hex, color: hex, background: `rgba(${rgb},0.12)`, boxShadow: `0 0 8px rgba(${rgb},0.25)` } : { opacity: 0.6 }}
                  onClick={() => toggleOpt(name)}
                >
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: hex }} />
                  {name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Learning Rate Slider & Presets */}
        <div>
          <div className="param-row">
            <span className="param-label">Learning Rate <SymbolTooltip sym="η">η</SymbolTooltip></span>
            <span className="param-value">{lr}</span>
          </div>
          <select
            className="hud-select"
            value={lr}
            onChange={e => setLr(parseFloat(e.target.value))}
          >
            {LR_OPTS.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
          <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
            {[0.001, 0.01, 0.1].map(preset => (
              <button
                key={preset}
                className="btn-secondary"
                style={{ flex: 1, padding: '3px 0', fontSize: 10.5, borderColor: lr === preset ? 'var(--accent)' : 'var(--border)' }}
                onClick={() => setLr(preset)}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Starting Coordinates & Surface */}
        <div>
          <div className="section-header">
            <span className="section-title">Surface & Starting Point</span>
            <div className="section-line" />
          </div>
          <div style={{ marginBottom: 8 }}>
            <label className="param-label" style={{ display: 'block', marginBottom: 4 }}>Loss Surface</label>
            <select
              className="hud-select"
              value={surface}
              onChange={e => { setSurface(e.target.value); setStep(0); }}
            >
              {Object.keys(SURFACES).map(k => (
                <option key={k} value={k}>{SURFACES[k].name}</option>
              ))}
            </select>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div>
              <span className="param-label">x₀</span>
              <input
                className="hud-input"
                type="number"
                value={x0}
                step="0.5"
                onChange={e => setX0(parseFloat(e.target.value) || 0)}
              />
            </div>
            <div>
              <span className="param-label">y₀</span>
              <input
                className="hud-input"
                type="number"
                value={y0}
                step="0.5"
                onChange={e => setY0(parseFloat(e.target.value) || 0)}
              />
            </div>
          </div>
          <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginTop: 4 }}>
            Tip: Click anywhere on 2D map to set (x₀, y₀)
          </div>
        </div>

        {/* Hyperparameters (Advanced Mode) */}
        {!isBeginner && (
          <div>
            <div className="section-header">
              <span className="section-title">Hyperparameters</span>
              <div className="section-line" />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div>
                <div className="param-row"><span className="param-label">Momentum <SymbolTooltip sym="β">β</SymbolTooltip></span><span className="param-value">{beta}</span></div>
                <input className="hud-range" type="range" min={0} max={0.99} step={0.01} value={beta} onChange={e => setBeta(parseFloat(e.target.value))} />
              </div>
              <div>
                <div className="param-row"><span className="param-label">Adam <SymbolTooltip sym="β₁">β₁</SymbolTooltip></span><span className="param-value">{beta1}</span></div>
                <input className="hud-range" type="range" min={0} max={0.99} step={0.01} value={beta1} onChange={e => setBeta1(parseFloat(e.target.value))} />
              </div>
              <div>
                <div className="param-row"><span className="param-label">Adam <SymbolTooltip sym="β₂">β₂</SymbolTooltip></span><span className="param-value">{beta2}</span></div>
                <input className="hud-range" type="range" min={0} max={0.999} step={0.001} value={beta2} onChange={e => setBeta2(parseFloat(e.target.value))} />
              </div>
              <div>
                <div className="param-row"><span className="param-label">Weight Decay <SymbolTooltip sym="λ">λ</SymbolTooltip></span><span className="param-value">{wd}</span></div>
                <input className="hud-range" type="range" min={0} max={0.05} step={0.001} value={wd} onChange={e => setWd(parseFloat(e.target.value))} />
              </div>
            </div>
          </div>
        )}

        {/* Animation Speed & View Options */}
        <div>
          <div className="section-header">
            <span className="section-title">Display & Speed</span>
            <div className="section-line" />
          </div>
          <div className="param-row"><span className="param-label">Speed</span><span className="param-value">{speed}ms</span></div>
          <input className="hud-range" type="range" min={10} max={120} step={5} value={speed} onChange={e => setSpeed(parseInt(e.target.value))} />
          <div style={{ display: 'flex', gap: 12, marginTop: 10, fontSize: 11.5 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
              <input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} />
              Labels
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
              <input type="checkbox" checked={gradArrow} onChange={e => setGradArrow(e.target.checked)} />
              Arrows
            </label>
          </div>
        </div>
      </aside>

      {/* ── RIGHT PANEL: Hero Visualization & Educational Content ── */}
      <section className="playground-content">
        {/* Compact Coach Strip */}
        {showCoach && (
          <CompactCoachStrip
            onOpenGuide={() => window.dispatchEvent(new CustomEvent('openHelp'))}
            onDismiss={() => setShowCoach(false)}
          />
        )}

        {/* Experiment Header & Mode Switcher */}
        <div className="glass-card" style={{ padding: '14px 18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: '0.08em', color: 'var(--accent)', textTransform: 'uppercase' }}>
                OPTIMIZATION EXPERIMENT
              </div>
              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                Finding the Minimum · {SURFACES[surface]?.name}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-sec)', marginTop: 2 }}>
                Start ({x0.toFixed(1)}, {y0.toFixed(1)}) → Goal (0, 0) · κ = {SURFACES[surface]?.kappa}
              </div>
            </div>

            <div className="mode-toggle">
              <button
                className={`mode-btn ${mode === 'explore' ? 'active' : ''}`}
                onClick={() => {
                  setMode('explore');
                  if (opts.length > 1) setOpts([opts[0]]);
                }}
              >
                Explore Single
              </button>
              <button
                className={`mode-btn ${mode === 'compare' ? 'active' : ''}`}
                onClick={() => {
                  setMode('compare');
                  if (opts.length <= 1) setOpts(['SGD', 'Adam', 'AdamW']);
                }}
              >
                Compare Multi
              </button>
            </div>
          </div>
        </div>

        {/* Dynamic Story & Live Status Strip */}
        <div className="telemetry-strip" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              className="live-dot"
              style={{
                background: status0 === 'DIVERGED' ? '#EF4444' : status0 === 'CONVERGED' ? '#10B981' : '#6366F1',
                boxShadow: `0 0 8px ${status0 === 'DIVERGED' ? '#EF4444' : status0 === 'CONVERGED' ? '#10B981' : '#6366F1'}`,
              }}
            />
            <span style={{ fontWeight: 600, color: status0 === 'DIVERGED' ? '#F87171' : 'var(--text-primary)' }}>
              {status0} — {getStory()}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontFamily: 'JetBrains Mono', fontSize: 11 }}>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11 }}>
              {maxSteps === -1
                ? `Step ${step} · Until convergence`
                : `Step ${step.toLocaleString()} / ${backendMax.toLocaleString()}`}
            </span>
            <span className="tele-sep">|</span>
            <span>Loss: <strong>{fmtLoss(curLoss0)}</strong> ({lossReduction}% drop)</span>
            <span className="tele-sep">|</span>
            <span>Dist: {dist0}</span>
          </div>
        </div>

        {/* ── HERO DUAL VISUALIZATION ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
          {/* Contour Map */}
          <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="section-header" style={{ marginBottom: 0 }}>
                <span className="section-title">Contour Map · Trajectories</span>
                <div className="section-line" />
              </div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10.5, color: '#A5B4FC' }}>
                {hoverCoord ? `x: ${hoverCoord.x.toFixed(2)}, y: ${hoverCoord.y.toFixed(2)} | L: ${fmtLoss(hoverCoord.loss)}` : 'Click to set (x₀, y₀)'}
              </div>
            </div>
            <div style={{ position: 'relative' }}>
              <canvas
                ref={contourRef}
                width={530}
                height={360}
                onClick={handleCanvasClick}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={() => setHoverCoord(null)}
                style={{ display: 'block', width: '100%', height: 'auto', cursor: 'crosshair' }}
              />
            </div>
          </div>

          {/* Convergence Loss Chart */}
          <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="section-header" style={{ marginBottom: 0 }}>
                <span className="section-title">Convergence · Loss Curve</span>
                <div className="section-line" />
              </div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Log scale · Lower is better</span>
            </div>
            <div style={{ padding: '6px 14px', background: 'rgba(0,0,0,0.2)', borderBottom: '1px solid var(--border)', fontSize: 11, color: '#A5B4FC' }}>
              {getComparisonInsight()}
            </div>
            <div style={{ position: 'relative' }}>
              <canvas
                ref={lossRef}
                width={530}
                height={330}
                onMouseMove={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const px = e.clientX - rect.left;
                  const st = Math.round(((px - 42) / (rect.width - 58)) * effectiveMax);
                  setHoverStep(Math.max(0, Math.min(effectiveMax, st)));
                }}
                onMouseLeave={() => setHoverStep(null)}
                style={{ display: 'block', width: '100%', height: 'auto', background: 'rgba(3,5,13,0.5)' }}
              />
              <div style={{ position: 'absolute', bottom: 6, left: 12, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {opts.map(n => (
                  <span key={n} style={{ fontSize: 9.5, fontFamily: 'JetBrains Mono', color: OPT_COLORS[n]?.hex }}>
                    ● {n}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Compare Mode Summary Matrix */}
        {mode === 'compare' && (
          <div className="glass-card" style={{ padding: 14 }}>
            <div className="section-header">
              <span className="section-title">Comparison Matrix</span>
              <div className="section-line" />
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Optimizer</th>
                    <th>Convergence Speed</th>
                    <th>Adaptivity</th>
                    <th>Weight Decay</th>
                    <th>Current Loss</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {opts.map(name => {
                    const { hex } = OPT_COLORS[name] || {};
                    const tr = OPT_TRAITS[name] || {};
                    const lVal = data?.losses?.[name]?.[Math.min(step, (data?.losses?.[name]?.length || 1) - 1)] ?? 0;
                    const st = data?.statuses?.[name]?.[Math.min(step, (data?.statuses?.[name]?.length || 1) - 1)] || 'RUNNING';
                    return (
                      <tr key={name}>
                        <td className="opt-name" style={{ color: hex }}>● {name}</td>
                        <td>{tr.speed}</td>
                        <td>{tr.adaptivity}</td>
                        <td>{tr.weightDecay}</td>
                        <td className="mono">{fmtLoss(lVal)}</td>
                        <td>
                          <span style={{ color: st === 'DIVERGED' ? '#F87171' : st === 'CONVERGED' ? '#34D399' : 'var(--text-sec)', fontWeight: 600 }}>
                            {st}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Experiment Result Summary (shown when all optimizers are done) ── */}
        {experimentSummary && (allDone || step === effectiveMax) && step > 0 && (
          <div className="glass-card" style={{ padding: '12px 18px' }}>
            <div className="section-header" style={{ marginBottom: 10 }}>
              <CheckCircle2 className="w-4 h-4" style={{ color: '#34D399', flexShrink: 0 }} />
              <span className="section-title">Experiment Complete</span>
              <div className="section-line" />
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                {maxSteps === -1 ? 'Until convergence' : `${backendMax.toLocaleString()} steps`}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 8 }}>
              {experimentSummary.map(r => {
                const { hex } = OPT_COLORS[r.name] || { hex: '#FFF' };
                const statusColor = r.finalStatus === 'CONVERGED' ? '#34D399'
                  : r.finalStatus === 'DIVERGED' ? '#F87171'
                  : 'var(--text-sec)';
                return (
                  <div
                    key={r.name}
                    style={{
                      background: 'var(--surface-elev)',
                      border: `1px solid ${r.finalStatus === 'CONVERGED' ? 'rgba(52,211,153,0.25)' : r.finalStatus === 'DIVERGED' ? 'rgba(248,113,113,0.25)' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-sm)',
                      padding: '8px 12px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 3,
                    }}
                  >
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, fontWeight: 700, color: hex }}>
                      {r.name}
                    </div>
                    <div style={{ fontSize: 11.5, color: statusColor, fontWeight: 600 }}>
                      {r.icon} {r.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Progressive Step Math Inspection ("What happened this step?") */}
        {step > 0 && data && (
          <div className="step-math-panel">
            <div className="step-math-header">
              <span style={{ fontWeight: 700, fontSize: 12.5, color: 'var(--text-primary)' }}>
                What happened this step? — Step {step.toLocaleString()} / {effectiveMax.toLocaleString()}
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Slope Inspection & Exact Math
              </span>
            </div>
            <div className="step-math-body">
              {opts.map(name => {
                const traj = data.trajectories[name];
                if (!traj || traj.length < 2) return null;
                const safeStep = Math.min(step, traj.length - 1);
                const cur = traj[safeStep];
                const prev = traj[Math.max(0, safeStep - 1)];
                return (
                  <StepMathBlock
                    key={name}
                    name={name}
                    cur={cur}
                    prev={prev}
                    safeStep={safeStep}
                    data={data}
                    surface={surface}
                    lr={lr}
                    beta={beta}
                    wd={wd}
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* Contextual Experiments */}
        <TryExperimentCards onApplyExperiment={handleApplyExperiment} />

        {/* Section A5: Conditioning Explorer */}
        <ConditioningExplorer
          currentSurface={surface}
          onSelectSurface={(s) => {
            setSurface(s);
            setPlaying(false);
            setStep(0);
          }}
          lr={lr}
        />

        {/* Section A4: Explain-As-You-Go Deep Dives */}
        <ExplainAsYouGo />
      </section>
    </div>
  );
}
