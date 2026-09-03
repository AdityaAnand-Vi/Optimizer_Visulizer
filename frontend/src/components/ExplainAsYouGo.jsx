import React, { useState } from 'react';

const DEEP_DIVES = [
  {
    id: 'nag',
    num: '1',
    color: '#3DD9FF',
    title: "Why NAG's Look-Ahead Gradient Reduces Overshoot vs. Plain Momentum",
    formula: 'Plain Momentum: g = ∇L(θₜ) → vₜ = β vₜ₋₁ + (1-β) g\nNAG Look-Ahead:  g_look = ∇L(θₜ - η β vₜ₋₁) → vₜ = β vₜ₋₁ + (1-β) g_look',
    mechanism: 'Plain Momentum computes the gradient at current position θₜ and adds it to past velocity, behaving like a blind heavy ball rolling full-speed into an opposing uphill slope before realizing it needs to turn back.',
    solution: 'NAG first takes a provisional look-ahead leap along the momentum vector θₜ - η·β·vₜ₋₁ (where it expects to be next). If that point climbs up the opposite ravine wall, ∇L immediately senses the rising slope and applies an anticipatory brake before the step is executed, dramatically reducing oscillation and overshoot.',
  },
  {
    id: 'adagrad',
    num: '2',
    color: '#49D99A',
    title: "Why AdaGrad Gives Different Effective Learning Rates to Different Parameters",
    formula: 'Gₜ = Gₜ₋₁ + gₜ²\nθₜ₊₁ = θₜ − (η / √(Gₜ + ε)) ⊙ gₜ',
    mechanism: 'On anisotropic loss surfaces like L(x, y) = x² + 50y², the vertical gradient g_y = 100y is orders of magnitude larger than the horizontal gradient g_x = 2x.',
    solution: 'AdaGrad accumulates squared gradients independently per coordinate (Gₜ,y = ∑ g_y²). The denominator √(Gₜ,y) grows rapidly along the steep y-axis, automatically throttling down its effective step size to prevent explosive bouncing, while keeping the effective rate along the shallow x-axis relatively high to accelerate progress toward the origin.',
  },
  {
    id: 'rmsprop',
    num: '3',
    color: '#2DD4BF',
    title: "How RMSProp Fixes AdaGrad's Continuously-Shrinking Learning Rate",
    formula: 'AdaGrad: Gₜ = Gₜ₋₁ + gₜ²  (unbounded sum)\nRMSProp: vₜ = β vₜ₋₁ + (1-β) gₜ²  (exponential moving average)',
    mechanism: "AdaGrad's fatal flaw: Because gₜ² ≥ 0, Gₜ grows monotonically with every step. As t → 200+, √(Gₜ) becomes so huge that effective step size η/√(Gₜ) drops to near zero. Training freezes prematurely long before reaching the minimum.",
    solution: 'RMSProp replaces the infinite sum with an exponentially decaying moving average (EMA, β=0.9), creating a finite memory horizon of ~10 steps. The denominator adapts to recent landscape curvature rather than total history, allowing the optimizer to keep moving continuously across flat valleys.',
  },
  {
    id: 'adamw',
    num: '4',
    color: '#F472B6',
    title: "The Role of Decoupled Weight Decay in AdamW vs. L2 Regularization",
    formula: 'L2 in Adam: gₜ = ∇L(θ) + λ θₜ  ⟹  step ∝ (∇L + λ θ) / (√v̂ₜ + ε)\nAdamW:      θₜ₊₁ = θₜ − η · [ m̂ₜ / (√v̂ₜ + ε) + λ θₜ ]',
    mechanism: 'In standard SGD, L2 regularization is mathematically identical to weight decay. But in adaptive methods like Adam, folding L2 into the gradient divides the penalty λθ by √(v̂ₜ). As a result, parameters with large historical gradients receive drastically weakened regularization!',
    solution: 'AdamW (Loshchilov & Hutter, 2017) decouples weight decay from the adaptive gradient update, applying a direct proportional shrinkage (1 - ηλ)θ to all parameters uniformly. This restores true scale-invariant regularization and delivers state-of-the-art generalization on Transformers and deep CNNs.',
  },
];

const OPT_CARDS = {
  SGD: { name: 'SGD', formula: 'θ ← θ − η·∇L', desc: 'Steps directly downhill opposite the gradient. Fast and simple, but severely zig-zags in ill-conditioned ravines.' },
  SGDMomentum: { name: 'Momentum', formula: 'v ← β·v + (1−β)·∇L,  θ ← θ − η·v', desc: 'Accumulates velocity in consistent directions, dampening oscillations and accelerating along shallow valleys.' },
  NAG: { name: 'NAG', formula: 'v ← β·v + (1−β)·∇L(θ − ηβv),  θ ← θ − η·v', desc: 'Evaluates gradients at a look-ahead position, providing predictive braking before overshooting valley minima.' },
  AdaGrad: { name: 'AdaGrad', formula: 'G += g²,  θ ← θ − η·g / √(G+ε)', desc: 'Adapts learning rates coordinate-wise based on historical gradient magnitudes. Great for sparse data; freezes in deep nets.' },
  RMSProp: { name: 'RMSProp', formula: 'v ← β·v + (1−β)·g²,  θ ← θ − η·g / √(v+ε)', desc: 'Fixes AdaGrad by replacing cumulative sums with exponential moving averages of squared gradients.' },
  Adam: { name: 'Adam', formula: 'm, v moving averages + bias corrections,  θ ← θ − η·m̂ / (√v̂+ε)', desc: 'Combines Momentum (1st moment) and RMSProp (2nd moment) with exact finite-sample bias correction.' },
  AdamW: { name: 'AdamW', formula: 'θ ← θ − η·[ m̂/(√v̂+ε) + λ·θ ]', desc: 'Decouples weight decay from adaptive gradient scaling, ensuring uniform regularization across all parameters.' },
};

export default function ExplainAsYouGo() {
  const [openSections, setOpenSections] = useState({ nag: true, adagrad: true, rmsprop: true, adamw: true });
  const [activeOpt, setActiveOpt] = useState('AdamW');

  const toggle = (id) => setOpenSections((p) => ({ ...p, [id]: !p[id] }));

  return (
    <div className="glass-card" style={{ padding: 18, marginTop: 14 }}>
      <div className="section-header" style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 16 }}>📖</span>
          <span className="section-title">Section A4: Explain-As-You-Go — Key Theoretical Insights</span>
        </div>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--accent)', background: 'rgba(99,102,241,0.1)', padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(99,102,241,0.25)' }}>
          Mathematical Foundations
        </span>
      </div>

      <p style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.5, marginBottom: 14 }}>
        Examine the mathematical rationale behind the progression from plain SGD to AdamW. Click any topic to inspect why each specific mechanism was engineered:
      </p>

      {/* 4 Core Deep Dives */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 18 }}>
        {DEEP_DIVES.map((d) => {
          const isOpen = openSections[d.id];
          return (
            <div
              key={d.id}
              style={{
                background: 'var(--surface-elev)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                overflow: 'hidden',
              }}
            >
              <div
                onClick={() => toggle(d.id)}
                style={{
                  padding: '10px 14px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer',
                  background: isOpen ? 'rgba(255,255,255,0.02)' : 'transparent',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: d.color, boxShadow: `0 0 6px ${d.color}` }} />
                  <span style={{ fontWeight: 700, fontSize: 12.5, color: 'var(--text-primary)' }}>
                    {d.num}. {d.title}
                  </span>
                </div>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{isOpen ? '▲' : '▼'}</span>
              </div>

              {isOpen && (
                <div style={{ padding: '10px 14px 14px', borderTop: '1px solid var(--border)', background: 'rgba(0,0,0,0.15)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, background: 'var(--bg-dark)', padding: '8px 12px', borderRadius: 4, color: d.color, whiteSpace: 'pre-wrap', border: `1px solid rgba(255,255,255,0.05)` }}>
                    {d.formula}
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-sec)', lineHeight: 1.5 }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Problem / Mechanism:</strong> {d.mechanism}
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-sec)', lineHeight: 1.5 }}>
                    <strong style={{ color: d.color }}>Solution & Impact:</strong> {d.solution}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 7-Optimizer Quick Switcher */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>⚡</span> 7-Optimizer Algorithmic Reference Cards:
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
          {Object.keys(OPT_CARDS).map((k) => (
            <button
              key={k}
              className={`btn-secondary ${activeOpt === k ? 'active' : ''}`}
              style={{
                fontSize: 11,
                padding: '4px 10px',
                borderColor: activeOpt === k ? 'var(--accent)' : 'var(--border)',
                background: activeOpt === k ? 'rgba(99,102,241,0.2)' : 'transparent',
                color: activeOpt === k ? '#A5B4FC' : 'var(--text-sec)',
              }}
              onClick={() => setActiveOpt(k)}
            >
              {k}
            </button>
          ))}
        </div>

        {(() => {
          const card = OPT_CARDS[activeOpt];
          if (!card) return null;
          return (
            <div style={{ background: 'var(--surface-elev)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontWeight: 700, fontSize: 13, color: '#A5B4FC' }}>{card.name}</span>
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: 'var(--text-muted)' }}>Update Rule</span>
              </div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: '#34D399', background: 'var(--bg-dark)', padding: '6px 10px', borderRadius: 4, marginBottom: 6 }}>
                {card.formula}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-sec)', lineHeight: 1.5 }}>
                {card.desc}
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
