import React from 'react';

const SURFACES = [
  { key: 'L1: c = 10', title: 'L₁: x² + 10y²', kappa: 10, note: 'Mild Curvature · Moderate Anisotropy', formula: 'L(x,y) = x² + 10y²' },
  { key: 'L2: c = 50 (default)', title: 'L₂: x² + 50y²', kappa: 50, note: 'Default Lab Bowl · Elongated Valley', formula: 'L(x,y) = x² + 50y²' },
  { key: 'L3: c = 100', title: 'L₃: x² + 100y²', kappa: 100, note: 'Ill-Conditioned · High Oscillation', formula: 'L(x,y) = x² + 100y²' },
  { key: 'L4: c = 1000', title: 'L₄: x² + 1000y²', kappa: 1000, note: 'Extreme Ravine · SGD Diverges at η=0.01', formula: 'L(x,y) = x² + 1000y²' },
];

export default function ConditioningExplorer({ currentSurface, onSelectSurface, lr = 0.01 }) {
  return (
    <div className="glass-card" style={{ padding: 18, marginTop: 14 }}>
      <div className="section-header" style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 16 }}>🌐</span>
          <span className="section-title">Section A5: Conditioning Number Explorer (L₁ → L₂ → L₃ → L₄)</span>
        </div>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--accent)', background: 'rgba(99,102,241,0.1)', padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(99,102,241,0.25)' }}>
          Hessian Anisotropy κ = λ_max / λ_min
        </span>
      </div>

      <p style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.5, marginBottom: 14 }}>
        The condition number κ = λ_max / λ_min of the Hessian ∇²L measures how narrow and elongated the bowl is. Click any bowl below to instantly load it into the playground and observe how non-adaptive optimizers (like SGD) violently zig-zag as κ surges:
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10, marginBottom: 14 }}>
        {SURFACES.map((s) => {
          const isSelected = currentSurface === s.key;
          const isExtreme = s.kappa >= 500;
          const isHigh = s.kappa >= 100;

          return (
            <div
              key={s.key}
              onClick={() => onSelectSurface(s.key)}
              style={{
                background: isSelected ? 'rgba(99,102,241,0.16)' : 'var(--surface-elev)',
                border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius: 'var(--radius-md)',
                padding: '12px 14px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: isSelected ? '0 0 14px rgba(99,102,241,0.25)' : 'none',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontWeight: 700, fontSize: 13, color: isSelected ? '#A5B4FC' : 'var(--text-primary)' }}>
                  {s.title}
                </span>
                <span
                  style={{
                    fontFamily: 'JetBrains Mono',
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: isExtreme ? 'rgba(239,68,68,0.15)' : isHigh ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.15)',
                    color: isExtreme ? '#F87171' : isHigh ? '#FBBF24' : '#34D399',
                    border: `1px solid ${isExtreme ? 'rgba(239,68,68,0.3)' : isHigh ? 'rgba(245,158,11,0.3)' : 'rgba(16,185,129,0.3)'}`,
                  }}
                >
                  κ = {s.kappa}
                </span>
              </div>

              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--text-sec)', marginBottom: 6 }}>
                {s.formula}
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                {s.note}
              </div>

              <div style={{ borderTop: '1px solid var(--border)', paddingTop: 6, fontSize: 10.5, display: 'flex', flexDirection: 'column', gap: 3 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#FF6B6B', fontWeight: 600 }}>SGD at η={lr}:</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: isHigh && lr >= 0.01 ? '#F87171' : '#34D399' }}>
                    {isHigh && lr >= 0.01 ? '⚠ Diverges' : 'Oscillates'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#7C6CFF', fontWeight: 600 }}>Adam at η={lr}:</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: '#34D399' }}>
                    ✓ Converges
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', fontSize: 11.5, color: '#C7D2FE', lineHeight: 1.5 }}>
        <strong style={{ color: '#A5B4FC' }}>Conditioning Theorem:</strong> For gradient descent on a quadratic with maximum Hessian eigenvalue λ_max, stability strictly requires η &lt; 2 / λ_max. On L₄, λ_max = 2000 ⟹ η &lt; 0.001. At η = 0.01, SGD explodes to infinity, while adaptive algorithms (Adam, RMSProp) automatically divide by √(v_t) and remain unconditionally stable.
      </div>
    </div>
  );
}
