import React from 'react';
import { HelpCircle, Activity, Layers, BookOpen, Command, X } from 'lucide-react';

const SHORTCUTS = [
  { key: 'Space', action: 'Run / Pause Experiment' },
  { key: '→ (ArrowRight)', action: 'Advance single step' },
  { key: 'R', action: 'Reset run to Step 0' },
  { key: '1 – 7', action: 'Toggle optimizer (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW)' },
];

export default function HowToUseModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 680 }}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <HelpCircle className="w-5 h-5 text-indigo-400" />
            <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
              Optimizer Lab User Guide &amp; Reference
            </div>
          </div>
          <button className="btn-secondary" style={{ padding: '4px 10px', fontSize: 12 }} onClick={onClose}>
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Section 1: Part A */}
          <div className="modal-info-card">
            <div className="modal-info-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Activity className="w-4 h-4 text-indigo-400" /> Part A: 2D Loss Surface Playground
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.5 }}>
              • Toggle any combination of optimizers (hotkeys 1–7) to compare trajectories simultaneously.<br />
              • Choose loss surfaces (L₁ to L₄) to test varying condition numbers (κ = 10 → 1000).<br />
              • <strong>Click anywhere on the 2D Contour Map</strong> to reposition the starting point (x₀, y₀) live.<br />
              • Set <strong>Experiment Length</strong> — choose 100 / 500 / 1,000 / 2,000 / 5,000 steps or <em>Until convergence</em>.<br />
              • Press <strong>Run Experiment</strong> or tap <kbd style={{ padding: '1px 5px', background: 'var(--bg-dark)', borderRadius: 3, border: '1px solid var(--border)', fontFamily: 'JetBrains Mono', fontSize: 10.5 }}>Space</kbd> to animate step-by-step downhill.
            </div>
          </div>

          {/* Section 2: Part B */}
          <div className="modal-info-card">
            <div className="modal-info-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Layers className="w-4 h-4 text-indigo-400" /> Part B: Neural Network Benchmark
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.5 }}>
              • Trains a 3-layer MLP (`Input(30) → Dense(16) → ReLU → Dense(8) → ReLU → Dense(1) → Sigmoid`) on Breast Cancer Wisconsin.<br />
              • Compares training &amp; validation loss curves, classification accuracy, and effective learning rates η_eff.<br />
              • Automatically computes the 1% convergence epoch in the summary comparison table.
            </div>
          </div>

          {/* Section 3: Part C */}
          <div className="modal-info-card">
            <div className="modal-info-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <BookOpen className="w-4 h-4 text-indigo-400" /> Part C: Theory &amp; Reflection Hub
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.5 }}>
              • Complete, mathematically verified answers to all Reflection Questions for Part A (A7.1–A7.8) and Part B (B4.1–B4.16).<br />
              • Full synthesis of the evolutionary progression from SGD to AdamW.
            </div>
          </div>

          {/* Keyboard Shortcuts Table */}
          <div style={{ background: 'var(--surface-elev)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '12px 14px' }}>
            <div style={{ fontWeight: 700, fontSize: 12, color: 'var(--text-primary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Command className="w-3.5 h-3.5 text-indigo-400" /> Keyboard Shortcuts
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {SHORTCUTS.map(sc => (
                <div key={sc.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, background: 'var(--bg-dark)', padding: '5px 8px', borderRadius: 4, border: '1px solid var(--border)' }}>
                  <span style={{ fontFamily: 'JetBrains Mono', color: '#A5B4FC', fontWeight: 600 }}>{sc.key}</span>
                  <span style={{ color: 'var(--text-sec)' }}>{sc.action}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={{ padding: '0 24px 20px', textAlign: 'right' }}>
          <button className="btn-primary" style={{ padding: '9px 24px', fontSize: 12 }} onClick={onClose}>
            Got it · Let&apos;s Explore
          </button>
        </div>
      </div>
    </div>
  );
}
