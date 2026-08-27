import React, { useState } from 'react';
import { BookOpen, FlaskConical, Cpu, Wrench } from 'lucide-react';

const PART_A_QUESTIONS = [
  {
    q: '1. Which optimizer shows the strongest zig-zag on the default bowl, and why?',
    a: 'Plain SGD exhibits the strongest cross-valley zig-zag. On L₂(x,y) = x² + 50y², the Hessian eigenvalues are λ₁=2 and λ₂=100 (condition number κ=50). Because vertical gradient ∇L_y = 100y is 50× larger than horizontal gradient ∇L_x = 2x, SGD takes massive steps bouncing across the steep walls while advancing at a crawl along the shallow x-axis.',
  },
  {
    q: '2. Which optimizer(s) most visibly reduce oscillation, and through what mechanism?',
    a: 'Momentum, NAG, RMSProp, Adam, and AdamW. Momentum and NAG use velocity averaging vₜ = β vₜ₋₁ + (1−β)gₜ to cancel out alternating ±y wall gradients while accumulating positive x velocity. RMSProp, Adam, and AdamW use coordinate-wise second moment scaling (1/√vₜ) to shrink steep-axis step sizes automatically.',
  },
  {
    q: '3. Which optimizer moves most efficiently along the shallow (x) direction while the y direction is corrected quickly?',
    a: 'Adam and AdamW move most efficiently. They combine 2nd-moment variance scaling (which immediately dampens the steep y oscillation) with 1st-moment velocity acceleration along x, driving straight toward the origin.',
  },
  {
    q: '4. Which optimizers use parameter-wise adaptive learning rates, and how can you tell from watching the animation?',
    a: 'AdaGrad, RMSProp, Adam, and AdamW. Visually, non-adaptive algorithms take steps parallel to the gradient (plunging almost vertically into the ravine), whereas adaptive algorithms take diagonal steps pointing directly toward the global minimum (0,0) by normalizing anisotropic curvatures.',
  },
  {
    q: '5. What visual difference do you observe between AdaGrad and RMSProp past ~200 iterations?',
    a: 'AdaGrad accumulates all past squared gradients (Gₜ = ∑ g²), so its denominator grows indefinitely. Past ~200 steps, its step size decays to near zero, causing the optimizer ball to freeze before reaching (0,0). RMSProp uses an exponential moving average (EMA), maintaining a finite memory horizon so it continues smoothly to convergence.',
  },
  {
    q: '6. What visual difference do you observe between RMSProp and Adam?',
    a: 'RMSProp uses raw instantaneous gradients in the numerator, resulting in slight trajectory jitter on noisy or curved slopes. Adam incorporates a 1st-moment momentum buffer (m̂ₜ) plus bias correction, producing a silky-smooth trajectory without high-frequency jitter.',
  },
  {
    q: '7. On this simple 2D problem, does AdamW visibly differ from Adam? Why or why not?',
    a: "No, Adam and AdamW produce virtually identical trajectories on 2D paraboloids when λ=10⁻³. The unregularized minimum is already at (0,0), exactly matching the weight decay target, and gradient forces near (8,8) dwarf the 0.008 weight decay term. AdamW's distinct advantage emerges in overparameterized neural networks.",
  },
  {
    q: '8. As condition number increases from L₁ to L₄, which optimizers remain stable and which diverge at η=0.01?',
    a: 'At η=0.01, plain SGD, Momentum, and NAG explode/diverge on L₃ (κ=100) and L₄ (κ=1000) because stability requires η < 2/λₘₐₓ (on L₄, η < 0.001). Conversely, AdaGrad, RMSProp, Adam, and AdamW remain completely stable across all surfaces due to automatic denominator scaling.',
  },
];

const PART_B_QUESTIONS = [
  { q: '1. Why does plain SGD tend to zig-zag on ill-conditioned surfaces, echoed in NN curves?', a: 'High curvature anisotropy causes gradients to point almost perpendicular to the optimal path. In neural nets, ill-conditioned weight landscapes produce noisy loss oscillations and slow convergence for plain mini-batch SGD.' },
  { q: '2. How does Momentum reduce oscillation mechanically?', a: 'By averaging velocity vₜ = βvₜ₋₁ + (1-β)gₜ. Opposing gradient components cancel out across steps, while consistent components accumulate up to a 1/(1-β) = 10× speed multiplier.' },
  { q: '3. How is NAG different from Momentum, and is the difference visible?', a: 'NAG computes the gradient at the look-ahead point (θ − βv). If momentum is about to carry weights up an opposing slope, the look-ahead gradient applies predictive braking, visibly reducing overshoot near sharp loss valleys.' },
  { q: '4. Why does AdaGrad reduce learning rate for large-gradient parameters?', a: 'It divides step size by √(∑ g² + ε). Parameters with frequent/large updates accumulate huge denominators, preventing destabilizing updates on sensitive weights.' },
  { q: '5. Why does AdaGrad eventually become too slow? Did effective LR plot show this?', a: 'Because g² ≥ 0, Gₜ is monotonically non-decreasing. Over epochs, η_eff = η/√(Gₜ+ε) continuously decays toward zero. The effective LR chart in Part B clearly displays this downward plunge.' },
  { q: "6. How does RMSProp solve AdaGrad's main weakness?", a: 'It replaces the cumulative sum with an exponential moving average vₜ = β vₜ₋₁ + (1-β) gₜ², giving a finite memory horizon of ~10 steps and allowing effective LR to recover in flat regions.' },
  { q: '7. What are the roles of mₜ and vₜ in Adam?', a: 'mₜ (1st moment) tracks gradient direction and momentum. vₜ (2nd moment) tracks gradient variance/scale per parameter for coordinate-wise learning rate normalization.' },
  { q: '8. Why is bias correction required early in Adam?', a: 'Since m₀=0 and v₀=0, uncorrected averages are heavily biased toward zero (at t=1, v₁ = 0.001 g₁²). Dividing by (1 - βᵗ) rescales moments to unbiased estimators, preventing explosive initial steps.' },
  { q: '9. How does Adam combine ideas from Momentum and RMSProp?', a: 'Adam places momentum-averaged gradients (m̂ₜ) in the numerator and RMSProp variance scaling (√v̂ₜ + ε) in the denominator, with exact bias corrections.' },
  { q: '10. What is the purpose of decoupled weight decay in AdamW vs L2 in Adam?', a: 'Folding L2 into the gradient divides the penalty λθ by √v̂, weakening regularization on large-gradient weights. AdamW decouples weight decay (subtracting ηλθ directly), restoring uniform scale-invariant regularization.' },
  { q: '11. Which optimizer converged fastest in your dashboard?', a: 'Adam and AdamW converged fastest, reaching within 1% of final validation loss within 16–20 epochs.' },
  { q: '12. Which optimizer produced the best test performance?', a: 'AdamW and Momentum achieved the highest test accuracy (~98.25% on Breast Cancer test set).' },
  { q: '13. Is fastest convergence always best generalization?', a: 'No. Adaptive optimizers can sometimes converge into sharper local minima. Momentum explores flatter basins. AdamW combines fast adaptive convergence with optimal decoupled regularization.' },
  { q: '14. Which optimizer was most sensitive to the learning rate slider?', a: 'Plain SGD and Momentum. Increasing η from 0.01 to 0.1 or 0.5 caused immediate gradient explosion and divergence.' },
  { q: '15. What happens as condition number increases?', a: 'Non-adaptive optimizers suffer cross-valley divergence. Adaptive optimizers dynamically rescale coordinates and maintain stable loss descent.' },
  { q: '16. Which optimizer would you pick for a new deep-learning project and why?', a: 'AdamW. It delivers rapid, reliable convergence out-of-the-box across vision, NLP, and tabular deep learning models with theoretically sound weight regularization.' },
];

const CONCLUSION_STEPS = [
  { name: '1. SGD', desc: 'Established gradient descent baseline; struggles on anisotropic surfaces with high condition numbers.' },
  { name: '2. Momentum & NAG', desc: 'Introduced physical velocity and predictive braking to dampen oscillations and accelerate along flat valleys.' },
  { name: '3. AdaGrad', desc: 'Pioneered coordinate-wise adaptive learning rates, though unbounded historical sums cause premature freezing.' },
  { name: '4. RMSProp', desc: 'Fixed AdaGrad via exponential moving averages, keeping step sizes responsive to local curvature.' },
  { name: '5. Adam', desc: 'Unified Momentum (1st moment) and RMSProp (2nd moment) with exact finite-sample bias corrections.' },
  { name: '6. AdamW', desc: 'Restored theoretical purity to weight decay by decoupling it from adaptive variance scaling, delivering state-of-the-art generalization.' },
];

/* ── Shared Section Label ───────────────────────────────────── */
function SectionLabel({ code, title, count, color = '#A5B4FC' }) {
  return (
    <div className="theory-section-label">
      <span className="theory-section-code">{code}</span>
      <div className="theory-section-divider" />
      <span className="theory-section-title">{title}</span>
      {count && (
        <span className="theory-section-count">{count}</span>
      )}
    </div>
  );
}

/* ── Question Card ──────────────────────────────────────────── */
function QuestionCard({ item, idx, accentColor }) {
  return (
    <div className="theory-question-card">
      <div className="theory-question-text" style={{ color: accentColor }}>
        {item.q}
      </div>
      <div className="theory-answer-text">
        {item.a}
      </div>
    </div>
  );
}

/* ── Main Component ─────────────────────────────────────────── */
export default function LabReportAnswers() {
  const [activeSection, setActiveSection] = useState('partA');

  return (
    <div className="theory-scroll-container">
      <div className="theory-inner">

        {/* ── Page Header ── */}
        <div className="theory-header-card">
          <div className="theory-header-inner">
            <div className="theory-header-left">
              <div className="theory-page-label">
                <BookOpen size={13} className="theory-page-label-icon" />
                Theory &amp; Reflection Hub
              </div>
              <h2 className="theory-page-title">
                Deep Learning Laboratory
              </h2>
              <p className="theory-page-subtitle">
                From SGD to AdamW — Complete mathematical analysis &amp; verified laboratory answers (Sections A7 &amp; B4)
              </p>
            </div>

            {/* Section Switcher */}
            <div className="theory-tab-group">
              <button
                className={`theory-tab-btn ${activeSection === 'partA' ? 'active' : ''}`}
                onClick={() => setActiveSection('partA')}
              >
                <FlaskConical size={13} />
                Part A — Section A7
              </button>
              <button
                className={`theory-tab-btn ${activeSection === 'partB' ? 'active' : ''}`}
                onClick={() => setActiveSection('partB')}
              >
                <Cpu size={13} />
                Part B — Section B4
              </button>
              <button
                className={`theory-tab-btn ${activeSection === 'conclusion' ? 'active' : ''}`}
                onClick={() => setActiveSection('conclusion')}
              >
                <BookOpen size={13} />
                Conclusion
              </button>
            </div>
          </div>
        </div>

        {/* ── Part A Questions ── */}
        {activeSection === 'partA' && (
          <div className="theory-section-body">
            <SectionLabel
              code="SECTION A7"
              title="2D Loss Surface Reflection Questions"
              count="8 questions"
            />
            {PART_A_QUESTIONS.map((item, idx) => (
              <QuestionCard key={idx} item={item} idx={idx} accentColor="#818CF8" />
            ))}
          </div>
        )}

        {/* ── Part B Questions ── */}
        {activeSection === 'partB' && (
          <div className="theory-section-body">
            <SectionLabel
              code="SECTION B4"
              title="Neural Network Optimizer Benchmark Reflection Questions"
              count="16 questions"
            />
            {PART_B_QUESTIONS.map((item, idx) => (
              <QuestionCard key={idx} item={item} idx={idx} accentColor="#34D399" />
            ))}
          </div>
        )}

        {/* ── Conclusion ── */}
        {activeSection === 'conclusion' && (
          <div className="theory-conclusion-card">
            <div className="theory-conclusion-heading">
              One-Page Evolutionary Conclusion: SGD → AdamW
            </div>

            <p className="theory-conclusion-intro">
              The development of first-order optimization algorithms in deep learning represents a systematic engineering response to the geometric challenges of high-dimensional non-convex loss surfaces:
            </p>

            <div className="theory-conclusion-grid">
              {CONCLUSION_STEPS.map((st) => (
                <div key={st.name} className="theory-conclusion-step">
                  <div className="theory-conclusion-step-name">{st.name}</div>
                  <div className="theory-conclusion-step-desc">{st.desc}</div>
                </div>
              ))}
            </div>

            <div className="theory-conclusion-future">
              <div className="theory-conclusion-future-heading">
                <Wrench size={13} />
                Future Tool Improvements
              </div>
              <p className="theory-conclusion-future-body">
                With additional development time, we would implement <strong>interactive WebGL 3D topography rendering</strong> with real-time camera orbital controls and vector field overlays, non-convex benchmark functions (Rosenbrock, Rastrigin, Ackley) to study saddle point escape dynamics, dynamic learning rate schedulers (Cosine Annealing with Warm Restarts), and stochastic gradient noise simulation N(0, σ²) on 2D surfaces.
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
