# Optimizer Lab: Experimental Results

This document contains reproducible empirical evidence gathered directly from the Optimizer Lab runtime environment (backend version 2.0). All numbers reflect the exact outputs of the custom pure-NumPy optimization loop.

## How to Reproduce the Experiments
Every experiment below includes the exact configuration used. You can reproduce these results exactly by launching the application (`npm run dev` and `python -m uvicorn main:app`) and entering these exact parameters into the UI.

---

## EXPERIMENT 1 — Learning Rate

**Objective:** Compare a small, moderate, and large learning rate.
**Configuration:**
- **Optimizer:** SGD
- **Loss Surface:** L2: c = 50 (default)
- **Starting Point:** (8.0, 8.0)
- **Iterations:** 500

| Learning Rate ($\eta$) | Final Loss | Final Position $(x, y)$ | Status |
|-----------------------|------------|-------------------------|--------|
| **0.001 (Small)**     | 8.644      | $(2.94, 1.05 \times 10^{-22})$ | RUNNING |
| **0.01 (Moderate)**   | $1.07 \times 10^{-7}$ | $(0.0003, 0.0)$ | CONVERGED |
| **0.1 (Large)**       | $2.09 \times 10^{7}$ | $(5.12, 648.0)$ | DIVERGED |

**Screenshot:** `documentation/screenshots/part-a/part-a-learning-rate.png`

---

## EXPERIMENT 2 — Optimizer Comparison

**Objective:** Compare all implemented optimizers using the same starting conditions.
**Configuration:**
- **Loss Surface:** L2: c = 50 (default)
- **Starting Point:** (8.0, 8.0)
- **Iterations:** 500
- **Learning Rate:** 0.01

| Optimizer | Final Loss | Final Position | Steps to Stop | Status |
|-----------|------------|----------------|---------------|--------|
| **SGD** | $1.07 \times 10^{-7}$ | $(0.0003, 0.0)$ | 500 | CONVERGED |
| **Momentum** | $3.16 \times 10^{-10}$ | $(1.77 \times 10^{-5}, -2.30 \times 10^{-11})$ | 500 | CONVERGED |
| **NAG** | $6.36 \times 10^{6}$ | $(7.87, 356.91)$ | 4 | DIVERGED |
| **AdaGrad** | 2923.59 | $(7.57, 7.57)$ | 500 | RUNNING |
| **RMSProp** | 457.25 | $(2.99, 2.99)$ | 500 | RUNNING |
| **Adam** | 725.90 | $(3.77, 3.77)$ | 500 | RUNNING |
| **AdamW** | 717.16 | $(3.74, 3.74)$ | 500 | RUNNING |

**Screenshot:** `documentation/screenshots/part-a/part-a-multiple-optimizers.png`

---

## EXPERIMENT 3 — Momentum

**Objective:** Compare SGD and SGD Momentum under identical conditions.
**Configuration:** (Same as Experiment 2)

**Result:** Momentum achieved a significantly lower final loss ($3.16 \times 10^{-10}$) compared to standard SGD ($1.07 \times 10^{-7}$), demonstrating the acceleration effect on the narrow valley floor. 

**Screenshot:** `documentation/screenshots/experiments/part-a-stable-convergence.png`

---

## EXPERIMENT 4 — Adaptive Optimization

**Objective:** Compare AdaGrad, RMSProp, Adam, and AdamW.
**Configuration:** (Same as Experiment 2)

**Result:** On the standard L2 surface, RMSProp (Loss: 457.25) outperformed AdaGrad (Loss: 2923.59) due to AdaGrad's rapidly decaying learning rate. Adam (725.90) and AdamW (717.16) performed similarly on the 2D surface where weight decay has a minor regularization effect. All adaptive optimizers required more than 500 steps to converge with $\eta=0.01$ due to the high condition number ($c=50$).

---

## EXPERIMENT 5 — Neural Benchmark

**Objective:** Run the benchmark using the application's standard configuration on the Breast Cancer dataset.
**Configuration:**
- **Epochs:** 100
- **Batch Size:** 32
- **Learning Rate:** 0.01

| Optimizer | Final Train Loss | Final Test Loss | Final Train Acc | Final Test Acc | Convergence Epoch |
|-----------|------------------|-----------------|-----------------|----------------|-------------------|
| **SGD** | 0.0573 | 0.0974 | 98.90% | 97.36% | 98 |
| **Momentum** | 0.0579 | 0.0975 | 98.90% | 97.36% | 98 |
| **NAG** | 0.0579 | 0.0975 | 98.90% | 97.36% | 98 |
| **AdaGrad** | 0.0267 | 0.0742 | 99.34% | 95.61% | 93 |
| **RMSProp** | 0.0007 | 0.1976 | 100.0% | 97.36% | 100 |
| **Adam** | 0.0061 | 0.2854 | 100.0% | 95.61% | 100 |
| **AdamW** | 0.0009 | 0.2166 | 100.0% | 94.73% | 100 |

**Screenshot:** `documentation/screenshots/part-b/part-b-results-table.png`

---

## EXPERIMENT 6 — Divergence

**Objective:** Identify a configuration that produces instability and document exactly what happens.
**Configuration:**
- **Optimizer:** SGD
- **Loss Surface:** L3: c = 100
- **Starting Point:** (8.0, 8.0)
- **Learning Rate:** 0.1

**Observed Result:** The simulation halted early. The final recorded loss was $2.31 \times 10^{6}$ at position $(6.4, -152.0)$. The telemetry UI flagged the state as `DIVERGED` with the message: *"DIVERGED — SGD diverged! Step size was too aggressive for steep valley walls."*

**Screenshot:** `documentation/screenshots/experiments/part-a-divergence.png`
