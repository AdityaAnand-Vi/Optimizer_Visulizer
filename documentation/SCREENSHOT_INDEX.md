# Optimizer Lab — Visual Evidence Index

This document serves as an index for the visual evidence captured during the automated browser verification session. All screenshots are located in the `documentation/screenshots/` directory.

---

## PART A: 2D Playground

### 1. `part-a/part-a-main-playground.png`
- **Feature demonstrated**: Main 2D Playground interface.
- **What the screenshot proves**: The full application layout loads correctly, including the contour canvas, sidebar controls, and charting area.
- **Relevant source/module**: `frontend/src/components/PartAPlayground.jsx`
- **Relevant documentation section**: Part A — 2D Playground Details

### 2. `part-a/part-a-optimizer-selection.png`
- **Feature demonstrated**: Optimizer selection pills.
- **What the screenshot proves**: Users can toggle any combination of the 7 supported optimizers (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW).
- **Relevant source/module**: `frontend/src/components/PartAPlayground.jsx` (Sidebar Section)
- **Relevant documentation section**: Optimizer Implementations

### 3. `part-a/part-a-learning-rate.png`
- **Feature demonstrated**: Learning Rate Controls ($\eta$).
- **What the screenshot proves**: The UI allows custom precision input for the core $\eta$ hyperparameter.
- **Relevant source/module**: `frontend/src/components/PartAPlayground.jsx`
- **Relevant documentation section**: Optimizer Implementations

### 4. `part-a/part-a-starting-point.png`
- **Feature demonstrated**: Custom $(x_0, y_0)$ inputs.
- **What the screenshot proves**: The user can manually define the exact starting point for the mathematical simulation, rather than relying solely on canvas clicks.
- **Relevant source/module**: `backend/main.py` (`Simulate2DRequest`)
- **Relevant documentation section**: Starting-point handling

### 5. `part-a/part-a-trajectory.png`
- **Feature demonstrated**: 2D Contour Map Trajectories.
- **What the screenshot proves**: The frontend accurately draws the loss surface contours and overlays the simulated step-by-step path of each active optimizer.
- **Relevant source/module**: `backend/main.py` (`/api/simulate-2d`)
- **Relevant documentation section**: Part A — 2D Playground Details

### 6. `part-a/part-a-loss-curve.png`
- **Feature demonstrated**: Log-scale Loss Curve chart.
- **What the screenshot proves**: The frontend parses the loss history array and plots a synchronized 1D representation of convergence speed over time.
- **Relevant source/module**: `frontend/src/components/PartAPlayground.jsx`
- **Relevant documentation section**: Visualization implementation

### 7. `part-a/part-a-telemetry.png`
- **Feature demonstrated**: Status telemetry readout.
- **What the screenshot proves**: Dynamic UI updates during playback reflecting the current step count, current loss value, and total distance traveled.
- **Relevant source/module**: `frontend/src/components/PartAPlayground.jsx` (Math Block)
- **Relevant documentation section**: Experiment/iteration handling

### 8. `part-a/part-a-multiple-optimizers.png`
- **Feature demonstrated**: Multi-optimizer comparison.
- **What the screenshot proves**: The backend architecture successfully handles running up to 7 unique stateful optimizers simultaneously on the same surface.
- **Relevant source/module**: `backend/optimizers.py`
- **Relevant documentation section**: Optimizer Implementations

---

## EXPERIMENTS

### 9. `experiments/part-a-stable-convergence.png`
- **Feature demonstrated**: Stable convergence behavior.
- **What the screenshot proves**: Shows the Momentum optimizer gracefully slowing down and hitting the mathematical minima without exploding, correctly triggering the `CONVERGED` state.
- **Relevant source/module**: `backend/main.py` (`simulate_2d` convergence logic)
- **Relevant documentation section**: Convergence calculations

### 10. `experiments/part-a-divergence.png`
- **Feature demonstrated**: Handled numerical divergence.
- **What the screenshot proves**: Running SGD with an aggressively high learning rate on a steep surface correctly triggers the backend's `DIVERGED` detection (NaN/Inf bounding) and the UI elegantly reports the failure rather than crashing.
- **Relevant source/module**: `backend/main.py` (Divergence bounds checking)
- **Relevant documentation section**: Divergence detection

---

## PART B: Neural Benchmark

### 11. `part-b/part-b-main.png`
- **Feature demonstrated**: Initial Benchmark UI.
- **What the screenshot proves**: The Part B dashboard loads successfully with the dataset metadata (455 Train Samples, 114 Test Samples, 30 Features).
- **Relevant source/module**: `frontend/src/components/PartBDashboard.jsx`
- **Relevant documentation section**: Part B — Neural Benchmark Details

### 12. `part-b/part-b-training-config.png`
- **Feature demonstrated**: Epochs and Batch Size configuration.
- **What the screenshot proves**: The sidebar exposes controls for adjusting the Neural Network's training loop parameters before firing the request.
- **Relevant source/module**: `backend/main.py` (`TrainNNRequest`)
- **Relevant documentation section**: Training configuration (Batch size / Epoch handling)

### 13. `part-b/part-b-training-loss.png`
- **Feature demonstrated**: Training Loss Graph (BCE).
- **What the screenshot proves**: Successfully visualizes the model's ability to fit the training data across all selected optimizers simultaneously.
- **Relevant source/module**: `backend/mlp.py` (`binary_cross_entropy`)
- **Relevant documentation section**: Benchmark metrics

### 14. `part-b/part-b-validation-loss.png`
- **Feature demonstrated**: Validation (Test) Loss Graph.
- **What the screenshot proves**: Tracks generalization error on held-out data, visually distinguishing optimizers that overfit from those that generalize well.
- **Relevant source/module**: `frontend/src/components/PartBDashboard.jsx`
- **Relevant documentation section**: Benchmark metrics

### 15. `part-b/part-b-test-accuracy.png`
- **Feature demonstrated**: Test Accuracy Graph.
- **What the screenshot proves**: The manual NumPy MLP implementation successfully learns the Breast Cancer classification task, reaching >95% accuracy.
- **Relevant source/module**: `backend/mlp.py` (`predict_class`)
- **Relevant documentation section**: Neural-network architecture

### 16. `part-b/part-b-results-table.png`
- **Feature demonstrated**: Final Results Summary.
- **What the screenshot proves**: The backend accurately returns the final calculated metrics (Final Train/Test Loss, Accuracy, and Convergence Epoch) for direct tabular comparison.
- **Relevant source/module**: `backend/main.py` (`calculate_convergence_epoch`)
- **Relevant documentation section**: Benchmark metrics

---

## PART C: Theory Hub

### 17. `part-c/part-c-theory.png`
- **Feature demonstrated**: Main Theory and Reflection Hub.
- **What the screenshot proves**: The educational layout integrates cleanly into the React Router structure.
- **Relevant source/module**: `frontend/src/components/LabReportAnswers.jsx`
- **Relevant documentation section**: Part C — Theory & Reflection Hub

### 18. `part-c/part-c-reflection.png`
- **Feature demonstrated**: Reflection questions.
- **What the screenshot proves**: Interactive elements like accordions and Q&A blocks function correctly to support learning.
- **Relevant source/module**: `frontend/src/components/LabReportAnswers.jsx`
- **Relevant documentation section**: Any other important educational functionality

### 19. `part-c/part-c-educational.png`
- **Feature demonstrated**: Advanced Educational/Modal Interactions.
- **What the screenshot proves**: The application supports deeper learning modes (like the interactive Guide & Hotkeys modal) that layer over the simulation tools without breaking them.
- **Relevant source/module**: `frontend/src/components/ExplainAsYouGo.jsx`
- **Relevant documentation section**: Part C — Theory & Reflection Hub
