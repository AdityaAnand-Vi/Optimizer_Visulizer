# Part A Documentation — 2D Playground

## 1. Purpose of Part A
Part A serves as an interactive sandbox for students to develop an intuitive geometric understanding of optimization algorithms. By visualizing parameter updates on a simplified 2D quadratic loss surface, students can observe the mechanics of gradient descent without the opacity of high-dimensional neural networks.

## 2. User Interface
The interface is split into a control sidebar on the left and a visualization suite on the right. The visualization suite consists of a top-down contour map showing the $(x, y)$ coordinate space and a 1D loss-vs-iteration chart.
*(Screenshot: `documentation/screenshots/part-a/part-a-main-playground.png`)*

## 3. Available Controls
- **Optimizer Selection**: Pills to toggle any combination of the 7 supported algorithms.
- **Learning Rate**: Dropdown and preset buttons to adjust the base step size ($\eta$).
- **Experiment Length**: Controls to set `max_steps`.
- **Loss Surface**: Dropdown to alter the mathematical steepness (condition number) of the valley.
*(Screenshot: `documentation/screenshots/part-a/part-a-optimizer-selection.png`)*

## 4. Loss Surface
The underlying mathematical function simulated by the backend is a quadratic bowl:
$$ L(x, y) = x^2 + c \cdot y^2 $$
Where $c$ is the condition number. A higher $c$ (e.g., $L4: c=1000$) creates a steep ravine that is difficult for standard SGD to navigate.

## 5. Starting Point
Users can manually input $(x_0, y_0)$ coordinates or click directly on the contour canvas to set the initialization parameters for the optimizers.
*(Screenshot: `documentation/screenshots/part-a/part-a-starting-point.png`)*

## 6. Gradient Calculation
Gradients are calculated analytically in `backend/main.py` rather than via autograd:
$$ \nabla L = \begin{bmatrix} 2x \\ 2cy \end{bmatrix} $$

## 7. Parameter Update
Parameters are passed to the selected optimizer classes in `backend/optimizers.py` inside a fixed iteration loop up to `max_steps`. Each optimizer maintains its own internal state (e.g., momentum velocity or Adam's moments) throughout the loop.

## 8. Optimizer Trajectories
The resulting parameter coordinates are stored in an array and passed back to the frontend, which draws them as lines traversing the SVG/Canvas contour map.
*(Screenshot: `documentation/screenshots/part-a/part-a-trajectory.png`)*

## 9. Loss-vs-Iteration Graph
The actual loss values $L(x_t, y_t)$ at each step are plotted on a logarithmic scale. This allows users to compare the convergence speed (vertical drop) of different optimizers over time (horizontal axis).
*(Screenshot: `documentation/screenshots/part-a/part-a-loss-curve.png`)*

## 10. Telemetry
The UI displays a live readout of the active optimizer's current status, step count, loss value, and absolute distance traveled.
*(Screenshot: `documentation/screenshots/part-a/part-a-telemetry.png`)*

## 11. Experiment Length
The user can specify how long the simulation runs before terminating. The default is 500 steps, which is sufficient for most adaptive optimizers to reach the minima on the $L2$ surface.

## 12. Playback System
To enhance educational value, the frontend features an animated playback system (Play, Pause, Step). Rather than retraining the math on the fly, the frontend incrementally slices the fully computed trajectory array received from the backend, guaranteeing frame-perfect synchronization across the contour map and loss curve without network latency.

## 13. Learning-Rate Experiments
Students can alter the learning rate to observe its profound impact on stability.
*(Screenshot: `documentation/screenshots/part-a/part-a-learning-rate.png`)*

## 14. Optimizer Comparison
By enabling multiple optimizers simultaneously, students can directly race them against each other on the identical surface.
*(Screenshot: `documentation/screenshots/part-a/part-a-multiple-optimizers.png`)*

## 15. Convergence
The backend explicitly flags a trajectory as `CONVERGED` when the loss drops below $1 \times 10^{-6}$ or the squared gradient magnitude drops below $1 \times 10^{-10}$.

## 16. Divergence
If an optimizer takes a step that results in numerical explosion (`NaN` or `Inf`), or escapes the absolute bounds of $1 \times 10^6$, the backend halts execution for that specific optimizer and flags it as `DIVERGED`. The frontend smoothly catches this and displays a warning, preventing the application from crashing.
*(Screenshot: `documentation/screenshots/experiments/part-a-divergence.png`)*

## 17. Educational Interpretation
Students are expected to learn that:
1. Standard SGD oscillates wildly in steep ravines.
2. Momentum significantly dampens these oscillations.
3. Adaptive methods (Adam/RMSProp) are robust against poor learning rate choices but can exhibit strange geometric paths.
4. Too high of a learning rate causes divergence, while too low causes freezing.
