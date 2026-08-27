# Optimizer Documentation

This document mathematically and technically details the 7 optimizers implemented from scratch in the Optimizer Lab. All equations strictly reflect the actual `backend/optimizers.py` implementation, not generic theoretical textbooks.

---

## 1. SGD (Stochastic Gradient Descent)

### Overview
Standard gradient descent without momentum or adaptive learning rates.

### Purpose
To step directly in the opposite direction of the gradient at the current position.

### Mathematical update rule
$$ \theta_t = \theta_{t-1} - \eta \cdot g_t $$

### Explanation of every variable
- $\theta_t$: Parameters at current step.
- $\theta_{t-1}$: Parameters at previous step.
- $\eta$: Learning rate (`self.lr`).
- $g_t$: Gradient at the current parameters.

### Step-by-step algorithm
1. Receive parameters and gradient.
2. Multiply gradient by learning rate.
3. Subtract from parameters.
4. Return new parameters.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.

### Exact implementation used in this project
```python
def step(self, params, grad):
    params = np.asarray(params, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    return params - self.lr * grad
```

### Advantages
Computationally extremely cheap. Requires no extra memory for state.

### Limitations
Struggles heavily in steep ravines (high condition number surfaces), oscillating across the walls instead of moving down the center.

### Expected behavior
Oscillates in ravines, crawls on plateaus.

### Observed behavior in Optimizer Lab
Confirmed in Experiment 1 and 6. SGD oscillates violently on $L3: c=100$ and diverges entirely if $\eta$ is pushed to $0.1$.

### Relevant screenshot
`documentation/screenshots/experiments/part-a-divergence.png`

### Relevant experiment
Experiment 1 (Learning Rate).

### Implementation notes
Purely stateless. The `reset()` method does nothing.

---

## 2. SGD Momentum

### Overview
Gradient descent with a velocity accumulator to damp oscillations.

### Purpose
To build up speed in directions with consistent gradients and damp out oscillations in directions with alternating gradients.

### Mathematical update rule
$$ v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t $$
$$ \theta_t = \theta_{t-1} - \eta \cdot v_t $$

*(Note: Some textbooks define momentum as $v_t = \beta v_{t-1} + \eta g_t$. The Optimizer Lab uses the PyTorch-style exponentially moving average variant for velocity.)*

### Explanation of every variable
- $v_t$: Velocity vector at current step.
- $\beta$: Momentum decay factor (`self.beta`).

### Step-by-step algorithm
1. Initialize velocity $v$ to zero on first step.
2. Update velocity as a weighted average of past velocity and current gradient.
3. Subtract scaled velocity from parameters.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.
- **Beta ($\beta$)**: Default `0.9`.

### Exact implementation used in this project
```python
def step(self, params, grad):
    if self.v is None:
        self.v = np.zeros_like(params)
    self.v = self.beta * self.v + (1.0 - self.beta) * grad
    return params - self.lr * self.v
```

### Advantages
Dramatically speeds up convergence in ravines compared to standard SGD.

### Limitations
Can overshoot the minimum due to accumulated velocity.

### Expected behavior
Smooths out oscillations, curves gracefully towards the minima.

### Observed behavior in Optimizer Lab
Confirmed in Experiment 3. Reached a final loss of $3.16 \times 10^{-10}$ in 500 steps, vastly outperforming SGD.

### Relevant screenshot
`documentation/screenshots/experiments/part-a-stable-convergence.png`

### Relevant experiment
Experiment 3 (Momentum vs SGD).

### Implementation notes
Stateful. The `v` vector must be cleared via `reset()` between experiments.

---

## 3. NAG (Nesterov Accelerated Gradient)

### Overview
Momentum that "looks ahead" before calculating the gradient.

### Purpose
To prevent the severe overshooting inherent in standard momentum by applying a corrective gradient calculated at the anticipated future position.

### Mathematical update rule
$$ \theta_{lookahead} = \theta_{t-1} - \beta \cdot v_{t-1} $$
$$ v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_{lookahead} $$
$$ \theta_t = \theta_{t-1} - \eta \cdot v_t $$

### Explanation of every variable
- $\theta_{lookahead}$: The parameters shifted forward by the current momentum.
- $g_{lookahead}$: The gradient evaluated at $\theta_{lookahead}$.

### Step-by-step algorithm
1. Expose a `get_lookahead_params(params)` method to the training loop.
2. The loop calculates the gradient at this lookahead point.
3. Update velocity using the lookahead gradient.
4. Update parameters.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.
- **Beta ($\beta$)**: Default `0.9`.

### Exact implementation used in this project
```python
def get_lookahead_params(self, params):
    if self.v is None:
        self.v = np.zeros_like(params)
    return params - self.beta * self.v

def step(self, params, grad):
    # 'grad' here is expected to be g_lookahead
    if self.v is None:
        self.v = np.zeros_like(params)
    self.v = self.beta * self.v + (1.0 - self.beta) * grad
    return params - self.lr * self.v
```

### Advantages
Better theoretical convergence rates for convex functions than standard momentum.

### Limitations
Requires architectural support to calculate gradients at a modified parameter state, which complicates standard training loops.

### Expected behavior
Sharper corrections when approaching the minima compared to standard momentum.

### Observed behavior in Optimizer Lab
In Experiment 2 on $L2$, NAG exhibited instability with the default $\beta=0.9$, diverging in 4 steps. This demonstrates the fragility of Nesterov on poorly scaled ravines.

### Relevant screenshot
`documentation/screenshots/part-a/part-a-multiple-optimizers.png`

### Relevant experiment
Experiment 2.

### Implementation notes
The backend `main.py` specifically intercepts NAG to call `opt.get_lookahead_params()` before passing the gradient to `opt.step()`.

---

## 4. AdaGrad

### Overview
Adaptive gradient algorithm that scales the learning rate per parameter based on historical gradients.

### Purpose
To perform larger updates for infrequent parameters and smaller updates for frequent parameters.

### Mathematical update rule
$$ G_t = G_{t-1} + g_t^2 $$
$$ \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t $$

### Explanation of every variable
- $G_t$: Accumulator for the sum of squared gradients.
- $\epsilon$: A small constant to prevent division by zero.

### Step-by-step algorithm
1. Square the current gradient and add it to $G$.
2. Divide the learning rate by $\sqrt{G} + \epsilon$.
3. Update parameters using this effective learning rate.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.
- **Epsilon ($\epsilon$)**: Default `1e-8`.

### Exact implementation used in this project
```python
def step(self, params, grad):
    if self.G is None:
        self.G = np.zeros_like(params)
    self.G += grad**2
    return params - self.lr * grad / (np.sqrt(self.G) + self.eps)
```

### Advantages
Eliminates the need to manually tune learning rate schedules.

### Limitations
$G_t$ grows monotonically, causing the learning rate to shrink to zero, eventually freezing the network prematurely.

### Expected behavior
Rapid initial progress followed by extreme slowdown.

### Observed behavior in Optimizer Lab
In Experiment 4, AdaGrad froze early, finishing with a high loss of 2923.59 compared to RMSProp.

### Relevant screenshot
`documentation/screenshots/part-a/part-a-multiple-optimizers.png`

### Relevant experiment
Experiment 4.

### Implementation notes
$G$ is initialized to zeros. 

---

## 5. RMSProp

### Overview
An adaptive learning rate method proposed by Geoff Hinton that resolves AdaGrad's diminishing learning rate problem.

### Purpose
To normalize gradients by an exponentially decaying average of squared gradients, rather than a cumulative sum.

### Mathematical update rule
$$ v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2 $$
$$ \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t $$

### Explanation of every variable
- $v_t$: Exponentially weighted moving average of squared gradients.

### Step-by-step algorithm
1. Update $v_t$ with the squared gradient.
2. Divide learning rate by $\sqrt{v_t} + \epsilon$.
3. Update parameters.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.
- **Beta ($\beta$)**: Default `0.9`.
- **Epsilon ($\epsilon$)**: Default `1e-8`.

### Exact implementation used in this project
```python
def step(self, params, grad):
    if self.v is None:
        self.v = np.zeros_like(params)
    self.v = self.beta * self.v + (1.0 - self.beta) * (grad**2)
    return params - self.lr * grad / (np.sqrt(self.v) + self.eps)
```

### Advantages
Maintains a healthy effective learning rate throughout training.

### Limitations
Does not utilize momentum for the gradient itself, only for the squared gradient accumulator.

### Expected behavior
Smooth progress across ravines without freezing.

### Observed behavior in Optimizer Lab
Outperformed AdaGrad significantly in Experiment 4 (Loss 457.25 vs 2923.59). In the Neural Benchmark (Experiment 5), it achieved 100% training accuracy.

### Relevant screenshot
`documentation/screenshots/part-b/part-b-results-table.png`

### Relevant experiment
Experiment 4 and 5.

### Implementation notes
Implemented using `1 - beta` weighting on the current squared gradient, similar to standard deep learning frameworks.

---

## 6. Adam (Adaptive Moment Estimation)

### Overview
Combines the benefits of RMSProp and Momentum, adding bias-correction for early iterations.

### Purpose
To compute adaptive learning rates for each parameter using both the first and second moments of the gradients.

### Mathematical update rule
$$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$
$$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$
$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
$$ \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

### Explanation of every variable
- $m_t$: 1st moment (momentum).
- $v_t$: 2nd moment (variance).
- $\hat{m}_t, \hat{v}_t$: Bias-corrected moments.
- $t$: Current step count.

### Step-by-step algorithm
1. Increment step count $t$.
2. Update $m$ and $v$.
3. Compute bias-corrected $\hat{m}$ and $\hat{v}$.
4. Update parameters using the corrected moments.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.
- **Beta1 ($\beta_1$)**: Default `0.9`.
- **Beta2 ($\beta_2$)**: Default `0.999`.
- **Epsilon ($\epsilon$)**: Default `1e-8`.

### Exact implementation used in this project
```python
def step(self, params, grad):
    if self.m is None:
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
    self.t += 1
    self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
    m_hat = self.m / (1.0 - self.beta1**self.t)
    v_hat = self.v / (1.0 - self.beta2**self.t)
    return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### Advantages
Highly robust default optimizer for almost all neural network architectures.

### Limitations
Can generalize worse than SGD with Momentum on some image classification tasks.

### Expected behavior
Fast initial convergence, resisting the rapid freezing of AdaGrad.

### Observed behavior in Optimizer Lab
Converged reliably in Part A. Reached 100% training accuracy in Experiment 5.

### Relevant screenshot
`documentation/screenshots/part-a/part-a-main-playground.png`

### Relevant experiment
Experiment 2 and 5.

### Implementation notes
Fully implements the required bias correction using $1 - \beta^t$.

---

## 7. AdamW

### Overview
Adam with decoupled weight decay.

### Purpose
To fix the improper handling of L2 regularization in standard Adam, applying weight decay directly to the parameters rather than mixing it into the gradient moments.

### Mathematical update rule
$$ \text{Moments calculated same as Adam} $$
$$ \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right) $$

### Explanation of every variable
- $\lambda$: Weight decay coefficient (`self.weight_decay`).

### Step-by-step algorithm
1. Compute moments exactly as Adam.
2. Apply the adaptive update.
3. Subtract an additional $\eta \cdot \lambda \cdot \theta_{t-1}$ from the parameters.

### Hyperparameters
- **Learning Rate ($\eta$)**: Default `0.01`.
- **Beta1 ($\beta_1$)**: Default `0.9`.
- **Beta2 ($\beta_2$)**: Default `0.999`.
- **Epsilon ($\epsilon$)**: Default `1e-8`.
- **Weight Decay ($\lambda$)**: Default `0.001`.

### Exact implementation used in this project
```python
def step(self, params, grad):
    self.t += 1
    self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
    m_hat = self.m / (1.0 - self.beta1**self.t)
    v_hat = self.v / (1.0 - self.beta2**self.t)
    return params - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params)
```

### Advantages
Better generalization on complex datasets (e.g., Transformers, modern CNNs) compared to standard Adam.

### Limitations
Requires tuning an additional hyperparameter ($\lambda$).

### Expected behavior
Identical to Adam on training loss, but theoretically better on validation loss for complex models.

### Observed behavior in Optimizer Lab
In Experiment 5, AdamW reached a Test Loss of 0.2166 vs Adam's 0.2854, demonstrating the generalization benefit.

### Relevant screenshot
`documentation/screenshots/part-b/part-b-training-loss.png`

### Relevant experiment
Experiment 5.

### Implementation notes
The weight decay is decoupled and applied *after* the adaptive step fraction is computed, multiplied by the learning rate.
