# ⚡ Optimizer Visualizer & Benchmark Suite

An interactive web application built with Streamlit and NumPy to visualize, analyze, and benchmark **7 optimization algorithms** implemented entirely from scratch without external autograd libraries.

---

## 🌟 Supported Optimizers

1. **SGD** (Stochastic Gradient Descent)
2. **Momentum** (SGD with Momentum)
3. **NAG** (Nesterov Accelerated Gradient)
4. **AdaGrad** (Adaptive Gradient Algorithm)
5. **RMSProp** (Root Mean Square Propagation)
6. **Adam** (Adaptive Moment Estimation)
7. **AdamW** (Adam with Decoupled Weight Decay)

---

## 📁 Repository & Code Architecture (Separation of Concerns)

```text
Optimizer Visualizer/
├── optimizers.py          # Pure NumPy optimization algorithm implementations (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW)
├── mlp.py                 # From-scratch 3-layer MLP neural network, loss function, manual backpropagation, and parameter flattening interface
├── part_a_playground.py   # Part A: 2D Loss Surface Playground visualization module & interactive controls
├── part_b_dashboard.py    # Part B: Neural Network Optimizer Benchmark dashboard & real-time training metrics
├── app.py                 # Main Streamlit application entry point with tabbed navigation & shared Simple/Advanced mode toggle
└── README.md              # Project documentation, architecture overview, and user guide
```

### Separation of Concerns:
- **`optimizers.py`**: Pure optimization math operating on 1D NumPy parameter arrays ($\theta$). No UI or neural network code.
- **`mlp.py`**: Neural network architecture (`Dense` layers, `ReLU`, `Sigmoid`, `BCE` loss, manual chain-rule backpropagation). Exposes `get_flat_params()`, `set_flat_params()`, and `flatten_grads()` so any optimizer from `optimizers.py` can train the network unmodified. No UI code.
- **`part_a_playground.py` & `part_b_dashboard.py`**: Pure visualization and UI layout scripts consuming `optimizers.py` and `mlp.py`.
- **`app.py`**: Top-level application coordinator managing shared session state (`Simple mode` vs `Advanced mode`) and tab routing.

---

## 🚀 How to Run the Application

You can run the suite using either the **Streamlit** dashboard or the **React + FastAPI** web app:

### Option 1: Streamlit Dashboard (Standard Entry Point)
Ensure dependencies (`streamlit`, `numpy`, `matplotlib`, `scikit-learn`) are installed in your Python environment:

```bash
streamlit run app.py
```

### Option 2: Full-Stack React + FastAPI Web Application
1. **Start the FastAPI Backend**:
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```
2. **Start the React Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

---

## 📖 How to Use the Tool

### 📑 Tab Navigation

#### 1. ⚡ Part A: Optimizer Playground
- **2D Contour Plot**: Visualizes each optimizer's trajectory on quadratic loss surfaces ($L(x,y) = x^2 + c \cdot y^2$) starting from $(x_0, y_0)$ towards $(0,0)$.
- **Ball-Rolling Visuals**: Each optimizer is rendered as a weighted ball leaving a fading 30-step trajectory trail.
- **Live Gradient Arrows**: Toggleable downhill arrows ($-\nabla L$) pointing in the direction of steepest descent.
- **Loss Decay Curve**: Real-time line plot comparing loss reduction rates on log-scale.
- **Status Diagnosis**: Rule-based automated captions explaining why an optimizer converged, oscillated, or diverged.

#### 2. 🧠 Part B: Neural Network Dashboard
- **Dataset Integration**: Trains an MLP (Input $\to$ 16 $\to$ 8 $\to$ 1) on the **Breast Cancer Wisconsin Dataset** (30 features, 455 train / 114 test samples).
- **Live Overlaid Charts**: 
  - Training Loss vs Epoch (log scale)
  - Validation / Test Loss vs Epoch (log scale)
  - Test Accuracy (%) vs Epoch
  - Effective Learning Rate per Epoch ($\eta_{\text{eff}} = \eta / (\sqrt{\hat{v}_0} + \epsilon)$ for adaptive algorithms in Advanced mode)
- **Automatic Convergence Epoch**: Computes the first epoch where test loss reaches and stays within 1% of its final value.

---

## 🕹️ Controls & Features

- **App Mode Toggle**:
  - **Simple Mode**: Hides hyperparameter sliders ($\beta, \beta_1, \beta_2, \lambda$), hides technical effective learning rate plots, and provides plain-English tooltips/captions.
  - **Advanced Mode**: Unlocks all hyperparameter sliders, technical explanation panels, and adaptive learning rate plots.
- **LR Sensitivity Presets**: Instant preset buttons ($\eta = 0.001$, $0.01$, $0.1$) to quickly compare convergence vs oscillation vs divergence.
- **Input Validation**: All user inputs (learning rate, epochs, batch size, coordinates, weight decay) strictly validate positivity and numerical sanity.

