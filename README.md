# ⚡ Optimizer Visualizer & Benchmark Suite (React + FastAPI Edition)

An interactive, responsive web application built with **React**, **FastAPI**, and **NumPy** to visualize, analyze, and benchmark **7 optimization algorithms** implemented entirely from scratch (without external autograd libraries).

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

## 📁 Repository & Code Architecture

This project strictly follows a separation of concerns, decoupling the pure mathematical backend from the interactive premium React frontend.

```text
Optimizer Visualizer/
├── backend/                 # FastAPI + pure NumPy ML models
│   ├── main.py              # API endpoints (simulate-2d, train-nn)
│   ├── optimizers.py        # Pure NumPy optimization algorithm implementations
│   └── mlp.py               # From-scratch 3-layer MLP, manual backprop
├── frontend/                # Vite + React (UI, State, Animations)
│   ├── src/
│   │   ├── components/
│   │   │   ├── PartAPlayground.jsx    # 2D Loss Surface & unlimited length experiments
│   │   │   ├── PartBDashboard.jsx     # Neural Network Dashboard with animated playback
│   │   │   ├── ExplainAsYouGo.jsx     # Dynamic educational insights
│   │   │   └── ConditioningExplorer.jsx
│   │   ├── App.jsx          # Routing & shared state
│   │   └── index.css        # Premium UI design tokens & styling
└── README.md                # Project documentation
```

### Separation of Concerns:
- **`optimizers.py`**: Pure optimization math operating on 1D NumPy parameter arrays ($\theta$). No UI or neural network code.
- **`mlp.py`**: Neural network architecture (`Dense` layers, `ReLU`, `Sigmoid`, `BCE` loss, manual chain-rule backpropagation). Exposes `get_flat_params()`, `set_flat_params()`, and `flatten_grads()`.
- **React Frontend**: Manages all UI, 60fps animations, crosshair synchronization, user input handling, and responsive layouts.

---

## 🚀 How to Run the Application

This application is built as a full-stack system. You must run both the backend and frontend servers simultaneously.

### 1. Start the FastAPI Backend
Ensure dependencies (`fastapi`, `uvicorn`, `numpy`, `scikit-learn`) are installed.
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### 2. Start the React Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

---

## 📖 Features & Capabilities

### 📑 1. Part A: Optimizer Playground (2D Loss Surface)
- **Interactive Topography**: Visualizes each optimizer's trajectory on a quadratic loss surface. Supports crosshair hovering for coordinate inspections.
- **Unlimited Experiment Lengths**: Users can select presets (100, 500, 1000, 5000) or specify custom iteration lengths to analyze long-term convergence behavior.
- **Ball-Rolling Visuals**: Each optimizer is rendered as a weighted ball leaving a dynamic trajectory trail.
- **Live Gradient Arrows**: Toggleable downhill arrows pointing in the direction of steepest descent.
- **Loss Decay Curve**: Real-time line plot comparing loss reduction rates on a log-scale, synced via hover interactions.

### 📑 2. Part B: Neural Network Benchmark
- **Dataset Integration**: Trains an MLP (Input $\to$ 16 $\to$ 8 $\to$ 1) on the **Breast Cancer Wisconsin Dataset** (30 features, 455 train / 114 test samples).
- **Animated Training Playback**: Completely decoupled from training execution. The backend trains the model once, and the frontend animates the historical results.
  - **Controls**: Play, Pause, Step (frame-by-frame), Replay, Scrubbing.
  - **Variable Speeds**: 0.5x, 1x, 2x, 4x.
- **Live Overlaid Charts**: 
  - Training Loss & Test Loss (BCE)
  - Test Accuracy (%)
  - Effective Learning Rate ($\eta_{\text{eff}}$) for adaptive algorithms.
  - *All charts feature synced crosshair hovering.*
- **Automated Insights ("What did we learn?")**: Dynamically generates educational observations (e.g., "AdamW showed signs of overfitting", "SGD converged the fastest") based on the live benchmark calculations.
- **Summary Metrics Deck**: Instantly identifies the Best Test Accuracy, Fastest Convergence epoch, and Lowest Test Loss.

---

## 🕹️ General UI Controls

- **App Mode Toggle**:
  - **Beginner Mode**: Hides complex hyperparameters ($\beta, \lambda$) and effective learning rate charts, offering plain-English tooltips and simplified interfaces.
  - **Advanced Mode**: Unlocks all hyperparameter sliders, technical explanation panels, and adaptive learning rate metrics for deep scientific analysis.
- **Responsive Layout**: Designed for seamless viewing across Desktop, Tablet, and Mobile viewports with independent sidebar scrolling.
- **Input Validation**: All user inputs strictly validate positivity, numerical sanity, and maximum caps.
