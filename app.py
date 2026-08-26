"""
Optimizer Visualizer & Benchmark Suite (Main Entry Point)
Unified Streamlit application running Part A (2D Loss Surfaces) and Part B (MLP Neural Network Benchmark).
"""

import streamlit as st
from part_a_playground import render_part_a
from part_b_dashboard import render_part_b
from ui_helpers import inject_custom_css

# -----------------------------------------------------------------------------
# Global Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Optimizer Visualizer & Benchmark Suite",
    page_icon="⚡",
    layout="wide",
)

inject_custom_css()

# Initialize global session state for App Mode
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Simple mode"

# -----------------------------------------------------------------------------
# Global Sidebar Control (App Mode Toggle shared across both tabs)
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Global Controls")
app_mode = st.sidebar.radio(
    "App Mode",
    options=["Simple mode", "Advanced mode"],
    index=0 if st.session_state.app_mode == "Simple mode" else 1,
    help="Simple mode presents beginner-friendly captions and hides technical sliders. Advanced mode unlocks all controls.",
    key="global_app_mode_toggle",
)
st.session_state.app_mode = app_mode

# -----------------------------------------------------------------------------
# Top-Level README Expander
# -----------------------------------------------------------------------------
st.title("⚡ Optimizer Visualizer & Benchmark Suite")

with st.expander("📖 How to Use This Visualizer Suite", expanded=False):
    st.markdown(
        """
        Welcome to the **Optimizer Visualizer & Benchmark Suite**! This application allows you to explore and compare 
        **7 optimization algorithms** (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, and AdamW) implemented entirely from scratch using NumPy.

        ### 📑 Tab Navigation
        - **⚡ Part A: Optimizer Playground**:
          Visualize optimization trajectories on 2D quadratic loss surfaces ($L(x,y) = x^2 + c \\cdot y^2$). Watch how each optimizer ball rolls downhill, 
          inspect fading momentum trails, toggle steepest descent gradient arrows, and analyze divergence vs oscillation vs convergence.
        - **🧠 Part B: Neural Network Dashboard**:
          Train a 3-layer Multi-Layer Perceptron (Input $\\to$ 16 $\\to$ 8 $\\to$ 1) on the **Breast Cancer Wisconsin Dataset** from scratch. Compare training/test loss reduction, 
          accuracy progression, effective learning rates per epoch, and automatic convergence epoch calculations.

        ### 🕹️ App Modes
        - **Simple mode** (Default): Designed for beginners. Hides complex hyperparameter sliders and technical plots, replacing them with plain-English tooltips and captions.
        - **Advanced mode**: Designed for researchers/students. Unlocks hyperparameter sliders ($\\beta, \\beta_1, \\beta_2, \\lambda$), technical explanation expanders, and effective learning rate charts.

        ### 🚀 How to Run the Application
        In your terminal, execute:
        ```bash
        streamlit run app.py
        ```
        """
    )

# -----------------------------------------------------------------------------
# Main Tabbed Interface
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "⚡ Part A: Optimizer Playground",
    "🧠 Part B: Neural Network Dashboard",
])

with tab1:
    render_part_a()

with tab2:
    render_part_b()
