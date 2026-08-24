"""
Streamlit App: Optimizer Playground (Part A)
Visualizes 7 optimization algorithms from scratch on loss surfaces with different condition numbers.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from optimizers import get_optimizer

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Optimizer Playground",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Fixed Color Dictionary & Loss Surfaces
# -----------------------------------------------------------------------------
OPTIMIZER_COLORS = {
    "SGD": "red",
    "Momentum": "orange",
    "NAG": "gold",
    "AdaGrad": "green",
    "RMSProp": "blue",
    "Adam": "purple",
    "AdamW": "brown",
}

LOSS_SURFACES = {
    "L1: c = 10": {
        "c": 10.0,
        "func": lambda x, y: x**2 + 10.0 * y**2,
        "grad": lambda theta: np.array([2.0 * theta[0], 20.0 * theta[1]], dtype=np.float64),
    },
    "L2: c = 50 (default)": {
        "c": 50.0,
        "func": lambda x, y: x**2 + 50.0 * y**2,
        "grad": lambda theta: np.array([2.0 * theta[0], 100.0 * theta[1]], dtype=np.float64),
    },
    "L3: c = 100": {
        "c": 100.0,
        "func": lambda x, y: x**2 + 100.0 * y**2,
        "grad": lambda theta: np.array([2.0 * theta[0], 200.0 * theta[1]], dtype=np.float64),
    },
    "L4: c = 1000": {
        "c": 1000.0,
        "func": lambda x, y: x**2 + 1000.0 * y**2,
        "grad": lambda theta: np.array([2.0 * theta[0], 2000.0 * theta[1]], dtype=np.float64),
    },
}

LOG_LR_OPTIONS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# -----------------------------------------------------------------------------
# Header & Expander
# -----------------------------------------------------------------------------
st.title("⚡ Optimizer Playground")

with st.expander("How to use this tool", expanded=False):
    st.write(
        "Welcome to the Optimizer Playground! This interactive tool lets you visualize and compare seven different "
        "optimization algorithms (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, and AdamW) on 2D quadratic loss surfaces "
        "with varying condition numbers ($L(x,y) = x^2 + c \\cdot y^2$). Use the sidebar controls to choose the surface, "
        "starting position, learning rate $\\eta$, momentum/decay parameters, and which optimizers to overlay. "
        "Use the Play, Pause, Step, and Reset controls to watch how each optimizer traverses the surface toward the "
        "global minimum at $(0,0)$ (marked with a gold star) alongside a real-time loss reduction line chart."
    )

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# -----------------------------------------------------------------------------
# Sidebar UI Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Optimization Settings")

# 1. Loss surface selectbox
surface_name = st.sidebar.selectbox("Loss surface", list(LOSS_SURFACES.keys()), index=1)
selected_surface = LOSS_SURFACES[surface_name]

# 2. Multiselect optimizers
selected_opts = st.sidebar.multiselect(
    "Optimizers to overlay",
    options=list(OPTIMIZER_COLORS.keys()),
    default=list(OPTIMIZER_COLORS.keys()),
)

# 3. Learning rate slider (log-spaced option slider)
lr = st.sidebar.select_slider("Learning rate (eta)", options=LOG_LR_OPTIONS, value=0.01)

# Validate learning rate
if lr <= 0:
    st.error("Learning rate eta must be greater than 0.")
    st.stop()

# 4. Hyperparameters
beta = st.sidebar.slider("beta (Momentum / RMSProp)", min_value=0.0, max_value=0.999, value=0.9, step=0.01)
beta1 = st.sidebar.slider("beta1 (Adam / AdamW)", min_value=0.0, max_value=0.999, value=0.9, step=0.01)
beta2 = st.sidebar.slider("beta2 (Adam / AdamW)", min_value=0.0, max_value=0.999, value=0.999, step=0.001)
weight_decay = st.sidebar.slider("lambda (AdamW weight decay)", min_value=0.0, max_value=0.1, value=0.001, step=0.001)

# 5. Starting point
col_x0, col_y0 = st.sidebar.columns(2)
x0 = col_x0.number_input("Starting x (x0)", value=8.0, step=0.5)
y0 = col_y0.number_input("Starting y (y0)", value=8.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Animation Controls")

# 6. Play, Pause, Step, Reset buttons
btn_col1, btn_col2, btn_col3, btn_col4 = st.sidebar.columns(4)

if btn_col1.button("▶ Play"):
    st.session_state.is_playing = True

if btn_col2.button("⏸ Pause"):
    st.session_state.is_playing = False

if btn_col3.button("⏭ Step"):
    st.session_state.is_playing = False
    st.session_state.current_step = min(st.session_state.current_step + 1, 500)

if btn_col4.button("↺ Reset"):
    st.session_state.is_playing = False
    st.session_state.current_step = 0

# Step slider / timeline scrubber
current_step = st.sidebar.slider(
    "Iteration step",
    min_value=0,
    max_value=500,
    value=st.session_state.current_step,
    key="step_slider_val",
)
# Keep current_step in sync with slider if user manually scrubs slider
if current_step != st.session_state.current_step:
    st.session_state.current_step = current_step

speed_ms = st.sidebar.slider("Animation speed (ms / step)", min_value=10, max_value=500, value=50, step=10)


# -----------------------------------------------------------------------------
# Trajectory Computation Function
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_all_trajectories(
    opts_tuple, s_name, init_x, init_y, lrate, b_val, b1_val, b2_val, wd_val, max_steps=500
):
    surf = LOSS_SURFACES[s_name]
    grad_fn = surf["grad"]
    func = surf["func"]

    trajectories = {}
    losses = {}

    for name in opts_tuple:
        kwargs = {"lr": lrate}
        if name in ["Momentum", "NAG", "RMSProp"]:
            kwargs["beta"] = b_val
        elif name in ["Adam", "AdamW"]:
            kwargs["beta1"] = b1_val
            kwargs["beta2"] = b2_val
            if name == "AdamW":
                kwargs["weight_decay"] = wd_val

        opt = get_optimizer(name, **kwargs)
        opt.reset()

        params = np.array([init_x, init_y], dtype=np.float64)
        traj = [params.copy()]
        loss_vals = [func(params[0], params[1])]

        for _ in range(max_steps):
            # Check for divergence or NaN
            if np.isnan(params).any() or np.isinf(params).any() or abs(params[0]) > 1e8 or abs(params[1]) > 1e8:
                traj.append(params.copy())
                loss_vals.append(loss_vals[-1] if not np.isnan(loss_vals[-1]) else np.nan)
                continue

            try:
                if name == "NAG":
                    params = opt.step(params, grad_fn)
                else:
                    g = grad_fn(params)
                    params = opt.step(params, g)

                l_val = func(params[0], params[1])
                traj.append(params.copy())
                loss_vals.append(l_val)
            except Exception:
                traj.append(params.copy())
                loss_vals.append(np.nan)

        trajectories[name] = np.array(traj)  # shape (501, 2)
        losses[name] = np.array(loss_vals)  # shape (501,)

    return trajectories, losses


# Compute trajectories
trajectories, losses = compute_all_trajectories(
    tuple(selected_opts),
    surface_name,
    x0,
    y0,
    lr,
    beta,
    beta1,
    beta2,
    weight_decay,
    max_steps=500,
)

curr_step = st.session_state.current_step

# -----------------------------------------------------------------------------
# Main Visualization Area (Two Synced Columns)
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

# --- View 1: Contour Plot ---
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    # Grid bounds
    max_x = max(abs(x0) * 1.2, 10.0)
    max_y = max(abs(y0) * 1.2, 10.0)

    # Compute contour mesh
    x_vals = np.linspace(-max_x, max_x, 200)
    y_vals = np.linspace(-max_y, max_y, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = selected_surface["func"](X, Y)

    # Filled contour plot
    contour_fill = ax1.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.85)
    ax1.contour(X, Y, Z, levels=15, colors="white", alpha=0.25, linewidths=0.5)
    fig1.colorbar(contour_fill, ax=ax1, label="Loss L(x, y)")

    # Global minimum marker
    ax1.plot(0, 0, "*", color="gold", markersize=14, markeredgecolor="black", label="Min (0,0)", zorder=10)

    # Trajectories up to current step
    for name in selected_opts:
        color = OPTIMIZER_COLORS[name]
        traj = trajectories[name][: curr_step + 1]

        # Filter out NaN for plotting
        valid_mask = ~np.isnan(traj).any(axis=1)
        valid_traj = traj[valid_mask]

        if len(valid_traj) > 0:
            ax1.plot(valid_traj[:, 0], valid_traj[:, 1], color=color, linewidth=2, label=name, alpha=0.9)
            ax1.plot(
                valid_traj[-1, 0],
                valid_traj[-1, 1],
                "o",
                color=color,
                markersize=6,
                markeredgecolor="black",
                zorder=9,
            )

    ax1.set_xlim(-max_x, max_x)
    ax1.set_ylim(-max_y, max_y)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Contour Plot of {surface_name} (Step {curr_step}/500)")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig1.tight_layout()
    st.pyplot(fig1)

# --- View 2: Loss vs Iteration Plot ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    iters = np.arange(curr_step + 1)

    for name in selected_opts:
        color = OPTIMIZER_COLORS[name]
        loss_history = losses[name][: curr_step + 1]

        ax2.plot(iters, loss_history, color=color, linewidth=2, label=name, alpha=0.9)
        if len(loss_history) > 0 and not np.isnan(loss_history[-1]):
            ax2.plot(
                curr_step,
                loss_history[-1],
                "o",
                color=color,
                markersize=6,
                markeredgecolor="black",
                zorder=9,
            )

    ax2.set_xlim(0, max(10, curr_step))
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss L(x, y)")
    ax2.set_yscale("log")
    ax2.set_title(f"Loss Reduction vs Iteration (Step {curr_step}/500)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig2.tight_layout()
    st.pyplot(fig2)

# -----------------------------------------------------------------------------
# Animation Loop Execution
# -----------------------------------------------------------------------------
if st.session_state.is_playing:
    if st.session_state.current_step < 500:
        time.sleep(speed_ms / 1000.0)
        st.session_state.current_step += 1
        st.rerun()
    else:
        st.session_state.is_playing = False
