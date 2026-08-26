"""
Streamlit App: Optimizer Playground (Part A)
Visualizes 7 optimization algorithms from scratch on loss surfaces with different condition numbers.
"""

import time
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import streamlit as st

from optimizers import get_optimizer
from ui_helpers import apply_instrument_theme, inject_custom_css, render_telemetry_strip

# -----------------------------------------------------------------------------
# Fixed Color Palette & Loss Surfaces (Unified across Part A & B)
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
# Explanations Dictionary
# -----------------------------------------------------------------------------
OPTIMIZER_EXPLANATIONS = {
    "SGD": (
        "**Stochastic Gradient Descent (SGD)** updates parameters directly in the direction of steepest descent. "
        "On ill-conditioned loss surfaces with high condition numbers, steep wall gradients dwarf shallow valley floor gradients, "
        "causing SGD to violently oscillate back and forth across the valley rather than progressing smoothly towards the minimum."
    ),
    "Momentum": (
        "**Momentum** accumulates a velocity vector of past gradients to accelerate progress down consistent slopes. "
        "While it dampens orthogonal oscillations, accumulated momentum can cause the optimizer to overshoot the minimum "
        "before turning back."
    ),
    "NAG": (
        "**Nesterov Accelerated Gradient (NAG)** computes gradients at a 'look-ahead' position $(\\theta - \\beta v)$ "
        "where momentum is about to carry the parameters. If momentum is carrying the optimizer up an opposing slope, the look-ahead "
        "gradient senses the rising terrain in advance and applies responsive braking, significantly reducing overshoot vs plain Momentum."
    ),
    "AdaGrad": (
        "**AdaGrad** tracks the sum of squared historical gradients ($G$) independently for each parameter. "
        "By scaling step sizes inversely by $\\sqrt{G}$, AdaGrad provides different effective learning rates per parameter: "
        "frequently updated parameters with large gradients receive smaller step sizes, while infrequently updated parameters receive larger step sizes."
    ),
    "RMSProp": (
        "**RMSProp** fixes AdaGrad's continuously shrinking learning rate. In AdaGrad, historical squared gradients sum up indefinitely, "
        "eventually causing learning rates to decay to near zero and prematurely stop training. RMSProp replaces the unbounded sum with an exponentially "
        "decaying moving average of squared gradients, keeping step sizes adaptable throughout training."
    ),
    "Adam": (
        "**Adam (Adaptive Moment Estimation)** combines the momentum strategy (tracking 1st moment $m$) with RMSProp (tracking 2nd moment $v$) "
        "along with bias-correction factors. It automatically adapts step sizes per parameter while leveraging gradient momentum."
    ),
    "AdamW": (
        "**AdamW** decouples weight decay from adaptive gradient updates. In standard Adam with folded L2 regularization, "
        "L2 decay gradients ($g + \\lambda \\theta$) get factored into the 2nd moment $v$, distorting the effective decay penalty for high-gradient parameters. "
        "AdamW subtracts weight decay directly from parameters after the Adam step, maintaining true, scale-invariant weight decay."
    ),
}


def analyze_optimizer_status(name, traj, loss_history, curr_step):
    """Generate a 1-2 sentence plain-language explanation of optimizer status."""
    if len(loss_history) == 0:
        return "No history recorded."

    curr_loss = loss_history[curr_step] if curr_step < len(loss_history) else loss_history[-1]
    init_loss = loss_history[0]
    curr_pos = traj[curr_step] if curr_step < len(traj) else traj[-1]

    if (
        np.isnan(curr_loss)
        or np.isinf(curr_loss)
        or curr_loss > 10.0 * init_loss
        or abs(curr_pos[0]) > 1e4
        or abs(curr_pos[1]) > 1e4
    ):
        return "❌ **Diverged:** The steps were too big and it overshot, moving further from the minimum."

    if curr_loss < 1e-3 or (
        curr_step >= 20
        and np.std(loss_history[max(0, curr_step - 20) : curr_step + 1]) < 1e-4 * max(1.0, curr_loss)
        and curr_loss < 0.1
    ):
        return "✅ **Settled into minimum:** Smoothly reached the bottom of the bowl."

    recent_traj = traj[max(0, curr_step - 30) : curr_step + 1]
    if len(recent_traj) > 5:
        y_diffs = np.diff(recent_traj[:, 1])
        sign_flips = np.sum(np.diff(np.sign(y_diffs)) != 0)
        if sign_flips >= 4 and curr_loss > 0.05:
            return "⚠️ **Still zig-zagging:** Bouncing back and forth across steep valley walls because the step size is too large for this direction."

    if curr_step >= 50 and (init_loss - curr_loss) / max(1.0, init_loss) < 0.3:
        return "🐢 **Slow progress:** Moving in the right direction, but taking very small steps."

    return "🏃 **Moving downhill:** Making steady progress towards the minimum."


@st.cache_data(show_spinner=False)
def compute_contour_grid(surface_name, max_x, max_y):
    func = LOSS_SURFACES[surface_name]["func"]
    x_vals = np.linspace(-max_x, max_x, 200)
    y_vals = np.linspace(-max_y, max_y, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(X, Y)
    return X, Y, Z


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

        trajectories[name] = np.array(traj)
        losses[name] = np.array(loss_vals)

    return trajectories, losses


def render_part_a():
    """Main rendering function for Part A (2D Loss Surface Playground)."""
    inject_custom_css()
    st.header("⚡ Part A: 2D Loss Surface Playground")

    # Read global app mode
    app_mode = st.session_state.get("app_mode", "Simple mode")
    is_simple = app_mode == "Simple mode"

    with st.expander("How to use Part A", expanded=False):
        st.write(
            "Visualize and compare seven optimization algorithms (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW) "
            "on 2D quadratic loss surfaces with varying condition numbers ($L(x,y) = x^2 + c \\cdot y^2$). "
            "Use sidebar controls to set parameters, and play the animation to watch trajectory paths and loss decay!"
        )

    if not is_simple:
        with st.expander("💡 Explain this optimizer (Technical)", expanded=False):
            exp_option = st.selectbox(
                "Select an optimizer to view its technical explanation:",
                options=list(OPTIMIZER_EXPLANATIONS.keys()),
                index=0,
                key="part_a_explain_opt",
            )
            st.markdown(OPTIMIZER_EXPLANATIONS[exp_option])

    # -------------------------------------------------------------------------
    # Session State Initialization
    # -------------------------------------------------------------------------
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "lr_slider_val" not in st.session_state:
        st.session_state.lr_slider_val = 0.01
    if "prev_surface" not in st.session_state:
        st.session_state.prev_surface = "L2: c = 50 (default)"

    # -------------------------------------------------------------------------
    # Sidebar Controls & Input Validation
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Part A Settings")

    surface_name = st.sidebar.selectbox(
        "Loss surface",
        list(LOSS_SURFACES.keys()),
        index=1,
        help="A narrower bowl (higher condition number) means one direction needs much smaller steps than the other.",
        key="part_a_surface_name",
    )
    selected_surface = LOSS_SURFACES[surface_name]
    c_val = selected_surface["c"]

    if st.session_state.prev_surface != surface_name:
        st.session_state.current_step = 0
        st.session_state.prev_surface = surface_name

    st.sidebar.caption(
        f"**Hessian Condition Number:** $\\kappa = {c_val:.0f}$. "
        "A narrower bowl means one direction requires much smaller step sizes than the other, causing plain SGD to violently zig-zag."
    )

    selected_opts = st.sidebar.multiselect(
        "Optimizers to overlay",
        options=list(OPTIMIZER_COLORS.keys()),
        default=list(OPTIMIZER_COLORS.keys()),
        help="Choose which optimization algorithms to compare on the loss surface.",
        key="part_a_selected_opts",
    )

    st.sidebar.markdown("### 🎯 LR Sensitivity Presets")
    col_eta1, col_eta2, col_eta3 = st.sidebar.columns(3)

    if col_eta1.button("η = 0.001", help="Set learning rate to 0.001", key="btn_eta1"):
        st.session_state.lr_slider_val = 0.001
        st.session_state.current_step = 0

    if col_eta2.button("η = 0.01", help="Set learning rate to 0.01", key="btn_eta2"):
        st.session_state.lr_slider_val = 0.01
        st.session_state.current_step = 0

    if col_eta3.button("η = 0.1", help="Set learning rate to 0.1", key="btn_eta3"):
        st.session_state.lr_slider_val = 0.1
        st.session_state.current_step = 0

    lr = st.sidebar.select_slider(
        "Learning rate (eta)",
        options=LOG_LR_OPTIONS,
        key="lr_slider_val",
        help="How big a step to take each time. Too big = overshoots and bounces around. Too small = takes forever to get anywhere.",
    )

    # Input Validation: Learning Rate
    if lr <= 0:
        st.error("Learning rate eta must be strictly greater than 0.")
        st.stop()

    if not is_simple:
        st.sidebar.markdown("### ⚙️ Hyperparameters")
        beta = st.sidebar.slider(
            "beta (Momentum / RMSProp)",
            min_value=0.0,
            max_value=0.999,
            value=0.9,
            step=0.01,
            help="How much 'memory' of past direction to keep.",
            key="part_a_beta",
        )
        beta1 = st.sidebar.slider(
            "beta1 (Adam / AdamW)",
            min_value=0.0,
            max_value=0.999,
            value=0.9,
            step=0.01,
            help="beta1 = memory of recent direction.",
            key="part_a_beta1",
        )
        beta2 = st.sidebar.slider(
            "beta2 (Adam / AdamW)",
            min_value=0.0,
            max_value=0.999,
            value=0.999,
            step=0.001,
            help="beta2 = memory of recent step sizes.",
            key="part_a_beta2",
        )
        weight_decay = st.sidebar.slider(
            "lambda (AdamW weight decay)",
            min_value=0.0,
            max_value=0.1,
            value=0.001,
            step=0.001,
            help="How much to shrink the weights a little on every step.",
            key="part_a_wd",
        )

        # Input Validation: Hyperparameters
        if weight_decay < 0:
            st.error("Weight decay lambda cannot be negative.")
            st.stop()
    else:
        beta = 0.9
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.001

    col_x0, col_y0 = st.sidebar.columns(2)
    x0 = col_x0.number_input(
        "Starting x (x0)",
        value=8.0,
        step=0.5,
        help="Where the optimization ball starts rolling from (x-axis).",
        key="part_a_x0",
    )
    y0 = col_y0.number_input(
        "Starting y (y0)",
        value=8.0,
        step=0.5,
        help="Where the optimization ball starts rolling from (y-axis).",
        key="part_a_y0",
    )

    # Input Validation: Coordinates
    if np.isnan(x0) or np.isnan(y0) or np.isinf(x0) or np.isinf(y0):
        st.error("Starting coordinates x0 and y0 must be valid finite numbers.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("Animation Controls")

    btn_col1, btn_col2, btn_col3, btn_col4 = st.sidebar.columns(4)

    if btn_col1.button("▶ Play", key="part_a_play", type="primary"):
        st.session_state.is_playing = True

    if btn_col2.button("⏸ Pause", key="part_a_pause"):
        st.session_state.is_playing = False

    if btn_col3.button("⏭ Step", key="part_a_step"):
        st.session_state.is_playing = False
        st.session_state.current_step = min(st.session_state.current_step + 1, 500)

    if btn_col4.button("↺ Reset", key="part_a_reset"):
        st.session_state.is_playing = False
        st.session_state.current_step = 0

    current_step = st.sidebar.slider(
        "Iteration step",
        min_value=0,
        max_value=500,
        value=st.session_state.current_step,
        key="step_slider_val",
    )
    if current_step != st.session_state.current_step:
        st.session_state.current_step = current_step

    speed_ms = st.sidebar.slider(
        "Animation speed (ms / step)", min_value=10, max_value=500, value=50, step=10, key="part_a_speed"
    )

    # -------------------------------------------------------------------------
    # Trajectory Computation & Rendering
    # -------------------------------------------------------------------------
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

    show_grad_arrow = st.checkbox(
        "Show gradient direction (downhill arrow)",
        value=True,
        help="Draws an arrow at each ball pointing in the direction of steepest descent.",
        key="part_a_show_arrow",
    )

    # Telemetry Strip
    telemetry_items = {"ITER": f"{curr_step}/500"}
    if selected_opts:
        first_opt = selected_opts[0]
        first_traj = trajectories[first_opt]
        first_loss = losses[first_opt]
        pos_at_step = first_traj[curr_step] if curr_step < len(first_traj) else first_traj[-1]
        loss_at_step = first_loss[curr_step] if curr_step < len(first_loss) else first_loss[-1]

        if not np.isnan(pos_at_step).any() and not np.isinf(pos_at_step).any():
            grad_at_step = selected_surface["grad"](pos_at_step)
            grad_norm_val = float(np.linalg.norm(grad_at_step))
            grad_norm_str = f"{grad_norm_val:.4f}"
        else:
            grad_norm_str = "N/A"

        loss_str = f"{loss_at_step:.4f}" if (not np.isnan(loss_at_step) and not np.isinf(loss_at_step)) else "Diverged"
        telemetry_items[f"{first_opt} LOSS"] = loss_str
        telemetry_items[f"{first_opt} GRAD NORM"] = grad_norm_str

    render_telemetry_strip(telemetry_items)

    col1, col2 = st.columns(2)

    # View 1: Contour Plot
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))

        max_x = max(abs(x0) * 1.2, 10.0)
        max_y = max(abs(y0) * 1.2, 10.0)

        X, Y, Z = compute_contour_grid(surface_name, max_x, max_y)

        contour_fill = ax1.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.85)
        ax1.contour(X, Y, Z, levels=10, colors="white", alpha=0.25, linewidths=0.5)
        cbar = fig1.colorbar(contour_fill, ax=ax1, label="Loss L(x, y)")
        cbar.ax.yaxis.set_tick_params(color="#E7EAEE")
        cbar.ax.tick_params(labelsize=8, labelcolor="#E7EAEE")
        cbar.set_label("Loss L(x, y)", color="#E7EAEE")
        cbar.outline.set_edgecolor("#2E3742")

        ax1.plot(0, 0, "*", color="gold", markersize=14, markeredgecolor="black", label="Min (0,0)", zorder=10)

        for name in selected_opts:
            color = OPTIMIZER_COLORS[name]
            traj = trajectories[name][: curr_step + 1]
            valid_mask = ~np.isnan(traj).any(axis=1)
            valid_traj = traj[valid_mask]

            if len(valid_traj) > 0:
                ax1.plot(valid_traj[:, 0], valid_traj[:, 1], color=color, linewidth=1.2, alpha=0.3)

                trail_len = 30
                recent_traj = valid_traj[-trail_len:]
                n_pts = len(recent_traj)
                if n_pts > 1:
                    segments = np.stack([recent_traj[:-1], recent_traj[1:]], axis=1)
                    r, g, b, _ = mcolors.to_rgba(color)
                    colors = [
                        (r, g, b, 0.15 + 0.8 * (i / max(1, n_pts - 1)))
                        for i in range(n_pts - 1)
                    ]
                    lc = LineCollection(segments, colors=colors, linewidths=2.8)
                    ax1.add_collection(lc)

                curr_pos = valid_traj[-1]
                ax1.plot(
                    curr_pos[0],
                    curr_pos[1],
                    "o",
                    color=color,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    zorder=12,
                    label=name,
                )

                if show_grad_arrow:
                    g = selected_surface["grad"](curr_pos)
                    neg_g = -g
                    g_norm = np.linalg.norm(neg_g)
                    if g_norm > 1e-8:
                        arrow_len = max_x * 0.12
                        dir_vec = (neg_g / g_norm) * arrow_len
                        ax1.quiver(
                            curr_pos[0],
                            curr_pos[1],
                            dir_vec[0],
                            dir_vec[1],
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            color=color,
                            width=0.007,
                            headwidth=4,
                            headlength=5,
                            zorder=13,
                        )

        ax1.set_xlim(-max_x, max_x)
        ax1.set_ylim(-max_y, max_y)
        ax1.set_xlabel("Parameter x")
        ax1.set_ylabel("Parameter y")
        ax1.set_title(f"Contour Trajectory Plot ({surface_name}) - Step {curr_step}/500")
        ax1.legend(loc="upper right", fontsize=7, framealpha=0.8)
        apply_instrument_theme(fig1, ax1)
        fig1.tight_layout()
        st.pyplot(fig1)

        if show_grad_arrow:
            st.caption("📍 *The arrow shows which way is downhill from here.*")

    # View 2: Loss Plot
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
                    markersize=8,
                    markeredgecolor="black",
                    zorder=9,
                )

        ax2.set_xlim(0, max(10, curr_step))
        ax2.set_xlabel("Iteration Step")
        ax2.set_ylabel("Loss L(x, y)")
        ax2.set_yscale("log")
        ax2.set_title(f"Loss Reduction vs Iteration - Step {curr_step}/500")
        ax2.legend(loc="upper right", fontsize=7, framealpha=0.8)
        apply_instrument_theme(fig2, ax2)
        fig2.tight_layout()
        st.pyplot(fig2)

    # Status Analysis Section
    st.markdown("### 🔍 Why did it stop here? (Optimizer Status Analysis)")
    status_cols = st.columns(min(len(selected_opts), 3) if selected_opts else 1)

    for i, opt_name in enumerate(selected_opts):
        col_target = status_cols[i % len(status_cols)]
        traj = trajectories[opt_name]
        loss_hist = losses[opt_name]
        status_msg = analyze_optimizer_status(opt_name, traj, loss_hist, curr_step)

        with col_target:
            st.markdown(f"**{opt_name}** ({OPTIMIZER_COLORS[opt_name].capitalize()})")
            st.markdown(status_msg)

    # Animation Loop Execution
    if st.session_state.is_playing:
        if st.session_state.current_step < 500:
            steps_per_frame = max(1, int(round(100.0 / speed_ms)))
            delay = (speed_ms * steps_per_frame) / 1000.0
            time.sleep(delay)
            st.session_state.current_step = min(500, st.session_state.current_step + steps_per_frame)
            st.rerun()
        else:
            st.session_state.is_playing = False


if __name__ == "__main__":
    st.set_page_config(page_title="Part A Playground", layout="wide")
    render_part_a()
