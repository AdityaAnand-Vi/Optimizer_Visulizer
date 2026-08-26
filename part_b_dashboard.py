"""
Streamlit App: Neural Network Optimizer Benchmark (Part B)
Benchmarks 7 optimization algorithms on a from-scratch MLP trained on Breast Cancer Wisconsin dataset.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp import MLP, binary_cross_entropy
from optimizers import get_optimizer
from ui_helpers import apply_instrument_theme, inject_custom_css, render_telemetry_strip

# -----------------------------------------------------------------------------
# Fixed Color Palette & Constants (Identical to Part A)
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

ADAPTIVE_OPTIMIZERS = ["AdaGrad", "RMSProp", "Adam", "AdamW"]

LOG_LR_OPTIONS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


@st.cache_data
def get_breast_cancer_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.shape[1]


def compute_effective_lr(opt, base_lr, param_idx=0):
    """Compute effective learning rate for adaptive optimizers."""
    name = opt.__class__.__name__
    if name == "AdaGrad":
        if opt.G is not None:
            return float(base_lr / (np.sqrt(opt.G[param_idx]) + opt.eps))
    elif name == "RMSProp":
        if opt.v is not None:
            return float(base_lr / (np.sqrt(opt.v[param_idx]) + opt.eps))
    elif name in ["Adam", "AdamW"]:
        if opt.v is not None and opt.t > 0:
            v_hat = opt.v[param_idx] / (1.0 - opt.beta2**opt.t)
            return float(base_lr / (np.sqrt(v_hat) + opt.eps))

    return float(base_lr)


def calculate_convergence_epoch(test_losses, tolerance=0.01):
    """
    Convergence epoch = first epoch where val loss comes within 1% of its final value
    and stays within 1% for all remaining epochs.
    """
    if len(test_losses) == 0:
        return "-"
    final_loss = test_losses[-1]
    if np.isnan(final_loss) or np.isinf(final_loss):
        return "Diverged"

    target_threshold = tolerance * abs(final_loss)

    for ep_idx in range(len(test_losses)):
        remaining = test_losses[ep_idx:]
        if all(abs(val - final_loss) <= target_threshold for val in remaining):
            return ep_idx + 1

    return len(test_losses)


def render_part_b():
    """Main rendering function for Part B (Neural Network Optimizer Benchmark)."""
    inject_custom_css()
    st.header("🧠 Part B: Neural Network Optimizer Benchmark")

    app_mode = st.session_state.get("app_mode", "Simple mode")
    is_simple = app_mode == "Simple mode"

    X_train, X_test, y_train, y_test, num_features = get_breast_cancer_data()

    with st.expander("How to use Part B", expanded=False):
        st.write(
            "Benchmark 7 optimization algorithms (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW) "
            "on a 3-layer Multi-Layer Perceptron (Input $\\to$ 16 $\\to$ 8 $\\to$ 1) trained from scratch "
            "on the **Breast Cancer Wisconsin Dataset**."
        )

    # UI Statistics Metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Features", f"{num_features}")
    m_col2.metric("Train Samples", f"{X_train.shape[0]}")
    m_col3.metric("Test Samples", f"{X_test.shape[0]}")
    m_col4.metric("Train/Test Split", "80% / 20%")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Sidebar Controls & Input Validation
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Part B Settings")

    selected_opts = st.sidebar.multiselect(
        "Optimizers to benchmark",
        options=list(OPTIMIZER_COLORS.keys()),
        default=list(OPTIMIZER_COLORS.keys()),
        help="Select which optimizers to train on the MLP simultaneously.",
        key="part_b_selected_opts",
    )

    lr = st.sidebar.select_slider(
        "Learning rate (eta)",
        options=LOG_LR_OPTIONS,
        value=0.01,
        help="Base learning rate parameter passed to each optimizer.",
        key="part_b_lr",
    )

    # Input Validation: Learning Rate
    if lr <= 0:
        st.error("Learning rate eta must be strictly greater than 0.")
        st.stop()

    epochs = st.sidebar.number_input(
        "Number of Epochs",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Total training epochs.",
        key="part_b_epochs",
    )

    # Input Validation: Epochs
    if epochs <= 0:
        st.error("Number of epochs must be a positive integer > 0.")
        st.stop()

    batch_size_choice = st.sidebar.selectbox(
        "Batch Size",
        options=[32, 64, 128, "Full Batch"],
        index=0,
        help="Size of mini-batches for gradient updates per epoch.",
        key="part_b_batch_choice",
    )

    if batch_size_choice == "Full Batch":
        batch_size = X_train.shape[0]
    else:
        batch_size = int(batch_size_choice)

    # Input Validation: Batch Size
    if batch_size <= 0:
        st.error("Batch size must be a positive integer > 0.")
        st.stop()

    if not is_simple:
        st.sidebar.markdown("### ⚙️ Hyperparameters")
        beta = st.sidebar.slider("beta (Momentum / RMSProp)", 0.0, 0.999, 0.9, step=0.01, key="part_b_beta")
        beta1 = st.sidebar.slider("beta1 (Adam / AdamW)", 0.0, 0.999, 0.9, step=0.01, key="part_b_beta1")
        beta2 = st.sidebar.slider("beta2 (Adam / AdamW)", 0.0, 0.999, 0.999, step=0.001, key="part_b_beta2")
        weight_decay = st.sidebar.slider(
            "lambda (AdamW weight decay)", 0.0, 0.1, 0.001, step=0.001, key="part_b_wd"
        )

        # Input Validation: Weight Decay
        if weight_decay < 0:
            st.error("Weight decay lambda cannot be negative.")
            st.stop()
    else:
        beta = 0.9
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.001

    train_button = st.sidebar.button("🚀 Train Optimizers", type="primary", use_container_width=True, key="part_b_train_btn")

    # Session State Training Containers
    if "has_trained" not in st.session_state:
        st.session_state.has_trained = False
    if "training_results" not in st.session_state:
        st.session_state.training_results = {}

    if train_button:
        if not selected_opts:
            st.warning("Please select at least one optimizer from the sidebar to train.")
            st.stop()

        st.session_state.has_trained = True
        st.session_state.training_results = {}

        for name in selected_opts:
            st.session_state.training_results[name] = {
                "train_loss": [],
                "test_loss": [],
                "train_acc": [],
                "test_acc": [],
                "eff_lr": [],
            }

        telemetry_ph = st.empty()
        progress_bar = st.progress(0, text="Initializing training...")
        status_box = st.empty()

        # Layout columns depending on mode
        if is_simple:
            plot_row1_col1, plot_row1_col2 = st.columns(2)
            plot_row2_col1 = st.container()

            chart1_ph = plot_row1_col1.empty()
            chart2_ph = plot_row1_col2.empty()
            chart3_ph = plot_row2_col1.empty()
            chart4_ph = None
        else:
            plot_row1_col1, plot_row1_col2 = st.columns(2)
            plot_row2_col1, plot_row2_col2 = st.columns(2)

            chart1_ph = plot_row1_col1.empty()
            chart2_ph = plot_row1_col2.empty()
            chart3_ph = plot_row2_col1.empty()
            chart4_ph = plot_row2_col2.empty()

        total_runs = len(selected_opts)

        for run_idx, opt_name in enumerate(selected_opts):
            kwargs = {"lr": lr}
            if opt_name in ["Momentum", "NAG", "RMSProp"]:
                kwargs["beta"] = beta
            elif opt_name in ["Adam", "AdamW"]:
                kwargs["beta1"] = beta1
                kwargs["beta2"] = beta2
                if opt_name == "AdamW":
                    kwargs["weight_decay"] = weight_decay

            mlp = MLP(input_dim=num_features, seed=42)
            opt = get_optimizer(opt_name, **kwargs)
            opt.reset()

            n_samples = X_train.shape[0]

            for epoch in range(1, epochs + 1):
                indices = np.arange(n_samples)
                np.random.shuffle(indices)

                for start_i in range(0, n_samples, batch_size):
                    end_i = min(start_i + batch_size, n_samples)
                    batch_idx = indices[start_i:end_i]
                    X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

                    flat_params = mlp.get_flat_params()
                    _, cache = mlp.forward(X_batch)
                    grads = mlp.backward(X_batch, y_batch, cache)
                    flat_grads = mlp.flatten_grads(grads)

                    updated_params = opt.step(flat_params, flat_grads)
                    mlp.set_flat_params(updated_params)

                # Record per-epoch metrics
                y_tr_pred, _ = mlp.forward(X_train)
                tr_loss = binary_cross_entropy(y_train, y_tr_pred)
                tr_acc = float(np.mean(mlp.predict_class(X_train).ravel() == y_train)) * 100.0

                y_te_pred, _ = mlp.forward(X_test)
                te_loss = binary_cross_entropy(y_test, y_te_pred)
                te_acc = float(np.mean(mlp.predict_class(X_test).ravel() == y_test)) * 100.0

                effective_lr_val = compute_effective_lr(opt, lr, param_idx=0)

                st.session_state.training_results[opt_name]["train_loss"].append(tr_loss)
                st.session_state.training_results[opt_name]["test_loss"].append(te_loss)
                st.session_state.training_results[opt_name]["train_acc"].append(tr_acc)
                st.session_state.training_results[opt_name]["test_acc"].append(te_acc)
                st.session_state.training_results[opt_name]["eff_lr"].append(effective_lr_val)

                first_opt = selected_opts[0] if selected_opts else None
                if first_opt and len(st.session_state.training_results[first_opt]["train_loss"]) > 0:
                    curr_tr_loss = st.session_state.training_results[first_opt]["train_loss"][-1]
                    curr_te_acc = st.session_state.training_results[first_opt]["test_acc"][-1]
                    telemetry_items = {
                        "EPOCH": f"{epoch}/{epochs}",
                        f"{first_opt} TRAIN LOSS": f"{curr_tr_loss:.4f}",
                        f"{first_opt} TEST ACC": f"{curr_te_acc:.2f}%",
                    }
                else:
                    telemetry_items = {"EPOCH": f"{epoch}/{epochs}"}

                with telemetry_ph.container():
                    render_telemetry_strip(telemetry_items)

                overall_progress = ((run_idx * epochs) + epoch) / (total_runs * epochs)
                progress_bar.progress(
                    overall_progress,
                    text=f"Training {opt_name}... (Epoch {epoch}/{epochs}) - Test Acc: {te_acc:.2f}%",
                )

                if epoch % 5 == 0 or epoch == epochs or epoch == 1:
                    # Plot 1: Train Loss
                    fig1, ax1 = plt.subplots(figsize=(5, 3.8))
                    for done_name in selected_opts:
                        hist = st.session_state.training_results[done_name]["train_loss"]
                        if len(hist) > 0:
                            ax1.plot(
                                np.arange(1, len(hist) + 1),
                                hist,
                                color=OPTIMIZER_COLORS[done_name],
                                label=done_name,
                                linewidth=1.8,
                            )
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Train Loss (BCE)")
                    ax1.set_yscale("log")
                    ax1.set_title("Training Loss vs Epoch")
                    ax1.legend(fontsize=7)
                    apply_instrument_theme(fig1, ax1)
                    fig1.tight_layout()
                    chart1_ph.pyplot(fig1)

                    # Plot 2: Test Loss
                    fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                    for done_name in selected_opts:
                        hist = st.session_state.training_results[done_name]["test_loss"]
                        if len(hist) > 0:
                            ax2.plot(
                                np.arange(1, len(hist) + 1),
                                hist,
                                color=OPTIMIZER_COLORS[done_name],
                                label=done_name,
                                linewidth=1.8,
                            )
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Test Loss (BCE)")
                    ax2.set_yscale("log")
                    ax2.set_title("Validation / Test Loss vs Epoch")
                    ax2.legend(fontsize=7)
                    apply_instrument_theme(fig2, ax2)
                    fig2.tight_layout()
                    chart2_ph.pyplot(fig2)

                    # Plot 3: Test Accuracy
                    fig3, ax3 = plt.subplots(figsize=(5, 3.8))
                    for done_name in selected_opts:
                        hist = st.session_state.training_results[done_name]["test_acc"]
                        if len(hist) > 0:
                            ax3.plot(
                                np.arange(1, len(hist) + 1),
                                hist,
                                color=OPTIMIZER_COLORS[done_name],
                                label=done_name,
                                linewidth=1.8,
                            )
                    ax3.set_xlabel("Epoch")
                    ax3.set_ylabel("Test Accuracy (%)")
                    ax3.set_title("Test Accuracy vs Epoch")
                    ax3.legend(fontsize=7)
                    apply_instrument_theme(fig3, ax3)
                    fig3.tight_layout()
                    chart3_ph.pyplot(fig3)

                    # Plot 4: Effective Learning Rate (Advanced mode only)
                    if not is_simple and chart4_ph is not None:
                        fig4, ax4 = plt.subplots(figsize=(5, 3.8))
                        has_adaptive = False
                        for done_name in selected_opts:
                            if done_name in ADAPTIVE_OPTIMIZERS:
                                hist = st.session_state.training_results[done_name]["eff_lr"]
                                if len(hist) > 0:
                                    ax4.plot(
                                        np.arange(1, len(hist) + 1),
                                        hist,
                                        color=OPTIMIZER_COLORS[done_name],
                                        label=done_name,
                                        linewidth=1.8,
                                    )
                                    has_adaptive = True
                        ax4.set_xlabel("Epoch")
                        ax4.set_ylabel("Effective Learning Rate (eta_eff)")
                        ax4.set_yscale("log")
                        ax4.set_title("Effective Learning Rate vs Epoch (Adaptive Only)")
                        if has_adaptive:
                            ax4.legend(fontsize=7)
                        apply_instrument_theme(fig4, ax4)
                        fig4.tight_layout()
                        chart4_ph.pyplot(fig4)

        progress_bar.progress(1.0, text="✅ Benchmark Training Complete!")
        status_box.success("All selected optimizers completed training!")

    # Display plain-English captions in Simple mode
    if is_simple and st.session_state.has_trained:
        st.caption(
            "💡 **Accuracy Progress:** Higher and further right means the network is getting better "
            "at telling the difference between malignant and benign tumors as it practices over more epochs."
        )

    # -------------------------------------------------------------------------
    # Render Final Results Summary Table
    # -------------------------------------------------------------------------
    if st.session_state.has_trained and st.session_state.training_results:
        st.markdown("---")
        st.markdown("### 📊 Benchmark Summary Results")

        summary_rows = []
        for opt_name, res in st.session_state.training_results.items():
            if len(res["train_loss"]) > 0:
                final_tr_loss = res["train_loss"][-1]
                final_te_loss = res["test_loss"][-1]
                final_tr_acc = res["train_acc"][-1]
                final_te_acc = res["test_acc"][-1]
                conv_ep = calculate_convergence_epoch(res["test_loss"], tolerance=0.01)

                summary_rows.append({
                    "Optimizer": opt_name,
                    "Final Train Loss": f"{final_tr_loss:.4f}",
                    "Final Test Loss": f"{final_te_loss:.4f}",
                    "Train Acc (%)": f"{final_tr_acc:.2f}%",
                    "Test Acc (%)": f"{final_te_acc:.2f}%",
                    "Convergence Epoch": f"{conv_ep}",
                })

        if summary_rows:
            st.dataframe(summary_rows, use_container_width=True)

    else:
        st.info("👈 Click **🚀 Train Optimizers** in the sidebar to run the benchmark.")


if __name__ == "__main__":
    st.set_page_config(page_title="Part B Dashboard", layout="wide")
    render_part_b()
