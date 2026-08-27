"""
FastAPI Backend for Interactive Optimizer Visualizer.
Exposes REST API endpoints:
- POST /api/simulate-2d: 2D Loss Surface simulation & trajectory calculation.
- POST /api/train-nn: Neural Network (MLP) benchmark training & metrics.
"""

from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import MLP, binary_cross_entropy
from optimizers import get_optimizer

app = FastAPI(title="Optimizer Visualizer API", version="2.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Surface Definitions
SURFACE_CONFIGS = {
    "L1": {"c": 10.0, "name": "L1: c = 10"},
    "L1: c = 10": {"c": 10.0, "name": "L1: c = 10"},
    "L2": {"c": 50.0, "name": "L2: c = 50 (default)"},
    "L2: c = 50 (default)": {"c": 50.0, "name": "L2: c = 50 (default)"},
    "L3": {"c": 100.0, "name": "L3: c = 100"},
    "L3: c = 100": {"c": 100.0, "name": "L3: c = 100"},
    "L4": {"c": 1000.0, "name": "L4: c = 1000"},
    "L4: c = 1000": {"c": 1000.0, "name": "L4: c = 1000"},
}


def get_surface_functions(c_val):
    func = lambda x, y: x**2 + c_val * (y**2)
    grad = lambda theta: np.array([2.0 * theta[0], 2.0 * c_val * theta[1]], dtype=np.float64)
    return func, grad


class Simulate2DRequest(BaseModel):
    surface_name: str = Field(default="L2: c = 50 (default)", description="Loss surface key")
    optimizers: List[str] = Field(default=["SGD", "SGDMomentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"])
    x0: float = Field(default=8.0, description="Starting X coordinate")
    y0: float = Field(default=8.0, description="Starting Y coordinate")
    lr: float = Field(default=0.01, description="Learning rate eta")
    beta: float = Field(default=0.9, description="Momentum/RMSProp beta factor")
    beta1: float = Field(default=0.9, description="Adam beta1 factor")
    beta2: float = Field(default=0.999, description="Adam beta2 factor")
    weight_decay: float = Field(default=0.001, description="AdamW weight decay lambda")
    max_steps: int = Field(default=500, description="Maximum iteration steps")


class TrainNNRequest(BaseModel):
    optimizers: List[str] = Field(default=["SGD", "SGDMomentum", "Adam", "AdamW"])
    lr: float = Field(default=0.01, description="Learning rate eta")
    epochs: int = Field(default=100, description="Total training epochs")
    batch_size: int = Field(default=32, description="Mini-batch size")
    beta: float = Field(default=0.9, description="Momentum/RMSProp beta factor")
    beta1: float = Field(default=0.9, description="Adam beta1 factor")
    beta2: float = Field(default=0.999, description="Adam beta2 factor")
    weight_decay: float = Field(default=0.001, description="AdamW weight decay lambda")


def compute_effective_lr(opt, base_lr, param_idx=0):
    """Calculate effective learning rate for adaptive optimizers."""
    name = opt.__class__.__name__
    if name == "AdaGrad" and opt.G is not None:
        return float(base_lr / (np.sqrt(opt.G[param_idx]) + opt.eps))
    elif name == "RMSProp" and opt.v is not None:
        return float(base_lr / (np.sqrt(opt.v[param_idx]) + opt.eps))
    elif name in ["Adam", "AdamW"] and opt.v is not None and opt.t > 0:
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


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Optimizer Visualizer API v2.0.0"}


@app.post("/api/simulate-2d")
def simulate_2d(req: Simulate2DRequest):
    if req.surface_name not in SURFACE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid surface_name: {req.surface_name}")

    c_val = SURFACE_CONFIGS[req.surface_name]["c"]
    func, grad_fn = get_surface_functions(c_val)

    max_x = max(abs(req.x0) * 1.2, 10.0)
    max_y = max(abs(req.y0) * 1.2, 10.0)

    x_vals = np.linspace(-max_x, max_x, 100).tolist()
    y_vals = np.linspace(-max_y, max_y, 100).tolist()
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(X, Y).tolist()

    trajectories = {}
    losses = {}
    gradients = {}
    deltas = {}
    statuses = {}
    final_statuses = {}
    peaks = {}

    for opt_name in req.optimizers:
        kwargs = {"lr": req.lr}
        if opt_name in ["SGDMomentum", "Momentum", "NAG", "RMSProp"]:
            kwargs["beta"] = req.beta
        elif opt_name in ["Adam", "AdamW"]:
            kwargs["beta1"] = req.beta1
            kwargs["beta2"] = req.beta2
            if opt_name == "AdamW":
                kwargs["weight_decay"] = req.weight_decay

        try:
            opt = get_optimizer(opt_name, **kwargs)
            opt.reset()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        params = np.array([req.x0, req.y0], dtype=np.float64)
        init_loss = float(func(params[0], params[1]))
        init_grad = grad_fn(params)

        traj = [params.copy().tolist()]
        loss_vals = [init_loss]
        grad_vals = [init_grad.copy().tolist()]
        delta_vals = [[0.0, 0.0]]
        step_statuses = ["RUNNING"]
        opt_status = "RUNNING"
        peak_theta = max(abs(req.x0), abs(req.y0))

        for step_i in range(req.max_steps):
            try:
                if opt_name == "NAG":
                    lookahead = opt.get_lookahead_params(params)
                    g = grad_fn(lookahead)
                    next_params = opt.step(params, g)
                else:
                    g = grad_fn(params)
                    next_params = opt.step(params, g)

                delta = next_params - params
                params = next_params
                l_val = float(func(params[0], params[1]))

                is_div = (
                    np.isnan(params).any() or
                    np.isinf(params).any() or
                    abs(params[0]) > 1e6 or
                    abs(params[1]) > 1e6 or
                    np.isnan(l_val) or
                    np.isinf(l_val) or
                    l_val > 1e6
                )

                safe_x = float(params[0]) if (np.isfinite(params[0]) and abs(params[0]) <= 1e12) else (1e12 if params[0] > 0 else -1e12)
                safe_y = float(params[1]) if (np.isfinite(params[1]) and abs(params[1]) <= 1e12) else (1e12 if params[1] > 0 else -1e12)
                safe_loss = l_val if (np.isfinite(l_val) and abs(l_val) <= 1e12) else 1e12
                safe_gx = float(g[0]) if np.isfinite(g[0]) else 1e12
                safe_gy = float(g[1]) if np.isfinite(g[1]) else 1e12
                safe_dx = float(delta[0]) if np.isfinite(delta[0]) else 0.0
                safe_dy = float(delta[1]) if np.isfinite(delta[1]) else 0.0

                traj.append([safe_x, safe_y])
                loss_vals.append(safe_loss)
                grad_vals.append([safe_gx, safe_gy])
                delta_vals.append([safe_dx, safe_dy])

                if is_div:
                    step_statuses.append("DIVERGED")
                    opt_status = "DIVERGED"
                    break
                elif safe_loss < 1e-6 or (safe_gx**2 + safe_gy**2) < 1e-10:
                    step_statuses.append("CONVERGED")
                    opt_status = "CONVERGED"
                else:
                    step_statuses.append("RUNNING")

                peak_theta = max(peak_theta, abs(safe_x), abs(safe_y))
            except Exception:
                step_statuses.append("DIVERGED")
                opt_status = "DIVERGED"
                break

        trajectories[opt_name] = traj
        losses[opt_name] = loss_vals
        gradients[opt_name] = grad_vals
        deltas[opt_name] = delta_vals
        statuses[opt_name] = step_statuses
        final_statuses[opt_name] = opt_status
        peaks[opt_name] = peak_theta

    return {
        "surface_name": req.surface_name,
        "condition_number": c_val,
        "max_x": max_x,
        "max_y": max_y,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": Z,
        "trajectories": trajectories,
        "losses": losses,
        "gradients": gradients,
        "deltas": deltas,
        "statuses": statuses,
        "final_statuses": final_statuses,
        "peaks": peaks,
    }


# Alias endpoint for Part A Playground
@app.post("/api/playground/trajectory")
def trajectory_alias(req: Simulate2DRequest):
    return simulate_2d(req)


@app.post("/api/train-nn")
def train_nn(req: TrainNNRequest):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    num_features = X.shape[1]

    results = {}
    summary_table = []

    for opt_name in req.optimizers:
        kwargs = {"lr": req.lr}
        if opt_name in ["SGDMomentum", "Momentum", "NAG", "RMSProp"]:
            kwargs["beta"] = req.beta
        elif opt_name in ["Adam", "AdamW"]:
            kwargs["beta1"] = req.beta1
            kwargs["beta2"] = req.beta2
            if opt_name == "AdamW":
                kwargs["weight_decay"] = req.weight_decay

        mlp = MLP(input_dim=num_features, seed=42)
        opt = get_optimizer(opt_name, **kwargs)
        opt.reset()

        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        eff_lrs = []

        n_samples = X_train_scaled.shape[0]
        bs = req.batch_size if req.batch_size > 0 else n_samples

        for epoch in range(1, req.epochs + 1):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start_i in range(0, n_samples, bs):
                end_i = min(start_i + bs, n_samples)
                batch_idx = indices[start_i:end_i]
                X_batch, y_batch = X_train_scaled[batch_idx], y_train[batch_idx]

                flat_params = mlp.get_flat_params()
                _, cache = mlp.forward(X_batch)
                grads = mlp.backward(X_batch, y_batch, cache)
                flat_grads = mlp.flatten_grads(grads)

                updated_params = opt.step(flat_params, flat_grads)
                mlp.set_flat_params(updated_params)

            y_tr_pred, _ = mlp.forward(X_train_scaled)
            tr_loss = binary_cross_entropy(y_train, y_tr_pred)
            tr_acc = float(np.mean(mlp.predict_class(X_train_scaled).ravel() == y_train)) * 100.0

            y_te_pred, _ = mlp.forward(X_test_scaled)
            te_loss = binary_cross_entropy(y_test, y_te_pred)
            te_acc = float(np.mean(mlp.predict_class(X_test_scaled).ravel() == y_test)) * 100.0

            eff_lr_val = compute_effective_lr(opt, req.lr, param_idx=0)

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            eff_lrs.append(eff_lr_val)

        conv_epoch = calculate_convergence_epoch(test_losses, tolerance=0.01)

        results[opt_name] = {
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_acc": train_accs,
            "test_acc": test_accs,
            "eff_lr": eff_lrs,
        }

        summary_table.append({
            "optimizer": opt_name,
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1],
            "final_train_acc": train_accs[-1],
            "final_test_acc": test_accs[-1],
            "convergence_epoch": conv_epoch,
        })

    return {
        "dataset": "Breast Cancer Wisconsin",
        "num_features": num_features,
        "train_samples": X_train_scaled.shape[0],
        "test_samples": X_test_scaled.shape[0],
        "results": results,
        "summary_table": summary_table,
    }


# Alias endpoint for Part B Dashboard
@app.post("/api/benchmark/train")
def train_alias(req: TrainNNRequest):
    return train_nn(req)

# Serve frontend static files
frontend_dist = os.path.join(os.path.dirname(__file__), "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    # For any other route, serve index.html (SPA routing fallback)
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str, request: Request):
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=404, content={"message": "Not Found"})
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
