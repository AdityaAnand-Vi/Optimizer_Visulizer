"""
Pure NumPy Multi-Layer Perceptron (MLP) for Binary Classification.
Architecture: Input -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid
"""

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.mean(
        y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
    )
    return float(loss)


class MLP:
    def __init__(self, input_dim, seed=42):
        if seed is not None:
            np.random.seed(seed)

        self.input_dim = input_dim
        self.params = {
            "W1": np.random.randn(input_dim, 16) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(16, dtype=np.float64),
            "W2": np.random.randn(16, 8) * np.sqrt(2.0 / 16),
            "b2": np.zeros(8, dtype=np.float64),
            "W3": np.random.randn(8, 1) * np.sqrt(2.0 / (8 + 1)),
            "b3": np.zeros(1, dtype=np.float64),
        }

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = np.maximum(0.0, Z1)

        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        A2 = np.maximum(0.0, Z2)

        Z3 = A2 @ self.params["W3"] + self.params["b3"]
        A3 = sigmoid(Z3)

        cache = (X, Z1, A1, Z2, A2, Z3, A3)
        return A3, cache

    def backward(self, X, y_true, cache):
        X, Z1, A1, Z2, A2, Z3, A3 = cache
        N = X.shape[0]
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)

        dZ3 = (A3 - y_true) / N
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0)

        dA2 = dZ3 @ self.params["W3"].T
        dZ2 = dA2 * (Z2 > 0.0).astype(np.float64)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.params["W2"].T
        dZ1 = dA1 * (Z1 > 0.0).astype(np.float64)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def get_flat_params(self):
        return np.concatenate([
            self.params["W1"].ravel(),
            self.params["b1"].ravel(),
            self.params["W2"].ravel(),
            self.params["b2"].ravel(),
            self.params["W3"].ravel(),
            self.params["b3"].ravel(),
        ])

    def set_flat_params(self, flat_params):
        flat_params = np.asarray(flat_params, dtype=np.float64)
        idx = 0

        w1_size = self.input_dim * 16
        self.params["W1"] = flat_params[idx : idx + w1_size].reshape(self.input_dim, 16)
        idx += w1_size

        b1_size = 16
        self.params["b1"] = flat_params[idx : idx + b1_size]
        idx += b1_size

        w2_size = 16 * 8
        self.params["W2"] = flat_params[idx : idx + w2_size].reshape(16, 8)
        idx += w2_size

        b2_size = 8
        self.params["b2"] = flat_params[idx : idx + b2_size]
        idx += b2_size

        w3_size = 8 * 1
        self.params["W3"] = flat_params[idx : idx + w3_size].reshape(8, 1)
        idx += w3_size

        b3_size = 1
        self.params["b3"] = flat_params[idx : idx + b3_size]
        idx += b3_size

    def flatten_grads(self, grads):
        return np.concatenate([
            grads["W1"].ravel(),
            grads["b1"].ravel(),
            grads["W2"].ravel(),
            grads["b2"].ravel(),
            grads["W3"].ravel(),
            grads["b3"].ravel(),
        ])

    def predict_class(self, X, threshold=0.5):
        probs, _ = self.forward(X)
        return (probs >= threshold).astype(int)
