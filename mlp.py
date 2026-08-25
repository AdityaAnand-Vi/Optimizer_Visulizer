"""
Multi-Layer Perceptron (MLP) for Binary Classification implemented from scratch in NumPy.
Architecture: Input -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid
"""

import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    """
    Numerically stable Binary Cross-Entropy (BCE) loss.
    y_true: shape (N, 1) or (N,)
    y_pred: shape (N, 1) or (N,)
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)

    # Clip probabilities to avoid log(0)
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
    )
    return float(loss)


class MLP:
    """
    3-layer Multi-Layer Perceptron for binary classification.
    Layer sizes: Input(input_dim) -> Dense(16) -> Dense(8) -> Dense(1)
    """

    def __init__(self, input_dim, seed=42):
        if seed is not None:
            np.random.seed(seed)

        self.input_dim = input_dim
        self.layer_sizes = [input_dim, 16, 8, 1]

        # Initialize weights with He (layers 1, 2) & Xavier/Glorot (layer 3)
        self.params = {
            "W1": np.random.randn(input_dim, 16) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(16, dtype=np.float64),
            "W2": np.random.randn(16, 8) * np.sqrt(2.0 / 16),
            "b2": np.zeros(8, dtype=np.float64),
            "W3": np.random.randn(8, 1) * np.sqrt(2.0 / (8 + 1)),
            "b3": np.zeros(1, dtype=np.float64),
        }

    def forward(self, X):
        """
        Forward pass.
        X: shape (N, input_dim)
        Returns (predictions A3, cache)
        """
        X = np.asarray(X, dtype=np.float64)

        # Layer 1: Dense(16) -> ReLU
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = np.maximum(0.0, Z1)

        # Layer 2: Dense(8) -> ReLU
        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        A2 = np.maximum(0.0, Z2)

        # Layer 3: Dense(1) -> Sigmoid
        Z3 = A2 @ self.params["W3"] + self.params["b3"]
        A3 = sigmoid(Z3)

        cache = (X, Z1, A1, Z2, A2, Z3, A3)
        return A3, cache

    def backward(self, X, y_true, cache):
        """
        Manual backpropagation via chain rule.
        Returns gradients dictionary matching parameter shapes.
        """
        X, Z1, A1, Z2, A2, Z3, A3 = cache
        N = X.shape[0]
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)

        # Layer 3 gradient (BCE + Sigmoid shortcut: dZ3 = (A3 - y) / N)
        dZ3 = (A3 - y_true) / N
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0)

        # Layer 2 gradient
        dA2 = dZ3 @ self.params["W3"].T
        dZ2 = dA2 * (Z2 > 0.0).astype(np.float64)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        # Layer 1 gradient
        dA1 = dZ2 @ self.params["W2"].T
        dZ1 = dA1 * (Z1 > 0.0).astype(np.float64)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }
        return grads

    def get_flat_params(self):
        """Flatten all parameters into a 1D NumPy array."""
        return np.concatenate([
            self.params["W1"].ravel(),
            self.params["b1"].ravel(),
            self.params["W2"].ravel(),
            self.params["b2"].ravel(),
            self.params["W3"].ravel(),
            self.params["b3"].ravel(),
        ])

    def set_flat_params(self, flat_params):
        """Set parameters from a flat 1D NumPy array."""
        flat_params = np.asarray(flat_params, dtype=np.float64)
        idx = 0

        # W1
        w1_size = self.input_dim * 16
        self.params["W1"] = flat_params[idx : idx + w1_size].reshape(self.input_dim, 16)
        idx += w1_size

        # b1
        b1_size = 16
        self.params["b1"] = flat_params[idx : idx + b1_size]
        idx += b1_size

        # W2
        w2_size = 16 * 8
        self.params["W2"] = flat_params[idx : idx + w2_size].reshape(16, 8)
        idx += w2_size

        # b2
        b2_size = 8
        self.params["b2"] = flat_params[idx : idx + b2_size]
        idx += b2_size

        # W3
        w3_size = 8 * 1
        self.params["W3"] = flat_params[idx : idx + w3_size].reshape(8, 1)
        idx += w3_size

        # b3
        b3_size = 1
        self.params["b3"] = flat_params[idx : idx + b3_size]
        idx += b3_size

    def flatten_grads(self, grads):
        """Flatten gradients dict into a 1D NumPy array matching get_flat_params order."""
        return np.concatenate([
            grads["W1"].ravel(),
            grads["b1"].ravel(),
            grads["W2"].ravel(),
            grads["b2"].ravel(),
            grads["W3"].ravel(),
            grads["b3"].ravel(),
        ])

    def predict(self, X):
        """Return output probabilities."""
        probs, _ = self.forward(X)
        return probs

    def predict_class(self, X, threshold=0.5):
        """Return binary class predictions (0 or 1)."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from optimizers import get_optimizer

    print("Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    print(f"Dataset loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples, {input_dim} features.\n")

    # Build MLP & Optimizer
    mlp = MLP(input_dim=input_dim, seed=42)
    opt = get_optimizer("Adam", lr=0.01)
    opt.reset()

    epochs = 150
    print(f"Training MLP with Adam(lr=0.01) for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        # Flatten current params
        flat_params = mlp.get_flat_params()

        # Forward pass
        y_pred, cache = mlp.forward(X_train)
        loss = binary_cross_entropy(y_train, y_pred)

        # Backward pass
        grads = mlp.backward(X_train, y_train, cache)
        flat_grads = mlp.flatten_grads(grads)

        # Optimizer step
        updated_flat_params = opt.step(flat_params, flat_grads)
        mlp.set_flat_params(updated_flat_params)

        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            # Test metrics
            y_test_pred, _ = mlp.forward(X_test)
            test_loss = binary_cross_entropy(y_test, y_test_pred)

            train_acc = np.mean(mlp.predict_class(X_train).ravel() == y_train)
            test_acc = np.mean(mlp.predict_class(X_test).ravel() == y_test)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%"
            )

    print("\nTraining complete! Final Evaluation:")
    final_test_pred, _ = mlp.forward(X_test)
    final_test_loss = binary_cross_entropy(y_test, final_test_pred)
    final_test_acc = np.mean(mlp.predict_class(X_test).ravel() == y_test)
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
