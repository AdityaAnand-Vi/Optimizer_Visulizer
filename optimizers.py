"""
Custom optimization algorithms implemented from scratch using NumPy.
"""

import numpy as np


class BaseOptimizer:
    """Base class for optimizers."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def reset(self):
        """Reset internal state."""
        pass

    def step(self, params, grad):
        """Perform a single optimization step."""
        raise NotImplementedError


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent (SGD)."""

    def __init__(self, lr=0.01):
        super().__init__(lr=lr)

    def reset(self):
        pass

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        params -= self.lr * grad
        return params


class Momentum(BaseOptimizer):
    """SGD with Momentum."""

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr=lr)
        self.beta = beta
        self.v = None

    def reset(self):
        self.v = None

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1.0 - self.beta) * grad
        params -= self.lr * self.v
        return params


SGDMomentum = Momentum


class NAG(BaseOptimizer):
    """Nesterov Accelerated Gradient (NAG)."""

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr=lr)
        self.beta = beta
        self.v = None

    def reset(self):
        self.v = None

    def step(self, params, grad_fn):
        params = np.asarray(params, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)
        lookahead = params - self.lr * self.beta * self.v
        grad = grad_fn(lookahead)
        self.v = self.beta * self.v + (1.0 - self.beta) * grad
        params -= self.lr * self.v
        return params


class AdaGrad(BaseOptimizer):
    """AdaGrad optimizer."""

    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr=lr)
        self.eps = eps
        self.G = None

    def reset(self):
        self.G = None

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G += grad**2
        params -= self.lr * grad / (np.sqrt(self.G) + self.eps)
        return params


class RMSProp(BaseOptimizer):
    """RMSProp optimizer."""

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr=lr)
        self.beta = beta
        self.eps = eps
        self.v = None

    def reset(self):
        self.v = None

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1.0 - self.beta) * (grad**2)
        params -= self.lr * grad / (np.sqrt(self.v) + self.eps)
        return params


class Adam(BaseOptimizer):
    """Adam optimizer."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


class AdamW(BaseOptimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-3):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        params -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params)
        return params


def get_optimizer(name, lr=0.01, **kwargs):
    """
    Factory function to get an optimizer instance by name.
    Supported names: 'SGD', 'Momentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW'.
    """
    optimizers = {
        "SGD": SGD,
        "Momentum": Momentum,
        "NAG": NAG,
        "AdaGrad": AdaGrad,
        "RMSProp": RMSProp,
        "Adam": Adam,
        "AdamW": AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    return optimizers[name](lr=lr, **kwargs)


if __name__ == "__main__":

    def f(theta):
        x, y = theta
        return x**2 + 50.0 * y**2

    def grad_f(theta):
        x, y = theta
        return np.array([2.0 * x, 100.0 * y], dtype=np.float64)

    optimizer_names = ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"]

    print("Sanity checking optimizers on toy quadratic f(x, y) = x^2 + 50*y^2 starting at (8, 8):\n")

    for name in optimizer_names:
        opt = get_optimizer(name, lr=0.1)
        params = np.array([8.0, 8.0], dtype=np.float64)
        print(f"--- {name} ---")
        print(f"Initial: x={params[0]:.4f}, y={params[1]:.4f}, f(x,y)={f(params):.4f}")

        for step_i in range(1, 11):
            if name == "NAG":
                params = opt.step(params, grad_f)
            else:
                grad = grad_f(params)
                params = opt.step(params, grad)
            print(f"Step {step_i:2d}: x={params[0]:.4f}, y={params[1]:.4f}, f(x,y)={f(params):.4f}")
        print()
