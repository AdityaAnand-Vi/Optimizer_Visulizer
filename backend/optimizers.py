"""
Stateful Optimization Algorithms implemented from first principles using pure NumPy.
Implements SGD, SGDMomentum, NAG, AdaGrad, RMSProp, Adam, and AdamW.
"""

import numpy as np


class BaseOptimizer:
    """Base class for all stateful optimizers."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def reset(self):
        """Reset internal optimizer state."""
        pass

    def step(self, params, grad):
        """Perform a single parameter update step."""
        raise NotImplementedError


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent (SGD)."""

    def __init__(self, lr=0.01):
        super().__init__(lr=lr)

    def reset(self):
        pass

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
        return params - self.lr * grad


class SGDMomentum(BaseOptimizer):
    """SGD with Momentum."""

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr=lr)
        self.beta = beta
        self.v = None

    def reset(self):
        self.v = None

    def step(self, params, grad):
        params = np.asarray(params, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1.0 - self.beta) * grad
        return params - self.lr * self.v


# Alias for backward compatibility
Momentum = SGDMomentum


class NAG(BaseOptimizer):
    """Nesterov Accelerated Gradient (NAG)."""

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr=lr)
        self.beta = beta
        self.v = None

    def reset(self):
        self.v = None

    def get_lookahead_params(self, params):
        """Compute lookahead parameters theta_lookahead = theta - lr * beta * v."""
        params = np.asarray(params, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)
        return params - self.lr * self.beta * self.v

    def step(self, params, grad):
        """
        Perform step. Accepts either grad array (computed at lookahead position)
        or a grad_fn callback.
        """
        params = np.asarray(params, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)

        if callable(grad):
            lookahead = self.get_lookahead_params(params)
            g = grad(lookahead)
        else:
            g = np.asarray(grad, dtype=np.float64)

        self.v = self.beta * self.v + (1.0 - self.beta) * g
        return params - self.lr * self.v


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
        grad = np.asarray(grad, dtype=np.float64)
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G += grad**2
        return params - self.lr * grad / (np.sqrt(self.G) + self.eps)


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
        grad = np.asarray(grad, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1.0 - self.beta) * (grad**2)
        return params - self.lr * grad / (np.sqrt(self.v) + self.eps)


class Adam(BaseOptimizer):
    """Adam optimizer with full bias correction."""

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
        grad = np.asarray(grad, dtype=np.float64)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


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
        grad = np.asarray(grad, dtype=np.float64)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params)


def get_optimizer(name, lr=0.01, **kwargs):
    """Factory function returning optimizer instances."""
    optimizers = {
        "SGD": SGD,
        "SGDMomentum": SGDMomentum,
        "Momentum": SGDMomentum,
        "NAG": NAG,
        "AdaGrad": AdaGrad,
        "RMSProp": RMSProp,
        "Adam": Adam,
        "AdamW": AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name](lr=lr, **kwargs)
