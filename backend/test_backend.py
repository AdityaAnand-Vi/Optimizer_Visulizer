"""
Unit tests for backend optimizers, MLP model, and FastAPI REST endpoints.
Asserts mathematical update accuracy and API response payloads for /api/simulate-2d and /api/train-nn.
"""

import unittest
from fastapi.testclient import TestClient
import numpy as np

from main import app
from model import MLP, binary_cross_entropy, sigmoid
from optimizers import SGD, SGDMomentum, NAG, AdaGrad, RMSProp, Adam, AdamW, get_optimizer


class TestOptimizers(unittest.TestCase):

    def test_sgd_update(self):
        opt = SGD(lr=0.1)
        params = np.array([2.0, -3.0])
        grads = np.array([0.5, 1.0])
        updated = opt.step(params, grads)
        expected = np.array([1.95, -3.1])
        np.testing.assert_allclose(updated, expected, atol=1e-7)

    def test_sgd_momentum_update(self):
        opt = SGDMomentum(lr=0.1, beta=0.9)
        params = np.array([2.0, -3.0])
        grads = np.array([0.5, 1.0])
        updated = opt.step(params, grads)
        expected = np.array([1.995, -3.01])
        np.testing.assert_allclose(updated, expected, atol=1e-7)

    def test_nag_lookahead(self):
        opt = NAG(lr=0.1, beta=0.9)
        params = np.array([2.0, -3.0])
        lookahead = opt.get_lookahead_params(params)
        np.testing.assert_allclose(lookahead, params, atol=1e-7)

        grads = np.array([0.5, 1.0])
        updated = opt.step(params, grads)
        next_lookahead = opt.get_lookahead_params(updated)
        expected_lookahead = updated - 0.9 * opt.v
        np.testing.assert_allclose(next_lookahead, expected_lookahead, atol=1e-7)

    def test_adagrad_update(self):
        opt = AdaGrad(lr=0.1, eps=1e-8)
        params = np.array([2.0, -3.0])
        grads = np.array([3.0, 4.0])
        updated = opt.step(params, grads)
        expected = np.array([1.9, -3.1])
        np.testing.assert_allclose(updated, expected, atol=1e-7)

    def test_rmsprop_update(self):
        opt = RMSProp(lr=0.1, beta=0.9, eps=1e-8)
        params = np.array([2.0, -3.0])
        grads = np.array([2.0, 4.0])
        updated = opt.step(params, grads)
        expected = params - 0.1 * np.array([2.0, 4.0]) / (np.sqrt(np.array([0.4, 1.6])) + 1e-8)
        np.testing.assert_allclose(updated, expected, atol=1e-7)

    def test_adam_bias_correction(self):
        opt = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        params = np.array([2.0, -3.0])
        grads = np.array([1.0, 2.0])
        updated = opt.step(params, grads)
        expected = np.array([1.9, -3.1])
        np.testing.assert_allclose(updated, expected, atol=1e-6)

    def test_adamw_decoupled_weight_decay(self):
        opt = AdamW(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
        params = np.array([2.0, -3.0])
        grads = np.array([1.0, 2.0])
        updated = opt.step(params, grads)
        expected = np.array([1.898, -3.097])
        np.testing.assert_allclose(updated, expected, atol=1e-6)


class TestMLP(unittest.TestCase):

    def test_mlp_forward_backward_shapes(self):
        mlp = MLP(input_dim=30, seed=42)
        X = np.random.randn(10, 30)
        y = np.random.randint(0, 2, size=(10, 1))

        probs, cache = mlp.forward(X)
        self.assertEqual(probs.shape, (10, 1))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        loss = binary_cross_entropy(y, probs)
        self.assertGreater(loss, 0.0)

        grads = mlp.backward(X, y, cache)
        self.assertEqual(grads["W1"].shape, (30, 16))
        self.assertEqual(grads["b1"].shape, (16,))
        self.assertEqual(grads["W2"].shape, (16, 8))
        self.assertEqual(grads["b2"].shape, (8,))
        self.assertEqual(grads["W3"].shape, (8, 1))
        self.assertEqual(grads["b3"].shape, (1,))


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_simulate_2d_endpoint(self):
        payload = {
            "surface_name": "L2: c = 50 (default)",
            "optimizers": ["SGD", "SGDMomentum", "Adam"],
            "x0": 8.0,
            "y0": 8.0,
            "lr": 0.01,
            "max_steps": 20,
        }
        res = self.client.post("/api/simulate-2d", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("trajectories", data)
        self.assertIn("losses", data)
        self.assertIn("SGD", data["trajectories"])
        self.assertEqual(len(data["trajectories"]["SGD"]), 21)

    def test_train_nn_endpoint(self):
        payload = {
            "optimizers": ["Adam"],
            "lr": 0.01,
            "epochs": 5,
            "batch_size": 64,
        }
        res = self.client.post("/api/train-nn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("results", data)
        self.assertIn("summary_table", data)
        self.assertEqual(len(data["results"]["Adam"]["train_loss"]), 5)
        self.assertEqual(len(data["summary_table"]), 1)


if __name__ == "__main__":
    unittest.main()
