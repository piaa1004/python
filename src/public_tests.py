"""직접 구현한 DNN의 핵심 요구사항을 검증하는 실행 가능한 검사 모듈."""

import unittest

import numpy as np

from opt_utils import (
    DenseLayer, ReLU, Tanh, SoftmaxCrossEntropy, build_model,
    evaluate, get_batches, load_data,
)
from optimization import gradient_check, train
from testCases import (
    activation_test_case, batch_test_case, dense_test_case,
    gradient_test_case, loss_test_cases,
)
from test_utils import assert_allclose, assert_shape, numerical_gradient, relative_error


class DNNTests(unittest.TestCase):
    def test_01_dense_forward_backward_and_initialization(self):
        case = dense_test_case()
        layer = DenseLayer(2, 3, np.random.default_rng(0))
        layer.W = case["W"].copy()
        layer.b = case["b"].copy()
        assert_allclose(layer.forward(case["X"]), case["out"], name="Dense out")
        assert_allclose(layer.backward(case["dout"]), case["dX"], name="Dense dX")
        assert_allclose(layer.dW, case["dW"], name="Dense dW")
        assert_allclose(layer.db, case["db"], name="Dense db")
        with self.assertRaises(ValueError):
            layer.forward(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            layer.backward(np.zeros((2, 2)))
        for initialization, numerator in [("he", 2.0), ("xavier", 1.0)]:
            initialized = DenseLayer(4, 5, np.random.default_rng(3), initialization)
            expected = np.random.default_rng(3).normal(size=(4, 5)) * np.sqrt(numerator / 4)
            assert_allclose(initialized.W, expected, name=initialization)
            assert_allclose(initialized.b, np.zeros((1, 5)), name="initial bias")

    def test_02_activation_derivatives(self):
        X, dout = activation_test_case()
        relu = ReLU()
        assert_allclose(relu.forward(X), np.maximum(X, 0), name="ReLU out")
        analytic = relu.backward(dout)
        assert_allclose(analytic, dout * (X > 0), name="ReLU derivative")
        numeric = numerical_gradient(lambda: np.sum(relu.forward(X) * dout), X)
        self.assertLess(relative_error(analytic, numeric), 1e-7)
        tanh = Tanh()
        out = tanh.forward(X)
        assert_allclose(out, np.tanh(X), name="Tanh out")
        analytic = tanh.backward(dout)
        numeric = numerical_gradient(lambda: np.sum(tanh.forward(X) * dout), X)
        self.assertLess(relative_error(analytic, numeric), 1e-7)
        for activation in [relu, tanh]:
            activation.forward(X)
            with self.assertRaises(ValueError):
                activation.backward(np.ones((1, X.shape[1])))

    def test_03_softmax_cross_entropy(self):
        for logits, y in loss_test_cases():
            loss = SoftmaxCrossEntropy()
            value = loss.forward(logits, y)
            self.assertTrue(np.isfinite(value))
            analytic = loss.backward()
            assert_shape(analytic, logits.shape, "dlogits")
            self.assertTrue(np.all(np.isfinite(analytic)))
            assert_allclose(analytic.sum(axis=1), np.zeros(len(y)), name="gradient row sums")
            numeric = numerical_gradient(lambda: loss.forward(logits, y), logits)
            self.assertLess(relative_error(analytic, numeric), 1e-6)
            self.assertAlmostEqual(value, loss.forward(logits + 10000.0, y), places=9)
        loss = SoftmaxCrossEntropy()
        self.assertAlmostEqual(loss.forward(np.zeros((2, 2)), np.array([0, 1])), np.log(2))
        assert_allclose(loss.backward(), np.array([[-.25, .25], [.25, -.25]]), name="batch mean gradient")
        with self.assertRaises(ValueError):
            loss.forward(np.zeros((2, 2)), np.array([0, 2]))
        with self.assertRaises(ValueError):
            loss.forward(np.zeros((2, 2)), np.array([[0], [1]]))

    def test_04_data_split_and_standardization(self):
        first = load_data(n_samples=200, seed=42)
        second = load_data(n_samples=200, seed=42)
        X, y, Xv, yv = first
        assert_shape(X, (160, 2), "train X")
        assert_shape(y, (160,), "train y")
        assert_shape(Xv, (40, 2), "validation X")
        assert_shape(yv, (40,), "validation y")
        assert_allclose(X.mean(axis=0), np.zeros(2), atol=1e-12, name="train mean")
        assert_allclose(X.std(axis=0), np.ones(2), atol=1e-12, name="train std")
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(set(np.unique(yv)), {0, 1})
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        self.assertFalse(np.shares_memory(X, Xv))
        for a, b in zip(first, second):
            assert_allclose(a, b, name="reproducible dataset")
        self.assertFalse(np.array_equal(X, load_data(n_samples=200, seed=43)[0]))

    def test_05_minibatch_coverage_and_alignment(self):
        X, y = batch_test_case()
        for batch_size in [1, 4, 20]:
            for shuffle in [False, True]:
                batches = list(get_batches(X, y, batch_size, shuffle=shuffle,
                                           rng=np.random.default_rng(7)))
                all_y = np.concatenate([yb for _, yb in batches])
                np.testing.assert_array_equal(np.sort(all_y), y)
                for xb, yb in batches:
                    self.assertGreater(len(yb), 0)
                    self.assertLessEqual(len(yb), batch_size)
                    np.testing.assert_array_equal(xb[:, 0], yb)
                    assert_shape(xb, (len(yb), 2), "batch X")
                if not shuffle:
                    np.testing.assert_array_equal(all_y, y)
        order_a = np.concatenate([b for _, b in get_batches(X, y, 4, rng=np.random.default_rng(11))])
        order_b = np.concatenate([b for _, b in get_batches(X, y, 4, rng=np.random.default_rng(11))])
        np.testing.assert_array_equal(order_a, order_b)
        self.assertFalse(np.array_equal(order_a, y))
        for batch_size in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                list(get_batches(X, y, batch_size))

    def test_06_model_and_sgd_update(self):
        model = build_model()
        dense = [layer for layer in model.layers if isinstance(layer, DenseLayer)]
        self.assertGreaterEqual(len(dense), 3)
        self.assertTrue(any(isinstance(layer, ReLU) for layer in model.layers))
        self.assertTrue(any(isinstance(layer, Tanh) for layer in model.layers))
        X, y = gradient_test_case()
        logits = model.forward(X)
        assert_shape(logits, (len(X), 2), "model logits")
        loss = SoftmaxCrossEntropy()
        initial_loss = loss.forward(logits, y)
        assert_shape(model.backward(loss.backward()), X.shape, "model dX")
        snapshots = []
        for layer in dense:
            assert_shape(layer.dW, layer.W.shape, "dW")
            assert_shape(layer.db, layer.b.shape, "db")
            snapshots.append((layer.W.copy(), layer.b.copy(), layer.dW.copy(), layer.db.copy()))
        learning_rate = 1e-3
        model.step(learning_rate)
        for layer, (W, b, dW, db) in zip(dense, snapshots):
            assert_allclose(layer.W, W - learning_rate * dW, name="SGD W")
            assert_allclose(layer.b, b - learning_rate * db, name="SGD b")
        self.assertLess(evaluate(model, X, y)[0], initial_loss)

    def test_07_gradient_check_and_parameter_restoration(self):
        model = build_model()
        X, y = gradient_test_case()
        dense = [layer for layer in model.layers if isinstance(layer, DenseLayer)]
        original = [(layer.W.copy(), layer.b.copy()) for layer in dense]
        errors = gradient_check(model, X, y)
        self.assertEqual(len(errors), 2 * len(dense))
        for name, error in errors.items():
            self.assertTrue(np.isfinite(error), name)
            self.assertLess(error, 1e-6, name)
        for layer, (W, b) in zip(dense, original):
            np.testing.assert_array_equal(layer.W, W)
            np.testing.assert_array_equal(layer.b, b)
        parameter = np.array([1.0, -2.0])
        saved = parameter.copy()
        assert_allclose(numerical_gradient(lambda: np.sum(parameter ** 2), parameter),
                        2 * parameter, name="central difference")
        np.testing.assert_array_equal(parameter, saved)
        def failing_function():
            raise RuntimeError("intentional evaluation failure")
        with self.assertRaises(RuntimeError):
            numerical_gradient(failing_function, parameter)
        np.testing.assert_array_equal(parameter, saved)

    def test_08_training_loss_and_accuracy(self):
        model, history = train(epochs=80, batch_size=32, learning_rate=0.05, seed=42)
        self.assertEqual(len(history), 80)
        for epoch, row in enumerate(history, start=1):
            self.assertEqual(row["epoch"], epoch)
            for name in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
                self.assertTrue(np.isfinite(row[name]), name)
            for name in ["train_accuracy", "val_accuracy"]:
                self.assertGreaterEqual(row[name], 0)
                self.assertLessEqual(row[name], 1)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"] * 0.5)
        self.assertLess(history[-1]["val_loss"], history[0]["val_loss"] * 0.5)
        self.assertGreaterEqual(history[-1]["val_accuracy"], 0.90)
        _, _, Xv, yv = load_data(seed=42)
        final_loss, final_accuracy = evaluate(model, Xv, yv)
        self.assertAlmostEqual(final_loss, history[-1]["val_loss"])
        self.assertAlmostEqual(final_accuracy, history[-1]["val_accuracy"])


def run_all_tests():
    """8개 검사를 실행하고, 실패하면 notebook에도 AssertionError를 전달한다."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DNNTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("DNN verification failed; inspect the test output above.")
    return result.testsRun


if __name__ == "__main__":
    run_all_tests()
