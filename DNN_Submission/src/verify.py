"""실행: python src/verify.py

자동미분 없이 중앙차분으로 각 모듈의 backward를 독립적으로 검증한다.
결과는 results/module_verification.json에 저장한다.
첨부 public_tests.py의 SGD 호환성 결과는 제출된 verification.json에 별도 기록했다.
"""
from pathlib import Path
import json
import traceback

import numpy as np
import opt_utils as u

RESULTS = []
RNG = np.random.default_rng(2026)


def numeric_gradient(array, scalar_function, epsilon=1e-5):
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        try:
            array[index] = original + epsilon
            plus = scalar_function()
            array[index] = original - epsilon
            minus = scalar_function()
            result[index] = (plus - minus) / (2 * epsilon)
        finally:
            array[index] = original
    return result


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) /
                 (np.linalg.norm(actual) + np.linalg.norm(expected) + 1e-12))


def check(name, function):
    try:
        details = function()
        RESULTS.append({"name": name, "passed": True, "details": details})
    except Exception as exc:
        RESULTS.append({"name": name, "passed": False,
                        "error": repr(exc), "traceback": traceback.format_exc()})


def dense_check():
    X = RNG.normal(size=(4, 3))
    upstream = RNG.normal(size=(4, 5))
    layer = u.DenseLayer(3, 5, rng=RNG)
    layer.forward(X)
    dX = layer.backward(upstream)
    scalar = lambda: float(np.sum(layer.forward(X) * upstream))
    errors = {
        "dW": relative_error(layer.dW.copy(), numeric_gradient(layer.W, scalar)),
        "db": relative_error(layer.db.copy(), numeric_gradient(layer.b, scalar)),
        "dX": relative_error(dX, numeric_gradient(X, scalar)),
    }
    assert all(value < 1e-8 for value in errors.values()), errors
    assert dX.shape == X.shape and layer.db.shape == (1, 5)
    return {"relative_errors": errors, "threshold": 1e-8}


def activation_check(cls):
    # ReLU inputs avoid its non-differentiable point for the numerical test.
    X = np.array([[-1.3, 0.6, 2.0], [0.4, -0.8, 1.7]])
    upstream = RNG.normal(size=X.shape)
    layer = cls()
    layer.forward(X)
    analytic = layer.backward(upstream)
    numeric = numeric_gradient(X, lambda: float(np.sum(layer.forward(X) * upstream)))
    error = relative_error(analytic, numeric)
    assert error < 1e-8, error
    if cls is u.ReLU:
        layer.forward(np.zeros((1, 1)))
        assert layer.backward(np.ones((1, 1)))[0, 0] == 0
    return {"relative_error": error, "threshold": 1e-8}


def softmax_check():
    logits = RNG.normal(size=(5, 3))
    y = np.array([0, 1, 2, 2, 0])
    criterion = u.SoftmaxCrossEntropy()
    loss = criterion.forward(logits, y)
    analytic = criterion.backward()
    numeric = numeric_gradient(logits, lambda: criterion.forward(logits, y))
    error = relative_error(analytic, numeric)
    assert error < 1e-8, error
    np.testing.assert_allclose(analytic.sum(axis=1), 0, atol=1e-15)
    shifted = criterion.forward(logits + 10000.0, y)
    np.testing.assert_allclose(shifted, loss, atol=1e-11)
    extreme_loss = criterion.forward(np.array([[1000., -1000.], [-1000., 1000.]]),
                                    np.array([1, 1]))
    assert np.isfinite(extreme_loss) and np.all(np.isfinite(criterion.backward()))
    np.testing.assert_allclose(extreme_loss, 1000.0)
    return {"relative_error": error, "threshold": 1e-8,
            "extreme_logits_loss": extreme_loss,
            "shift_invariance_absolute_error": abs(shifted - loss)}


def average_gradient_check():
    X = RNG.normal(size=(7, 3))
    y = RNG.integers(0, 2, size=7)
    layer = u.DenseLayer(3, 2, rng=RNG)
    criterion = u.SoftmaxCrossEntropy()
    loss = criterion.forward(layer.forward(X), y)
    layer.backward(criterion.backward())
    dW, db = layer.dW.copy(), layer.db.copy()
    doubled_loss = criterion.forward(layer.forward(np.concatenate([X, X])),
                                      np.concatenate([y, y]))
    layer.backward(criterion.backward())
    np.testing.assert_allclose(loss, doubled_loss, rtol=1e-14)
    np.testing.assert_allclose(dW, layer.dW, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(db, layer.db, rtol=1e-13, atol=1e-13)
    return {"description": "Duplicating a batch preserves mean loss and Dense gradients."}


def full_network_check():
    data = u.load_dataset()
    X, y = data[0][:5].copy(), data[1][:5]
    network = u.initialize_parameters()
    originals = [(layer.W.copy(), layer.b.copy()) for layer in network.dense_layers]
    original_X = X.copy()
    result = u.gradient_check(network, X, y)
    assert result["passed"], result
    for layer, (W, b) in zip(network.dense_layers, originals):
        np.testing.assert_array_equal(layer.W, W)
        np.testing.assert_array_equal(layer.b, b)
    np.testing.assert_array_equal(X, original_X)
    assert network.layers[0].cache["X"].shape == X.shape
    return {"relative_errors": {row["parameter"]: row["relative_error"]
                               for row in result["results"]},
            "all_parameters_and_input_restored": True,
            "threshold": result["threshold"]}


def data_check():
    X, y = u.load_data()
    X_again, y_again = u.load_data()
    np.testing.assert_array_equal(X, X_again)
    np.testing.assert_array_equal(y, y_again)
    ids = np.arange(len(X)).reshape(-1, 1)
    id_train, id_test, yt, ye = u.split_data(ids, y)
    assert len(id_train) == 960 and len(id_test) == 240
    assert not set(id_train[:, 0]) & set(id_test[:, 0])
    assert set(id_train[:, 0]) | set(id_test[:, 0]) == set(range(1200))
    np.testing.assert_allclose(yt.mean(), ye.mean())
    raw_pool, raw_test, y_pool, _ = u.split_data(X, y, .2, 10)
    raw_train, raw_val, _, _ = u.split_data(raw_pool, y_pool, .2, 11)
    train, _, val, _, test, _, stats = u.load_dataset()
    np.testing.assert_allclose(stats["mean"], raw_train.mean(axis=0, keepdims=True))
    np.testing.assert_allclose(stats["std"], raw_train.std(axis=0, keepdims=True))
    np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(train.std(axis=0), 1, atol=1e-14)
    np.testing.assert_allclose(val, (raw_val - stats["mean"]) / stats["std"])
    np.testing.assert_allclose(test, (raw_test - stats["mean"]) / stats["std"])
    return {"final_split_sizes": [len(train), len(val), len(test)],
            "stratified_disjoint_split": True, "train_only_standardization": True}


def batches_check():
    X = np.arange(22).reshape(11, 2)
    y = np.arange(11)
    batches = list(u.get_batches(X, y, 4, shuffle=False))
    assert [len(bx) for bx, _ in batches] == [4, 4, 3]
    assert all(bx.shape == (len(by), 2) and by.ndim == 1 for bx, by in batches)
    np.testing.assert_array_equal(np.concatenate([bx for bx, _ in batches]), X)
    np.testing.assert_array_equal(np.concatenate([by for _, by in batches]), y)
    shuffled = list(u.get_batches(X, y, 4, rng=np.random.default_rng(13)))
    y_shuffled = np.concatenate([by for _, by in shuffled])
    X_shuffled = np.concatenate([bx for bx, _ in shuffled])
    np.testing.assert_array_equal(np.sort(y_shuffled), y)
    np.testing.assert_array_equal(X_shuffled[:, 0], 2 * y_shuffled)
    assert not np.array_equal(y_shuffled, y)
    return {"batch_sizes": [4, 4, 3], "covers_each_sample_once": True,
            "shuffle_preserves_features_labels_alignment": True}


def learning_check():
    data = u.load_dataset(n_samples=300)
    X, y, X_val, y_val = data[:4]
    initial_loss = u.evaluate(u.initialize_parameters(), X, y)["loss"]
    network, history = u.model(X, y, X_val, y_val, num_epochs=100,
                               mini_batch_size=31, print_cost=False)
    final = u.evaluate(network, X, y)
    assert len(history) == 100
    assert final["loss"] < 0.5 * initial_loss, (initial_loss, final)
    assert final["accuracy"] > .9, final
    assert all(np.isfinite(row[key]) for row in history
               for key in ("train_loss", "val_loss", "batch_loss"))
    return {"epochs": 100, "batch_size": 31,
            "initial_train_loss": initial_loss,
            "final_train_loss": final["loss"],
            "final_train_accuracy": final["accuracy"]}


def main():
    check("Dense dW, db, dX central finite differences", dense_check)
    check("ReLU central finite difference and zero derivative convention", lambda: activation_check(u.ReLU))
    check("Tanh central finite difference", lambda: activation_check(u.Tanh))
    check("Softmax cross entropy gradient and numerical stability", softmax_check)
    check("Mean loss gradient normalized once per batch", average_gradient_check)
    check("All DNN parameters and input gradient check; state restoration", full_network_check)
    check("Data generation, disjoint stratified split, train-only standardization", data_check)
    check("Mini-batch remainder, coverage, feature-label alignment", batches_check)
    check("Actual mini-batch learning reduces loss with manual backprop", learning_check)

    report = {
        "scope": "Independent central-difference and learning verification of src/opt_utils.py",
        "passed": all(row["passed"] for row in RESULTS),
        "n_passed": sum(row["passed"] for row in RESULTS),
        "n_checks": len(RESULTS),
        "checks": RESULTS,
    }
    destination = Path(__file__).resolve().parents[1] / "results" / "module_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
