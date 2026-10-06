"""Optimization.ipynb의 실습 흐름을 따른 NumPy DNN 구현.

모든 입력/중간 출력은 (batch, features); 자동미분을 사용하지 않는다.
"""

import numpy as np
import matplotlib.pyplot as plt


# 1 - Data pipeline
def load_data(n_samples=1200, noise=0.12, seed=3):
    """NumPy만 사용하여 두 동심원 클래스의 2D toy 데이터를 생성한다."""
    if n_samples < 20 or noise < 0:
        raise ValueError("n_samples >= 20, noise >= 0 이어야 합니다.")
    rng = np.random.default_rng(seed)
    y = np.arange(n_samples) % 2
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    radii = np.where(y == 0, 1.0, 0.45)
    X = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    X += rng.normal(0.0, noise, size=X.shape)
    order = rng.permutation(n_samples)
    return X[order].astype(np.float64), y[order].astype(np.int64)


def split_data(X, y, test_size=0.2, seed=10):
    """각 클래스 비율을 유지하는 train/test split을 직접 구현한다."""
    assert X.ndim == 2 and y.shape == (len(X),)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size는 0과 1 사이여야 합니다.")
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_test = int(round(len(indices) * test_size))
        if n_test == 0 or n_test == len(indices):
            raise ValueError("각 클래스에 train/test 샘플이 필요합니다.")
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize(X_train, *other_sets):
    """train 통계만 사용한다. validation/test의 정보 유출을 방지한다."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    scaled = tuple((X - mean) / std for X in (X_train,) + other_sets)
    return scaled, {"mean": mean, "std": std}


def load_dataset(n_samples=1200, noise=0.12, seed=3):
    """원본 실습의 load_dataset 이름을 유지하며 분할/표준화도 수행한다.

    먼저 train/test를 8:2로 분할한 뒤 train의 20%를 validation으로 분리한다.
    최종 비율은 train:validation:test = 64:16:20이다.
    """
    X, y = load_data(n_samples, noise, seed)
    X_pool, X_test, y_pool, y_test = split_data(X, y, 0.2, seed=10)
    X_train, X_val, y_train, y_val = split_data(X_pool, y_pool, 0.2, seed=11)
    (X_train, X_val, X_test), stats = standardize(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test, stats


# 2 - Gradient Descent (첨부 실습의 딕셔너리/업데이트 식 유지)
def update_parameters_with_gd(parameters, grads, learning_rate):
    """W <- W - learning_rate*dW; b <- b - learning_rate*db."""
    if learning_rate <= 0:
        raise ValueError("learning_rate는 양수여야 합니다.")
    L = len(parameters) // 2
    for l in range(1, L + 1):
        assert parameters["W" + str(l)].shape == grads["dW" + str(l)].shape
        assert parameters["b" + str(l)].shape == grads["db" + str(l)].shape
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


# 3 - Mini-Batch Gradient Descent
def get_batches(X, y, batch_size, shuffle=True, rng=None):
    """(batch, features) 미니배치 iterator. 마지막 작은 배치도 포함한다."""
    assert X.ndim == 2 and y.shape == (len(X),)
    if batch_size < 1 or not isinstance(batch_size, (int, np.integer)):
        raise ValueError("batch_size는 양의 정수여야 합니다.")
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        selected = indices[start:start + batch_size]
        yield X[selected], y[selected]


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """원본 함수 이름을 유지한 래퍼. 새 배치 방향과 1D label을 사용한다."""
    return list(get_batches(X, Y, mini_batch_size, rng=np.random.default_rng(seed)))


# 4 - Dense / Activations / Loss
class DenseLayer:
    def __init__(self, in_features, out_features, initialization="he", rng=None):
        rng = np.random.default_rng(3) if rng is None else rng
        if initialization == "he":
            scale = np.sqrt(2.0 / in_features)
        elif initialization == "xavier":
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError("initialization은 he 또는 xavier입니다.")
        self.W = rng.normal(size=(in_features, out_features)) * scale
        # He/Xavier는 가중치 초기화 규칙; bias는 표준적인 0 초기화.
        self.b = np.zeros((1, out_features), dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.initialization = initialization
        self.cache = None

    def forward(self, X):
        assert X.ndim == 2 and X.shape[1] == self.W.shape[0]
        out = X @ self.W + self.b
        self.cache = {"X": X, "out_shape": out.shape}
        assert out.shape == (X.shape[0], self.W.shape[1])
        return out

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward를 먼저 실행하세요.")
        X = self.cache["X"]
        assert dout.shape == self.cache["out_shape"]
        # loss.backward에서 이미 배치 평균을 적용했으므로 다시 나누지 않는다.
        self.dW = X.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dX = dout @ self.W.T
        assert self.dW.shape == self.W.shape and self.db.shape == self.b.shape
        assert dX.shape == X.shape
        return dX


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        assert X.ndim == 2
        self.cache = X
        out = np.maximum(0.0, X)
        assert out.shape == X.shape
        return out

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward를 먼저 실행하세요.")
        assert dout.shape == self.cache.shape
        # 0에서의 미분은 관례에 따라 0으로 정의한다.
        dX = dout * (self.cache > 0)
        assert dX.shape == self.cache.shape
        return dX


class Tanh:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        assert X.ndim == 2
        out = np.tanh(X)
        self.cache = out
        assert out.shape == X.shape
        return out

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward를 먼저 실행하세요.")
        assert dout.shape == self.cache.shape
        dX = dout * (1.0 - self.cache ** 2)
        assert dX.shape == self.cache.shape
        return dX


class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None

    def forward(self, logits, y):
        assert logits.ndim == 2 and logits.shape[0] > 0
        assert y.shape == (len(logits),) and np.issubdtype(y.dtype, np.integer)
        assert np.all((0 <= y) & (y < logits.shape[1]))
        # log-sum-exp: 큰 logits에도 overflow/underflow로 log(0)이 생기지 않는다.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        probs = np.exp(log_probs)
        loss = -np.mean(log_probs[np.arange(len(y)), y])
        self.cache = {"probs": probs, "y": y.copy()}
        assert np.isfinite(loss) and probs.shape == logits.shape
        return float(loss)

    def backward(self):
        if self.cache is None:
            raise RuntimeError("forward를 먼저 실행하세요.")
        probs, y = self.cache["probs"], self.cache["y"]
        dlogits = probs.copy()
        dlogits[np.arange(len(y)), y] -= 1.0
        dlogits /= len(y)
        assert dlogits.shape == probs.shape
        return dlogits


# 5 - Model container
class DeepNeuralNetwork:
    def __init__(self, layers):
        self.layers = list(layers)
        if sum(isinstance(layer, DenseLayer) for layer in self.layers) < 3:
            raise ValueError("최소 3개 Dense(은닉층 2개 + 출력층)가 필요합니다.")
        assert isinstance(self.layers[-1], DenseLayer)

    @property
    def dense_layers(self):
        return [layer for layer in self.layers if isinstance(layer, DenseLayer)]

    def forward(self, X):
        assert X.ndim == 2
        out = X
        for layer in self.layers:
            out = layer.forward(out)
            assert out.ndim == 2 and out.shape[0] == X.shape[0]
        return out  # 최종 Dense의 logits (추가 활성함수 없음)

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, learning_rate):
        # 원본 실습의 update_parameters_with_gd를 실제 SGD 경로에서 사용한다.
        parameters, grads = {}, {}
        for l, layer in enumerate(self.dense_layers, 1):
            parameters["W" + str(l)], parameters["b" + str(l)] = layer.W, layer.b
            grads["dW" + str(l)], grads["db" + str(l)] = layer.dW, layer.db
        parameters = update_parameters_with_gd(parameters, grads, learning_rate)
        for l, layer in enumerate(self.dense_layers, 1):
            layer.W, layer.b = parameters["W" + str(l)], parameters["b" + str(l)]


def initialize_parameters(layers_dims=(2, 16, 8, 2), seed=3):
    """원본 layers_dims 설정 방식 유지. Dense → ReLU → Dense → Tanh → Dense."""
    if len(layers_dims) != 4:
        raise ValueError("이번 실험은 입력/은닉1/은닉2/출력의 4개 차원을 받습니다.")
    rng = np.random.default_rng(seed)
    return DeepNeuralNetwork([
        DenseLayer(layers_dims[0], layers_dims[1], "he", rng), ReLU(),
        DenseLayer(layers_dims[1], layers_dims[2], "xavier", rng), Tanh(),
        DenseLayer(layers_dims[2], layers_dims[3], "xavier", rng),
    ])


def forward_propagation(X, network):
    """캐시는 각 layer 내부에 보관한다."""
    return network.forward(X)


def backward_propagation(dlogits, network):
    return network.backward(dlogits)


def compute_cost(logits, y):
    return SoftmaxCrossEntropy().forward(logits, y)


def predict(network, X):
    return np.argmax(network.forward(X), axis=1)


def evaluate(network, X, y):
    logits = network.forward(X)
    return {"loss": compute_cost(logits, y),
            "accuracy": float(np.mean(np.argmax(logits, axis=1) == y))}


# 6 - Gradient checking
def gradient_check(network, X, y, epsilon=1e-5):
    """모든 Dense의 W,b와 입력 X를 중앙차분으로 검사한다.

    relative_error = ||analytic-numeric|| / (||analytic||+||numeric||+1e-12)
    각 perturbation에서 파라미터를 복원하고, 마지막에 캐시/gradient도 복원한다.
    ReLU의 0 근방은 미분 불가능하므로 작은 검사 배치를 사용한다.
    """
    criterion = SoftmaxCrossEntropy()
    criterion.forward(network.forward(X), y)
    analytic_dX = network.backward(criterion.backward()).copy()
    relu_margin = min(float(np.min(np.abs(layer.cache)))
                      for layer in network.layers if isinstance(layer, ReLU))
    if relu_margin < 100 * epsilon:
        raise ValueError("ReLU 입력이 0에 가깝습니다. 다른 검사 배치를 사용하세요.")
    analytic = [(layer.dW.copy(), layer.db.copy()) for layer in network.dense_layers]
    rows = []

    def compare(name, parameter, analytic_grad):
        numeric_grad = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            old_value = parameter[index]
            try:
                parameter[index] = old_value + epsilon
                loss_plus = criterion.forward(network.forward(X), y)
                parameter[index] = old_value - epsilon
                loss_minus = criterion.forward(network.forward(X), y)
                numeric_grad[index] = (loss_plus - loss_minus) / (2.0 * epsilon)
            finally:
                parameter[index] = old_value
        relative_error = np.linalg.norm(analytic_grad - numeric_grad) / (
            np.linalg.norm(analytic_grad) + np.linalg.norm(numeric_grad) + 1e-12)
        rows.append({"parameter": name, "shape": list(parameter.shape),
                     "n_checked": int(parameter.size),
                     "relative_error": float(relative_error),
                     "max_absolute_error": float(np.max(np.abs(analytic_grad - numeric_grad)))})

    for l, (layer, (dW, db)) in enumerate(zip(network.dense_layers, analytic), 1):
        compare("W" + str(l), layer.W, dW)
        compare("b" + str(l), layer.b, db)
    compare("X (dX)", X, analytic_dX)
    criterion.forward(network.forward(X), y)
    network.backward(criterion.backward())
    return {"epsilon": epsilon, "threshold": 1e-6,
            "relu_min_abs_input": relu_margin, "results": rows,
            "passed": bool(all(row["relative_error"] < 1e-6 for row in rows))}


# 7 - Training loop (첨부 model()의 epoch/mini-batch/forward/backward/update 순서 유지)
def model(X, Y, X_val, Y_val, layers_dims=(2, 16, 8, 2), optimizer="gd",
          learning_rate=0.1, mini_batch_size=32, num_epochs=300,
          print_cost=True, seed=3):
    if optimizer not in ("gd", "sgd"):
        raise ValueError("이번 제출물은 필수 optimizer인 SGD를 구현합니다.")
    if num_epochs < 1:
        raise ValueError("num_epochs는 양수여야 합니다.")
    network = initialize_parameters(layers_dims, seed)
    criterion = SoftmaxCrossEntropy()
    history = []
    rng = np.random.default_rng(10)  # 두 학습률 실험에서 같은 배치 순서
    for i in range(num_epochs):
        minibatches = get_batches(X, Y, mini_batch_size, shuffle=True, rng=rng)
        cost_total, n_seen = 0.0, 0
        for minibatch_X, minibatch_Y in minibatches:
            logits = forward_propagation(minibatch_X, network)
            cost = criterion.forward(logits, minibatch_Y)
            backward_propagation(criterion.backward(), network)
            network.step(learning_rate)
            cost_total += cost * len(minibatch_X)
            n_seen += len(minibatch_X)
        assert n_seen == len(X)
        # epoch 끝의 동일한 파라미터로 train/val을 평가하여 비교한다.
        train_metrics = evaluate(network, X, Y)
        val_metrics = evaluate(network, X_val, Y_val)
        row = {"epoch": i + 1, "batch_loss": cost_total / n_seen,
               "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]}
        history.append(row)
        if print_cost and (i == 0 or (i + 1) % 50 == 0 or i + 1 == num_epochs):
            print("Epoch %3d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%" % (
                i + 1, row["train_loss"], 100 * row["train_accuracy"],
                row["val_loss"], 100 * row["val_accuracy"]))
    return network, history


# 8 - Experiment / Visualization
def run_experiments(data, learning_rates=(0.01, 0.1), num_epochs=300, print_cost=True):
    X_train, y_train, X_val, y_val, X_test, y_test, stats = data
    networks, histories, summaries = {}, {}, []
    for lr in learning_rates:
        if print_cost:
            print("\nMini-batch SGD, learning_rate =", lr)
        network, history = model(X_train, y_train, X_val, y_val,
                                 learning_rate=lr, num_epochs=num_epochs, print_cost=print_cost)
        key = "lr_" + str(lr)
        networks[key], histories[key] = network, history
        summaries.append({"experiment": key, "learning_rate": lr,
                          "initial_train_loss": evaluate(initialize_parameters(), X_train, y_train)["loss"],
                          "train_loss": history[-1]["train_loss"],
                          "train_accuracy": history[-1]["train_accuracy"],
                          "val_loss": history[-1]["val_loss"],
                          "val_accuracy": history[-1]["val_accuracy"]})
    # 테스트 데이터는 학습률 선택에 쓰지 않는다. val loss로 선택 후 최종 1회 평가.
    best = min(summaries, key=lambda row: row["val_loss"])
    final_test = evaluate(networks[best["experiment"]], X_test, y_test)
    return networks, histories, {"experiments": summaries,
                                "selected_experiment": best["experiment"],
                                "selection_rule": "lowest final validation loss",
                                "test_loss": final_test["loss"],
                                "test_accuracy": final_test["accuracy"],
                                "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
                                "layers_dims": [2, 16, 8, 2], "epochs": num_epochs,
                                "batch_size": 32, "data_seed": 3, "model_seed": 3,
                                "batch_seed": 10, "noise": 0.12,
                                "standardization": {k: v.tolist() for k, v in stats.items()}}


def plot_learning_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, history in histories.items():
        epochs = [row["epoch"] for row in history]
        line, = axes[0].plot(epochs, [row["train_loss"] for row in history], label=key + " train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], "--", color=line.get_color(), label=key + " val")
        axes[1].plot(epochs, [row["train_accuracy"] for row in history], color=line.get_color(), label=key + " train")
        axes[1].plot(epochs, [row["val_accuracy"] for row in history], "--", color=line.get_color(), label=key + " val")
    for ax, ylabel in zip(axes, ("Softmax cross entropy", "Accuracy")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_title("Training / validation loss")
    axes[1].set_title("Training / validation accuracy")
    fig.tight_layout()
    return fig


def plot_decision_boundary(network, X, y):
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    Z = predict(network, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Standardized x1")
    ax.set_ylabel("Standardized x2")
    ax.set_title("NumPy DNN decision boundary (test set)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
