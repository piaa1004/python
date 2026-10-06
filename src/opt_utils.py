"""DNN 레이어·데이터·모델 유틸리티. 모든 입력은 (batch, features).
첨부 opt_utils.py의 역할을 유지하면서 수동 미분과 새 배열 방향에 맞게 재구성했다.
"""

import numpy as np


def load_data(n_samples=1000, seed=42):
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 10:
        raise ValueError("n_samples must be an integer >= 10")
    rng = np.random.default_rng(seed)
    y = np.arange(n_samples) % 2
    t = rng.uniform(0, 2 * np.pi, n_samples)
    r = np.where(y == 0, 1.0, 0.4)
    X = np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(0, 0.06, (n_samples, 2))
    idx = rng.permutation(n_samples)
    cut = int(0.8 * n_samples)
    a, b = (idx[:cut], idx[cut:])
    mean, std = (X[a].mean(0), np.maximum(X[a].std(0), 1e-12))
    return ((X[a] - mean) / std, y[a], (X[b] - mean) / std, y[b])


def get_batches(X, y, batch_size, shuffle=True, rng=None):
    if X.ndim != 2 or y.shape != (len(X),) or (not len(X)):
        raise ValueError('X:(N,D), y:(N,)')
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError('positive integer batch size')
    rng = np.random.default_rng() if rng is None else rng
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        i = idx[start:start + batch_size]
        yield (X[i], y[i])


class DenseLayer:

    def __init__(self, in_features, out_features, rng, initialization='he'):
        if initialization not in ('he', 'xavier'):
            raise ValueError('he or xavier')
        self.W = rng.normal(size=(in_features, out_features)) * np.sqrt((2 if initialization == 'he' else 1) / in_features)
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError('input shape')
        self.X = X.copy()
        return X @ self.W + self.b

    def backward(self, dout):
        if dout.shape != (len(self.X), self.W.shape[1]):
            raise ValueError('gradient shape')
        self.dW = self.X.T @ dout
        self.db = dout.sum(0, keepdims=True)
        return dout @ self.W.T


class ReLU:

    def forward(self, X):
        self.mask = X > 0
        return np.maximum(X, 0)

    def backward(self, dout):
        if dout.shape != self.mask.shape:
            raise ValueError('gradient shape')
        return dout * self.mask


class Tanh:

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out.copy()

    def backward(self, dout):
        if dout.shape != self.out.shape:
            raise ValueError('gradient shape')
        return dout * (1 - self.out ** 2)


class SoftmaxCrossEntropy:

    def forward(self, logits, y):
        if logits.ndim != 2 or not len(logits) or y.shape != (len(logits),):
            raise ValueError('logits:(N,C), y:(N,)')
        if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= logits.shape[1]):
            raise ValueError('class indices')
        z = logits - logits.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        self.probs = np.exp(logp)
        self.y = y.copy()
        return float(-logp[np.arange(len(y)), y].mean())

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.y)), self.y] -= 1
        return grad / len(self.y)


class DeepNeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, learning_rate):
        if learning_rate <= 0:
            raise ValueError('positive learning rate')
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db


def build_model(seed=7):
    rng = np.random.default_rng(seed)
    return DeepNeuralNetwork([DenseLayer(2, 16, rng), ReLU(), DenseLayer(16, 8, rng, 'xavier'), Tanh(), DenseLayer(8, 2, rng, 'xavier')])


def evaluate(model, X, y):
    logits = model.forward(X)
    return (SoftmaxCrossEntropy().forward(logits, y), float(np.mean(logits.argmax(1) == y)))
