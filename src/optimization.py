"""Mini-batch SGD 학습과 직접 구현한 gradient checking."""
import numpy as np
from opt_utils import (
    DenseLayer, SoftmaxCrossEntropy, load_data, get_batches,
    build_model, evaluate,
)
from test_utils import numerical_gradient, relative_error


def gradient_check(model, X, y, epsilon=1e-5):
    """모든 Dense의 W/b를 중앙차분으로 검사하고 원래 파라미터/cache를 복원한다."""
    loss = SoftmaxCrossEntropy()
    loss.forward(model.forward(X), y)
    model.backward(loss.backward())
    analytical = [
        (i, name, getattr(layer, 'd' + name).copy())
        for i, layer in enumerate(model.layers)
        if isinstance(layer, DenseLayer)
        for name in ('W', 'b')
    ]
    errors = {}
    for i, name, analytic in analytical:
        parameter = getattr(model.layers[i], name)
        numeric = numerical_gradient(
            lambda: loss.forward(model.forward(X), y), parameter, epsilon
        )
        errors[f'layer_{i}.{name}'] = relative_error(analytic, numeric)
    # 수치미분이 덮어쓴 cache를 원래 파라미터 상태로 다시 계산한다.
    loss.forward(model.forward(X), y)
    model.backward(loss.backward())
    return errors


def train(epochs=200, batch_size=32, learning_rate=0.05, seed=42):
    """매 epoch 끝의 동일 파라미터로 전체 train/validation 지표를 기록한다."""
    if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError('epochs must be a positive integer')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    X, y, X_val, y_val = load_data(seed=seed)
    model = build_model(seed=7)
    loss = SoftmaxCrossEntropy()
    rng = np.random.default_rng(seed + 1)
    history = []
    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in get_batches(X, y, batch_size, rng=rng):
            loss.forward(model.forward(X_batch), y_batch)
            model.backward(loss.backward())
            model.step(learning_rate)
        train_loss, train_accuracy = evaluate(model, X, y)
        val_loss, val_accuracy = evaluate(model, X_val, y_val)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })
    return model, history
