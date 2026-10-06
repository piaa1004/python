"""배치 우선 DNN 검증에 사용하는 작은 입력과 기대값."""

import numpy as np


def dense_test_case():
    """손으로 계산할 수 있는 Dense 순전파/역전파 사례."""
    return {
        "X": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "W": np.arange(6, dtype=np.float64).reshape(2, 3),
        "b": np.ones((1, 3), dtype=np.float64),
        "dout": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "out": np.array([[7.0, 10.0, 13.0], [13.0, 20.0, 27.0]]),
        "dW": np.array([[13.0, 17.0, 21.0], [18.0, 24.0, 30.0]]),
        "db": np.array([[5.0, 7.0, 9.0]]),
        "dX": np.array([[8.0, 26.0], [17.0, 62.0]]),
    }


def activation_test_case():
    # ReLU의 비미분점인 0은 수치미분 검사에서 제외한다.
    return (
        np.array([[-1.2, -0.3, 0.4], [0.7, -0.8, 1.5]]),
        np.array([[0.2, -0.5, 0.8], [1.0, 0.6, -0.4]]),
    )


def loss_test_cases():
    """이진/다중 클래스, batch=1, 큰 로짓의 안정성을 함께 확인한다."""
    return [
        (np.array([[0.0, 0.0]]), np.array([1], dtype=np.int64)),
        (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([0, 1])),
        (
            np.array([[0.2, -0.5, 1.3], [1.0, 0.4, -0.7]]),
            np.array([2, 0]),
        ),
        (
            np.array([[1000.0, -1000.0], [-1000.0, 1000.0]]),
            np.array([1, 0]),
        ),
    ]


def batch_test_case():
    """첫 번째 feature에 ID를 넣어 shuffle 후 X/y 정렬을 확인한다."""
    y = np.arange(11, dtype=np.int64)
    X = np.column_stack((y.astype(float), y.astype(float) ** 2))
    return X, y


def gradient_test_case():
    return (
        np.array(
            [[0.23, -0.7], [1.3, 0.2], [-0.6, 0.91], [0.8, -1.1]],
            dtype=np.float64,
        ),
        np.array([0, 1, 1, 0], dtype=np.int64),
    )
