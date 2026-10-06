"""shape/값 검증 및 자동미분 없이 수행하는 중앙차분 도구."""

import numpy as np


def assert_shape(actual, expected, name="array"):
    expected = tuple(expected)
    if np.shape(actual) != expected:
        raise AssertionError(
            f"{name}: expected shape {expected}, got {np.shape(actual)}"
        )


def assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, name="array"):
    """broadcast로 잘못된 shape가 통과하지 않도록 shape도 검사한다."""
    assert_shape(actual, np.shape(expected), name)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                               err_msg=name)


def numerical_gradient(function, parameter, epsilon=1e-5):
    """function()의 스칼라 결과를 parameter 각 원소로 중앙차분한다.

    function은 parameter를 참조하는 인자 없는 함수이다.
    원소를 +/- epsilon 바꿔 평가한 뒤 원래 값을 반드시 복원한다.
    """
    if not isinstance(parameter, np.ndarray):
        raise TypeError("parameter must be a NumPy array")
    if not np.issubdtype(parameter.dtype, np.floating):
        raise TypeError("parameter must have a floating-point dtype")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    gradient = np.zeros_like(parameter, dtype=np.float64)
    for index in np.ndindex(parameter.shape):
        original = parameter[index].copy()
        try:
            parameter[index] = original + epsilon
            plus = float(function())
            parameter[index] = original - epsilon
            minus = float(function())
            gradient[index] = (plus - minus) / (2.0 * epsilon)
        finally:
            parameter[index] = original
    return gradient


def relative_error(analytic, numeric):
    """L2 norm 기반 상대오차. 0 gradient끼리 비교하는 경우도 처리한다."""
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)
    assert_shape(analytic, numeric.shape, "gradient")
    return float(
        np.linalg.norm(analytic - numeric)
        / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12)
    )
