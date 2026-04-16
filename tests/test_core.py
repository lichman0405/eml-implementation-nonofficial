from __future__ import annotations

import numpy as np

import eml


def assert_close(actual, expected, *, atol: float = 1e-9, rtol: float = 1e-9) -> None:
    actual_array = np.asarray(actual, dtype=np.complex128)
    expected_array = np.asarray(expected, dtype=np.complex128)
    assert np.allclose(actual_array, expected_array, atol=atol, rtol=rtol, equal_nan=True)


def test_constants_match_reference_values() -> None:
    assert_close(eml.E, np.e)
    assert_close(eml.I, 1j)
    assert_close(eml.PI, np.pi)
    assert_close(eml.GOLDEN_RATIO, (1 + np.sqrt(5)) / 2)


def test_arithmetic_primitives_match_numpy() -> None:
    left = np.array([0.5, 2.0, -3.0])
    right = np.array([4.0, -1.5, 0.25])

    assert_close(eml.add(left, right), left + right)
    assert_close(eml.subtract(left, right), left - right)
    assert_close(eml.multiply(left, right), left * right)
    assert_close(eml.divide(left, right), left / right)
    assert_close(eml.power(np.abs(left) + 0.5, 0.5), np.power(np.abs(left) + 0.5, 0.5))


def test_log_is_inverse_of_exp_on_positive_axis() -> None:
    values = np.linspace(-2.0, 2.0, 11)
    assert_close(eml.log(eml.exp(values)), values)


def test_hypot_matches_pythagorean_reference() -> None:
    assert_close(eml.hypot(3.0, 4.0), 5.0)
