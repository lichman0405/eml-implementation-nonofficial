from __future__ import annotations

import numpy as np
import pytest

import eml


def assert_close(actual, expected, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    actual_array = np.asarray(actual, dtype=np.complex128)
    expected_array = np.asarray(expected, dtype=np.complex128)
    assert np.allclose(actual_array, expected_array, atol=atol, rtol=rtol, equal_nan=True)


@pytest.mark.parametrize(
    ("func", "reference", "values"),
    [
        (eml.exp, np.exp, np.linspace(-2.0, 2.0, 21)),
        (eml.exp2, np.exp2, np.linspace(-2.0, 2.0, 21)),
        (eml.expm1, np.expm1, np.linspace(-2.0, 2.0, 21)),
        (eml.log, np.log, np.linspace(0.25, 4.0, 21)),
        (eml.log2, np.log2, np.linspace(0.25, 4.0, 21)),
        (eml.log10, np.log10, np.linspace(0.25, 4.0, 21)),
        (eml.log1p, np.log1p, np.linspace(-0.8, 4.0, 21)),
        (eml.sqrt, np.sqrt, np.linspace(0.0, 5.0, 21)),
        (eml.sin, np.sin, np.linspace(-1.5, 1.5, 21)),
        (eml.cos, np.cos, np.linspace(-1.5, 1.5, 21)),
        (eml.tan, np.tan, np.linspace(-1.0, 1.0, 21)),
        (eml.sinh, np.sinh, np.linspace(-2.0, 2.0, 21)),
        (eml.cosh, np.cosh, np.linspace(-2.0, 2.0, 21)),
        (eml.tanh, np.tanh, np.linspace(-2.0, 2.0, 21)),
        (eml.asin, np.arcsin, np.linspace(-0.8, 0.8, 21)),
        (eml.acos, np.arccos, np.linspace(-0.8, 0.8, 21)),
        (eml.atan, np.arctan, np.linspace(-3.0, 3.0, 21)),
        (eml.asinh, np.arcsinh, np.linspace(-3.0, 3.0, 21)),
        (eml.acosh, np.arccosh, np.linspace(1.1, 4.0, 21)),
        (eml.atanh, np.arctanh, np.linspace(-0.8, 0.8, 21)),
    ],
)
def test_unary_functions_against_numpy(func, reference, values) -> None:
    assert_close(func(values), reference(values))


def test_reciprocal_trigonometric_functions() -> None:
    values = np.linspace(0.4, 1.2, 17)
    assert_close(eml.sec(values), 1 / np.cos(values))
    assert_close(eml.csc(values), 1 / np.sin(values))
    assert_close(eml.cot(values), 1 / np.tan(values))


def test_inverse_reciprocal_trigonometric_functions() -> None:
    values = np.linspace(1.2, 3.0, 17)
    assert_close(eml.asec(values), np.arccos(1 / values))
    assert_close(eml.acsc(values), np.arcsin(1 / values))
    assert_close(eml.acot(values), np.arctan(1 / values))


def test_logistic_sigmoid_against_reference_formula() -> None:
    values = np.linspace(-8.0, 8.0, 41)
    expected = 1 / (1 + np.exp(-values))
    assert_close(eml.logistic_sigmoid(values), expected)


def test_binary_helpers_against_numpy() -> None:
    left = np.linspace(0.5, 2.5, 13)
    right = np.linspace(1.0, 3.0, 13)
    assert_close(eml.avg(left, right), (left + right) / 2)
    assert_close(eml.log_base(left + 1.0, right + 1.0), np.log(right + 1.0) / np.log(left + 1.0))


def test_standard_library_style_helpers() -> None:
    values = np.linspace(-5.0, 5.0, 21)
    assert_close(eml.absolute(values), np.abs(values))
    assert_close(eml.radians(np.array([0.0, 90.0, 180.0])), np.deg2rad([0.0, 90.0, 180.0]))
    assert_close(eml.degrees(np.array([0.0, np.pi / 2, np.pi])), np.rad2deg([0.0, np.pi / 2, np.pi]))


def test_lowercase_constant_aliases() -> None:
    assert_close(eml.e, np.e)
    assert_close(eml.i, 1j)
    assert_close(eml.pi, np.pi)
    assert_close(eml.tau, 2 * np.pi)
    assert_close(eml.phi, (1 + np.sqrt(5)) / 2)
