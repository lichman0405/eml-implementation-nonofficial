from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

ArrayLike: TypeAlias = Any
DEFAULT_REAL_TOL = 1e-10


def _as_complex(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


def _to_python_scalar(value: np.ndarray | np.generic) -> ArrayLike:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finalize(value: ArrayLike, *, real_tol: float = DEFAULT_REAL_TOL) -> ArrayLike:
    array = np.asarray(value)
    if np.iscomplexobj(array) and np.all(np.abs(array.imag) <= real_tol):
        array = array.real
    return _to_python_scalar(array)


def eml(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    left_array = _as_complex(left)
    right_array = _as_complex(right)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        result = np.exp(left_array) - np.log(right_array)
    return _finalize(result)


def eml_exp(value: ArrayLike) -> ArrayLike:
    return eml(value, 1)


def eml_log(value: ArrayLike) -> ArrayLike:
    return eml(1, eml_exp(eml(1, value)))


def eml_zero() -> ArrayLike:
    return eml_log(1)


def eml_sub(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    return eml(eml_log(left), eml_exp(right))


def eml_neg(value: ArrayLike) -> ArrayLike:
    return eml_sub(eml_zero(), value)


def eml_add(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    return eml_sub(left, eml_neg(right))


def eml_inv(value: ArrayLike) -> ArrayLike:
    return eml_exp(eml_neg(eml_log(value)))


def eml_mul(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    return eml_exp(eml_add(eml_log(left), eml_log(right)))


def eml_div(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    return eml_mul(left, eml_inv(right))


def eml_pow(base: ArrayLike, exponent: ArrayLike) -> ArrayLike:
    return eml_exp(eml_mul(exponent, eml_log(base)))


def eml_int(value: int) -> ArrayLike:
    if value == 0:
        return eml_zero()
    if value == 1:
        return 1.0
    if value < 0:
        return eml_neg(eml_int(-value))

    acc: ArrayLike | None = None
    term: ArrayLike = 1.0
    remaining = value

    while remaining > 0:
        if remaining & 1:
            acc = term if acc is None else eml_add(acc, term)
        term = eml_add(term, term)
        remaining >>= 1

    return acc


def eml_rational(numerator: int, denominator: int) -> ArrayLike:
    if denominator == 0:
        raise ZeroDivisionError("EML rational values require a non-zero denominator.")
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 1:
        return eml_int(numerator)

    magnitude = eml_div(eml_int(abs(numerator)), eml_int(denominator))
    return magnitude if numerator >= 0 else eml_neg(magnitude)


__all__ = [
    "ArrayLike",
    "DEFAULT_REAL_TOL",
    "eml",
    "eml_add",
    "eml_div",
    "eml_exp",
    "eml_int",
    "eml_inv",
    "eml_log",
    "eml_mul",
    "eml_neg",
    "eml_pow",
    "eml_rational",
    "eml_sub",
    "eml_zero",
]
