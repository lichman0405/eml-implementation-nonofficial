from __future__ import annotations

from ._constants import HALF, I, ONE, ONE_EIGHTY, PI, TEN, THREE, TWO
from ._core import ArrayLike, eml_add, eml_div, eml_exp, eml_int, eml_inv, eml_log, eml_mul, eml_neg, eml_pow, eml_sub

add = eml_add
subtract = eml_sub
multiply = eml_mul
divide = eml_div
power = eml_pow
reciprocal = eml_inv
negate = eml_neg


def identity(value: ArrayLike) -> ArrayLike:
    return value


def exp(value: ArrayLike) -> ArrayLike:
    return eml_exp(value)


def exp2(value: ArrayLike) -> ArrayLike:
    return power(TWO, value)


def expm1(value: ArrayLike) -> ArrayLike:
    return subtract(exp(value), ONE)


def log(value: ArrayLike, base: ArrayLike | None = None) -> ArrayLike:
    natural = eml_log(value)
    if base is None:
        return natural
    return divide(natural, eml_log(base))


def log_base(base: ArrayLike, value: ArrayLike) -> ArrayLike:
    return divide(eml_log(value), eml_log(base))


def log2(value: ArrayLike) -> ArrayLike:
    return log_base(TWO, value)


def log10(value: ArrayLike) -> ArrayLike:
    return log_base(TEN, value)


def log1p(value: ArrayLike) -> ArrayLike:
    return log(add(ONE, value))


def half(value: ArrayLike) -> ArrayLike:
    return divide(value, TWO)


def square(value: ArrayLike) -> ArrayLike:
    return multiply(value, value)


def sqrt(value: ArrayLike) -> ArrayLike:
    return power(value, HALF)


def cbrt(value: ArrayLike) -> ArrayLike:
    return power(value, divide(ONE, THREE))


def absolute(value: ArrayLike) -> ArrayLike:
    return sqrt(square(value))


def avg(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    return half(add(left, right))


def hypot(left: ArrayLike, right: ArrayLike) -> ArrayLike:
    return sqrt(add(square(left), square(right)))


def radians(value: ArrayLike) -> ArrayLike:
    return divide(multiply(value, PI), ONE_EIGHTY)


def degrees(value: ArrayLike) -> ArrayLike:
    return divide(multiply(value, ONE_EIGHTY), PI)


def logistic_sigmoid(value: ArrayLike) -> ArrayLike:
    return multiply(HALF, add(ONE, tanh(half(value))))


def sin(value: ArrayLike) -> ArrayLike:
    imag_value = multiply(I, value)
    numerator = subtract(exp(imag_value), exp(negate(imag_value)))
    denominator = multiply(TWO, I)
    return divide(numerator, denominator)


def cos(value: ArrayLike) -> ArrayLike:
    imag_value = multiply(I, value)
    numerator = add(exp(imag_value), exp(negate(imag_value)))
    return divide(numerator, TWO)


def tan(value: ArrayLike) -> ArrayLike:
    return divide(sin(value), cos(value))


def cot(value: ArrayLike) -> ArrayLike:
    return divide(cos(value), sin(value))


def sec(value: ArrayLike) -> ArrayLike:
    return reciprocal(cos(value))


def csc(value: ArrayLike) -> ArrayLike:
    return reciprocal(sin(value))


def sinh(value: ArrayLike) -> ArrayLike:
    return divide(subtract(exp(value), exp(negate(value))), TWO)


def cosh(value: ArrayLike) -> ArrayLike:
    return divide(add(exp(value), exp(negate(value))), TWO)


def tanh(value: ArrayLike) -> ArrayLike:
    return divide(sinh(value), cosh(value))


def asin(value: ArrayLike) -> ArrayLike:
    inner = add(negate(multiply(I, value)), sqrt(subtract(ONE, square(value))))
    return multiply(I, log(inner))


def acos(value: ArrayLike) -> ArrayLike:
    radical = multiply(sqrt(subtract(value, ONE)), sqrt(add(value, ONE)))
    return multiply(I, log(add(value, radical)))


def atan(value: ArrayLike) -> ArrayLike:
    numerator = add(negate(I), value)
    denominator = subtract(negate(I), value)
    factor = negate(divide(I, TWO))
    return multiply(factor, log(divide(numerator, denominator)))


def acot(value: ArrayLike) -> ArrayLike:
    return atan(reciprocal(value))


def asec(value: ArrayLike) -> ArrayLike:
    return acos(reciprocal(value))


def acsc(value: ArrayLike) -> ArrayLike:
    return asin(reciprocal(value))


def asinh(value: ArrayLike) -> ArrayLike:
    return log(add(value, sqrt(add(square(value), ONE))))


def acosh(value: ArrayLike) -> ArrayLike:
    radical = multiply(sqrt(add(value, ONE)), sqrt(subtract(value, ONE)))
    return log(add(value, radical))


def atanh(value: ArrayLike) -> ArrayLike:
    return multiply(HALF, log(divide(add(ONE, value), subtract(ONE, value))))


arcsin = asin
arccos = acos
arctan = atan
arcsinh = asinh
arccosh = acosh
arctanh = atanh
sigmoid = logistic_sigmoid
abs_value = absolute


__all__ = [
    "absolute",
    "abs_value",
    "acos",
    "acosh",
    "acot",
    "acsc",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "asec",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "avg",
    "cbrt",
    "cos",
    "cosh",
    "cot",
    "csc",
    "degrees",
    "divide",
    "exp",
    "exp2",
    "expm1",
    "half",
    "hypot",
    "identity",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_base",
    "logistic_sigmoid",
    "multiply",
    "negate",
    "power",
    "radians",
    "reciprocal",
    "sec",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
]
