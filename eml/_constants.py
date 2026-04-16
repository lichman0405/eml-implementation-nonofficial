from __future__ import annotations

from ._core import eml_add, eml_div, eml_exp, eml_int, eml_inv, eml_log, eml_mul, eml_neg, eml_pow, eml_zero

ONE = 1.0
ZERO = eml_zero()
NEG_ONE = eml_neg(ONE)
TWO = eml_int(2)
THREE = eml_int(3)
FOUR = eml_int(4)
FIVE = eml_int(5)
TEN = eml_int(10)
ONE_EIGHTY = eml_int(180)

HALF = eml_div(ONE, TWO)
QUARTER = eml_div(ONE, FOUR)
E = eml_exp(ONE)


def _imag_unit():
    return eml_neg(eml_exp(eml_div(eml_log(NEG_ONE), TWO)))


I = _imag_unit()
PI = eml_mul(I, eml_log(NEG_ONE))
TAU = eml_add(PI, PI)
GOLDEN_RATIO = eml_div(eml_add(ONE, eml_pow(FIVE, HALF)), TWO)
PHI = GOLDEN_RATIO

SQRT_TWO = eml_pow(TWO, HALF)
SQRT_THREE = eml_pow(THREE, HALF)
LN_TWO = eml_log(TWO)
LN_TEN = eml_log(TEN)
INV_E = eml_inv(E)
INV_PI = eml_inv(PI)

e = E
i = I
pi = PI
tau = TAU
phi = GOLDEN_RATIO

__all__ = [
    "E",
    "FIVE",
    "FOUR",
    "GOLDEN_RATIO",
    "HALF",
    "I",
    "INV_E",
    "INV_PI",
    "LN_TEN",
    "LN_TWO",
    "NEG_ONE",
    "ONE",
    "ONE_EIGHTY",
    "PHI",
    "PI",
    "QUARTER",
    "SQRT_THREE",
    "SQRT_TWO",
    "TAU",
    "TEN",
    "THREE",
    "TWO",
    "ZERO",
    "e",
    "i",
    "phi",
    "pi",
    "tau",
]
