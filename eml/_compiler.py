from __future__ import annotations

import re
from typing import Any

from sympy import Abs, Add, E as SYM_E, Float, GoldenRatio as SYM_GOLDEN_RATIO, I as SYM_I, Integer, Mul, Pow, Rational, Symbol
from sympy import cos, cosh, exp as sym_exp, log as sym_log, pi as SYM_PI, sin, sinh, sqrt, sympify, tan, tanh

from ._tree import EMLTree

SYM_TAU = 2 * SYM_PI


def _sec(value):
    return 1 / cos(value)


def _csc(value):
    return 1 / sin(value)


def _cot(value):
    return cos(value) / sin(value)


def _asec(value):
    return _acos_log(1 / value)


def _acsc(value):
    return _asin_log(1 / value)


def _acot(value):
    return _atan_log(1 / value)


def _asin_log(value):
    return SYM_I * sym_log(-SYM_I * value + sqrt(1 - value**2))


def _acos_log(value):
    return SYM_I * sym_log(value + sqrt(value - 1) * sqrt(value + 1))


def _atan_log(value):
    return (-SYM_I / 2) * sym_log((-SYM_I + value) / (-SYM_I - value))


def _asinh_log(value):
    return sym_log(value + sqrt(value**2 + 1))


def _acosh_log(value):
    return sym_log(value + sqrt(value + 1) * sqrt(value - 1))


def _atanh_log(value):
    return Rational(1, 2) * sym_log((1 + value) / (1 - value))


def _half(value):
    return value / 2


def _square(value):
    return value**2


def _cbrt(value):
    return value ** Rational(1, 3)


def _avg(left, right):
    return (left + right) / 2


def _hypot(left, right):
    return sqrt(left**2 + right**2)


def _logistic_sigmoid(value):
    return Rational(1, 2) * (1 + tanh(value / 2))


def _absolute(value):
    return sqrt(value**2)


def _plus(left, right):
    return left + right


def _subtract(left, right):
    return left - right


def _times(left, right):
    return left * right


def _divide(left, right):
    return left / right


def _power(left, right):
    return left**right


def _minus(value):
    return -value


def _inv(value):
    return 1 / value


def _identity(value):
    return value


def _log_dispatch_python(*args):
    if len(args) == 1:
        return sym_log(args[0])
    if len(args) == 2:
        value, base = args
        return sym_log(value) / sym_log(base)
    raise TypeError("log expects one or two arguments.")


def _log_dispatch_wolfram(*args):
    if len(args) == 1:
        return sym_log(args[0])
    if len(args) == 2:
        base, value = args
        return sym_log(value) / sym_log(base)
    raise TypeError("Log expects one or two arguments.")


def _log_base(base, value):
    return sym_log(value) / sym_log(base)


def _log2(value):
    return sym_log(value) / sym_log(2)


def _log10(value):
    return sym_log(value) / sym_log(10)


def _log1p(value):
    return sym_log(1 + value)


def _exp2(value):
    return 2**value


def _expm1(value):
    return sym_exp(value) - 1


def _degrees(value):
    return value * 180 / SYM_PI


def _radians(value):
    return value * SYM_PI / 180


LOCALS = {
    "E": SYM_E,
    "e": SYM_E,
    "I": SYM_I,
    "i": SYM_I,
    "GoldenRatio": SYM_GOLDEN_RATIO,
    "golden_ratio": SYM_GOLDEN_RATIO,
    "phi": SYM_GOLDEN_RATIO,
    "Phi": SYM_GOLDEN_RATIO,
    "Pi": SYM_PI,
    "pi": SYM_PI,
    "Tau": SYM_TAU,
    "tau": SYM_TAU,
    "exp": sym_exp,
    "Exp": sym_exp,
    "log": _log_dispatch_python,
    "Log": _log_dispatch_wolfram,
    "ln": sym_log,
    "Ln": sym_log,
    "sin": sin,
    "Sin": sin,
    "cos": cos,
    "Cos": cos,
    "tan": tan,
    "Tan": tan,
    "sec": _sec,
    "Sec": _sec,
    "csc": _csc,
    "Csc": _csc,
    "cot": _cot,
    "Cot": _cot,
    "asin": _asin_log,
    "arcsin": _asin_log,
    "ArcSin": _asin_log,
    "acos": _acos_log,
    "arccos": _acos_log,
    "ArcCos": _acos_log,
    "atan": _atan_log,
    "arctan": _atan_log,
    "ArcTan": _atan_log,
    "asec": _asec,
    "arcsec": _asec,
    "ArcSec": _asec,
    "acsc": _acsc,
    "arccsc": _acsc,
    "ArcCsc": _acsc,
    "acot": _acot,
    "arccot": _acot,
    "ArcCot": _acot,
    "sinh": sinh,
    "Sinh": sinh,
    "cosh": cosh,
    "Cosh": cosh,
    "tanh": tanh,
    "Tanh": tanh,
    "asinh": _asinh_log,
    "arcsinh": _asinh_log,
    "ArcSinh": _asinh_log,
    "acosh": _acosh_log,
    "arccosh": _acosh_log,
    "ArcCosh": _acosh_log,
    "atanh": _atanh_log,
    "arctanh": _atanh_log,
    "ArcTanh": _atanh_log,
    "sqrt": sqrt,
    "Sqrt": sqrt,
    "half": _half,
    "Half": _half,
    "square": _square,
    "Square": _square,
    "Sqr": _square,
    "cbrt": _cbrt,
    "Cbrt": _cbrt,
    "avg": _avg,
    "Avg": _avg,
    "hypot": _hypot,
    "Hypot": _hypot,
    "logistic_sigmoid": _logistic_sigmoid,
    "sigmoid": _logistic_sigmoid,
    "LogisticSigmoid": _logistic_sigmoid,
    "absolute": _absolute,
    "abs": _absolute,
    "Abs": _absolute,
    "identity": _identity,
    "Identity": _identity,
    "plus": _plus,
    "Plus": _plus,
    "subtract": _subtract,
    "Subtract": _subtract,
    "times": _times,
    "Times": _times,
    "divide": _divide,
    "Divide": _divide,
    "pow": _power,
    "Pow": _power,
    "power": _power,
    "Power": _power,
    "minus": _minus,
    "Minus": _minus,
    "inv": _inv,
    "Inv": _inv,
    "log_base": _log_base,
    "LogBase": _log_base,
    "log2": _log2,
    "Log2": _log2,
    "log10": _log10,
    "Log10": _log10,
    "log1p": _log1p,
    "Log1p": _log1p,
    "exp2": _exp2,
    "Exp2": _exp2,
    "expm1": _expm1,
    "Expm1": _expm1,
    "degrees": _degrees,
    "Degrees": _degrees,
    "radians": _radians,
    "Radians": _radians,
}


def preprocess_source(source: str) -> str:
    text = source.strip()
    if not text:
        raise ValueError("Empty input expression.")
    text = re.sub(r"\b(?:np|numpy|math)\.", "", text)
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("^", "**")
    return text


def parse_expression(source: str):
    expression = sympify(preprocess_source(source), locals=LOCALS)
    if callable(expression) or not hasattr(expression, "rewrite"):
        raise TypeError(f"Input did not parse as a symbolic expression: {source!r}")
    return normalize_expression(expression)


def normalize_expression(expression, *, max_iter: int = 12):
    current = expression
    for _ in range(max_iter):
        updated = current.rewrite(sym_log).rewrite(sym_exp).rewrite(Pow)
        if updated == current:
            break
        current = updated
    return current


def _leaf(value: str) -> EMLTree:
    return EMLTree.leaf(value)


def _node(left: EMLTree, right: EMLTree) -> EMLTree:
    return EMLTree.node(left, right)


def _tree_exp(value: EMLTree) -> EMLTree:
    return _node(value, _leaf("1"))


def _tree_log(value: EMLTree) -> EMLTree:
    return _node(_leaf("1"), _tree_exp(_node(_leaf("1"), value)))


def _tree_zero() -> EMLTree:
    return _tree_log(_leaf("1"))


def _tree_sub(left: EMLTree, right: EMLTree) -> EMLTree:
    return _node(_tree_log(left), _tree_exp(right))


def _tree_neg(value: EMLTree) -> EMLTree:
    return _tree_sub(_tree_zero(), value)


def _tree_add(left: EMLTree, right: EMLTree) -> EMLTree:
    return _tree_sub(left, _tree_neg(right))


def _tree_inv(value: EMLTree) -> EMLTree:
    return _tree_exp(_tree_neg(_tree_log(value)))


def _tree_mul(left: EMLTree, right: EMLTree) -> EMLTree:
    return _tree_exp(_tree_add(_tree_log(left), _tree_log(right)))


def _tree_div(left: EMLTree, right: EMLTree) -> EMLTree:
    return _tree_mul(left, _tree_inv(right))


def _tree_pow(base: EMLTree, exponent: EMLTree) -> EMLTree:
    return _tree_exp(_tree_mul(exponent, _tree_log(base)))


def _tree_int(value: int) -> EMLTree:
    if value == 0:
        return _tree_zero()
    if value == 1:
        return _leaf("1")
    if value < 0:
        return _tree_neg(_tree_int(-value))

    acc: EMLTree | None = None
    term = _leaf("1")
    remaining = value
    while remaining > 0:
        if remaining & 1:
            acc = term if acc is None else _tree_add(acc, term)
        term = _tree_add(term, term)
        remaining >>= 1
    return acc


def _tree_rational(numerator: int, denominator: int) -> EMLTree:
    if denominator == 0:
        raise ZeroDivisionError("Rational literals require a non-zero denominator.")
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 1:
        return _tree_int(numerator)

    magnitude = _tree_div(_tree_int(abs(numerator)), _tree_int(denominator))
    return magnitude if numerator >= 0 else _tree_neg(magnitude)


def _compile_atom(expression) -> EMLTree:
    if isinstance(expression, Integer):
        return _tree_int(int(expression))
    if isinstance(expression, Rational):
        return _tree_rational(int(expression.p), int(expression.q))
    if isinstance(expression, Float):
        rational = Rational(str(expression))
        return _tree_rational(int(rational.p), int(rational.q))
    if expression == SYM_E:
        return _tree_exp(_leaf("1"))
    if expression == SYM_I:
        minus_one = _tree_neg(_leaf("1"))
        two = _tree_int(2)
        return _tree_neg(_tree_exp(_tree_div(_tree_log(minus_one), two)))
    if expression == SYM_PI:
        imag_unit = _compile_atom(SYM_I)
        return _tree_mul(imag_unit, _tree_log(_tree_neg(_leaf("1"))))
    if expression == SYM_TAU:
        pi_tree = _compile_atom(SYM_PI)
        return _tree_add(pi_tree, pi_tree)
    if expression == SYM_GOLDEN_RATIO:
        return _tree_div(_tree_add(_leaf("1"), _tree_pow(_tree_int(5), _tree_rational(1, 2))), _tree_int(2))
    if isinstance(expression, Symbol):
        return _leaf(expression.name)
    raise TypeError(f"Unsupported atom: {expression!r}")


def _compile_node(expression) -> EMLTree:
    if expression.is_Atom:
        return _compile_atom(expression)

    func = getattr(expression, "func", None)
    if func is sym_exp and len(expression.args) == 1:
        return _tree_exp(_compile_node(expression.args[0]))
    if func is sym_log and len(expression.args) == 1:
        return _tree_log(_compile_node(expression.args[0]))
    if isinstance(expression, Pow):
        base, exponent = expression.as_base_exp()
        return _tree_pow(_compile_node(base), _compile_node(exponent))
    if isinstance(expression, Mul):
        factors = list(expression.args)
        result = _compile_node(factors[0])
        for factor in factors[1:]:
            result = _tree_mul(result, _compile_node(factor))
        return result
    if isinstance(expression, Add):
        terms = list(expression.args)
        result = _compile_node(terms[0])
        for term in terms[1:]:
            result = _tree_add(result, _compile_node(term))
        return result

    normalized = normalize_expression(expression)
    if normalized != expression:
        return _compile_node(normalized)
    raise TypeError(f"Unsupported expression node: {expression!r}")


def compile_tree(source: str) -> EMLTree:
    return _compile_node(parse_expression(source))


def compile_source(source: str) -> str:
    return compile_tree(source).to_source()


def expression_stats(expression: str | EMLTree) -> dict[str, int]:
    tree = compile_tree(expression) if isinstance(expression, str) else expression
    return {
        "depth": tree.depth(),
        "leaf_count": tree.leaf_count(),
        "node_count": tree.node_count(),
    }


def evaluate(expression: str | EMLTree, /, **variables: Any):
    tree = compile_tree(expression) if isinstance(expression, str) else expression
    return tree.evaluate(variables)


__all__ = [
    "EMLTree",
    "compile_source",
    "compile_tree",
    "evaluate",
    "expression_stats",
    "normalize_expression",
    "parse_expression",
    "preprocess_source",
]
