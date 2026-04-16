from __future__ import annotations

import numpy as np

import eml


def assert_close(actual, expected, *, atol: float = 1e-8, rtol: float = 1e-8) -> None:
    actual_array = np.asarray(actual, dtype=np.complex128)
    expected_array = np.asarray(expected, dtype=np.complex128)
    assert np.allclose(actual_array, expected_array, atol=atol, rtol=rtol, equal_nan=True)


def test_compile_source_emits_only_eml_structure() -> None:
    source = eml.compile_source("sin(x) + sqrt(1 + x**2)")
    assert "EML(" in source
    assert "sin" not in source.lower()
    assert "sqrt" not in source.lower()


def test_compile_tree_returns_metrics_and_pretty_output() -> None:
    tree = eml.compile_tree("sin(x) + sqrt(1 + x**2)")
    stats = eml.expression_stats(tree)
    assert isinstance(tree, eml.EMLTree)
    assert stats["depth"] > 0
    assert stats["leaf_count"] > 1
    assert "EML" in tree.pretty()
    assert tree.to_source() == eml.compile_source("sin(x) + sqrt(1 + x**2)")


def test_tree_graph_exports_and_summary_helpers() -> None:
    tree = eml.compile_tree("sin(x) + sqrt(1 + x**2)")
    mermaid = tree.to_mermaid()
    dot = tree.to_dot()
    assert mermaid.startswith("graph TD")
    assert "-->" in mermaid
    assert dot.startswith("digraph EMLTree")
    assert "->" in dot
    assert tree.variables() == ["x"]
    assert tree.level_widths()[0] == 1
    assert tree.leaf_frequencies()["1"] >= 1


def test_evaluate_matches_reference_expression() -> None:
    values = np.linspace(-0.75, 0.75, 17)
    actual = eml.evaluate("sin(x) + sqrt(1 + x**2)", x=values)
    expected = np.sin(values) + np.sqrt(1 + values**2)
    assert_close(actual, expected)


def test_evaluate_supports_binary_expression() -> None:
    x_values = np.linspace(0.5, 1.5, 11)
    y_values = np.linspace(1.0, 2.0, 11)
    actual = eml.evaluate("hypot(x, y) + log(x * y)", x=x_values, y=y_values)
    expected = np.sqrt(x_values**2 + y_values**2) + np.log(x_values * y_values)
    assert_close(actual, expected)


def test_compiler_supports_wolfram_style_and_aliases() -> None:
    values = np.linspace(0.5, 2.0, 9)
    actual = eml.evaluate("Sin[x] + Log[2, x] + LogisticSigmoid[x]", x=values)
    expected = np.sin(values) + np.log(values) / np.log(2) + 1 / (1 + np.exp(-values))
    assert_close(actual, expected)


def test_compiler_strips_common_module_prefixes_and_evaluates_tree() -> None:
    values = np.linspace(-0.75, 0.75, 17)
    tree = eml.compile_tree("np.sin(x) + math.sqrt(1 + x^2)")
    actual = eml.evaluate(tree, x=values)
    expected = np.sin(values) + np.sqrt(1 + values**2)
    assert_close(actual, expected)

