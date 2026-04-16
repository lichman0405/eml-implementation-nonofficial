from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sympy import E as SYM_E, I as SYM_I, pi as SYM_PI, sympify

from ._compiler import compile_source, compile_tree, evaluate, expression_stats, preprocess_source

VALUE_LOCALS = {
    "e": SYM_E,
    "E": SYM_E,
    "i": SYM_I,
    "I": SYM_I,
    "pi": SYM_PI,
    "Pi": SYM_PI,
    "tau": 2 * SYM_PI,
    "Tau": 2 * SYM_PI,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile symbolic expressions into pure EML form.")
    parser.add_argument("expression", nargs="?", help="Expression to compile. If omitted, stdin is used.")
    parser.add_argument("--file", dest="input_file", help="Read the input expression from a file.")
    parser.add_argument(
        "--show",
        choices=("source", "tree", "stats", "summary", "mermaid", "dot", "json", "all"),
        default="source",
        help="Select the output format.",
    )
    parser.add_argument("--assign", nargs="*", default=[], metavar="NAME=VALUE", help="Optional variable bindings used for evaluation.")
    parser.add_argument("--output", help="Optional output file path.")
    return parser


def _read_expression(expression: str | None, input_file: str | None) -> str:
    if expression and input_file:
        raise ValueError("Provide either an inline expression or --file, not both.")
    if input_file:
        return Path(input_file).read_text(encoding="utf-8").strip()
    if expression:
        return expression
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    raise ValueError("Missing input expression.")


def _literal_or_symbolic_value(text: str) -> Any:
    stripped = text.strip()
    try:
        literal = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        symbolic = sympify(stripped, locals=VALUE_LOCALS)
        numeric = complex(symbolic.evalf())
        if abs(numeric.imag) <= 1e-12:
            return float(numeric.real)
        return numeric

    if isinstance(literal, (list, tuple)):
        return np.asarray(literal, dtype=np.complex128)
    return literal


def parse_assignments(items: list[str]) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid assignment {item!r}. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError(f"Invalid assignment {item!r}. Symbol name is empty.")
        bindings[cleaned_name] = _literal_or_symbolic_value(raw_value)
    return bindings


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_serialize_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _serialize_value(value.item())
    if isinstance(value, complex):
        if abs(value.imag) <= 1e-12:
            return value.real
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def build_payload(expression: str, assignments: dict[str, Any]) -> dict[str, Any]:
    tree = compile_tree(expression)
    stats = expression_stats(tree)
    payload = {
        "input": expression,
        "preprocessed": preprocess_source(expression),
        "source": tree.to_source(),
        "stats": stats,
        "tree": tree.to_nested(),
        "pretty_tree": tree.pretty(),
        "summary": {
            "variables": tree.variables(),
            "leaf_frequencies": tree.leaf_frequencies(),
            "level_widths": tree.level_widths(),
            "internal_node_count": tree.internal_node_count(),
        },
        "mermaid": tree.to_mermaid(),
        "dot": tree.to_dot(),
    }
    if assignments:
        payload["bindings"] = {name: _serialize_value(value) for name, value in assignments.items()}
        payload["evaluation"] = _serialize_value(evaluate(tree, **assignments))
    return payload


def render_payload(payload: dict[str, Any], *, mode: str) -> str:
    if mode == "source":
        return str(payload["source"])
    if mode == "tree":
        return str(payload["pretty_tree"])
    if mode == "stats":
        stats = payload["stats"]
        return "\n".join(
            [
                f"depth: {stats['depth']}",
                f"leaf_count: {stats['leaf_count']}",
                f"node_count: {stats['node_count']}",
            ]
        )
    if mode == "summary":
        summary = payload["summary"]
        lines = [
            f"variables: {', '.join(summary['variables']) if summary['variables'] else '(none)'}",
            f"internal_node_count: {summary['internal_node_count']}",
            f"level_widths: {summary['level_widths']}",
            "leaf_frequencies:",
        ]
        for leaf, count in summary["leaf_frequencies"].items():
            lines.append(f"  {leaf}: {count}")
        return "\n".join(lines)
    if mode == "mermaid":
        return str(payload["mermaid"])
    if mode == "dot":
        return str(payload["dot"])
    if mode == "json":
        serializable = {key: value for key, value in payload.items() if key != "pretty_tree"}
        return json.dumps(serializable, indent=2)
    if mode == "all":
        lines = [
            "Input expression:",
            str(payload["input"]),
            "",
            "Preprocessed expression:",
            str(payload["preprocessed"]),
            "",
            "Compiled EML source:",
            str(payload["source"]),
            "",
            "Tree statistics:",
            f"depth: {payload['stats']['depth']}",
            f"leaf_count: {payload['stats']['leaf_count']}",
            f"node_count: {payload['stats']['node_count']}",
            "",
            "Structural summary:",
            f"variables: {', '.join(payload['summary']['variables']) if payload['summary']['variables'] else '(none)'}",
            f"internal_node_count: {payload['summary']['internal_node_count']}",
            f"level_widths: {payload['summary']['level_widths']}",
            f"leaf_frequencies: {payload['summary']['leaf_frequencies']}",
            "",
            "Pretty tree:",
            str(payload["pretty_tree"]),
        ]
        if "evaluation" in payload:
            lines.extend([
                "",
                "Evaluation:",
                json.dumps(payload["evaluation"], indent=2),
            ])
        return "\n".join(lines)
    raise ValueError(f"Unsupported output mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expression = _read_expression(args.expression, args.input_file)
    assignments = parse_assignments(args.assign)
    payload = build_payload(expression, assignments)
    rendered = render_payload(payload, mode=args.show)

    if args.output:
        Path(args.output).write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

