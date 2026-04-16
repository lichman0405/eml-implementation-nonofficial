# eml

`eml` is a Python library that evaluates elementary functions by reducing them to the single binary operator

```text
eml(x, y) = exp(x) - log(y)
```

and the distinguished constant `1`.

The implementation follows the construction from the paper *All elementary functions from a single binary operator* (arXiv:2603.21852). Once the primitive `eml` is fixed, the exported arithmetic, constants, and higher-level functions are built from pure EML composition rather than direct calls to NumPy trigonometric or logarithmic helpers.

## Acknowledgement

The central mathematical idea belongs entirely to Andrzej Odrzywołek. The paper is remarkable: conceptually sharp, technically constructive, and unusually generous in reproducibility. It is the kind of result that feels both surprising and inevitable once stated well. The associated research toolkit is also impressive engineering work, not just a bare proof sketch.

This repository is not an attempt to supersede that work. It is an independent Python implementation built in admiration of it.

## Why build this project?

The original paper and repository are the right place to learn the discovery itself. This project exists for a different reason: to turn the construction into a clean Python package with a stable API, test coverage, compiler and benchmark CLIs, GitHub Pages documentation, and publishable benchmark artifacts.

In other words, the paper provides the mathematical breakthrough; this repository provides a software-facing translation of that breakthrough for day-to-day Python use.

## Features

- Pure EML implementations of arithmetic, powers, logarithms, roots, trigonometric functions, hyperbolic functions, and inverse variants
- Reconstructed constants including `e`, `i`, `pi`, `tau`, the golden ratio, and several convenience constants such as `sqrt(2)` and `ln(10)`
- Standard-library style helpers such as `log2`, `log10`, `exp2`, `expm1`, `log1p`, `degrees`, and `radians`
- A compiler that accepts both Python-style and Wolfram-style expressions and emits pure EML source or an explicit EML tree
- A reusable benchmark suite plus a CLI script that compares numerical accuracy and runtime against NumPy
- A GitHub Pages-ready documentation site under `docs/`, including theory notes, usage guidance, and benchmark plots

## Installation with uv

```bash
uv venv .venv --clear
uv sync --dev
```

## Quick start

```python
import numpy as np
import eml

print(eml.sin(1.0))
print(eml.log(eml.e))
print(eml.hypot(3.0, 4.0))
print(eml.log1p(0.5))
print(eml.degrees(eml.pi / 2))

x = np.linspace(-1.0, 1.0, 5)
print(eml.cos(x))
print(eml.absolute(x))
```

## Compiler usage

The compiler accepts standard Python syntax:

```python
import eml

tree = eml.compile_tree("sin(x) + sqrt(1 + x**2)")
print(tree.pretty())
print(tree.to_source())
print(eml.expression_stats(tree))
print(eml.evaluate(tree, x=0.5))
```

It also accepts Wolfram-style syntax and common aliases:

```python
import eml

source = eml.compile_source("Sin[x] + Log[2, x] + LogisticSigmoid[x]")
print(source)

value = eml.evaluate("np.sin(x) + math.sqrt(1 + x^2)", x=0.5)
print(value)
```

## Compile CLI

The package now exposes a dedicated command-line compiler through the uv environment:

```bash
uv run eml-compile "sin(x) + sqrt(1 + x**2)"
```

Show a full report with tree statistics and evaluate the expression at `x = pi / 2`:

```bash
uv run eml-compile "Sin[x] + Log[2, x]" --show all --assign x=2
```

Emit JSON instead of plain text and write it to disk:

```bash
uv run eml-compile "np.sin(x) + math.sqrt(1 + x^2)" --show json --assign x=0.5 --output benchmark_results/compiled_expression.json
```

Generate Mermaid or Graphviz output for downstream visualization:

```bash
uv run eml-compile "sin(x) + sqrt(1 + x**2)" --show mermaid --output benchmark_results/tree.mmd
uv run eml-compile "sin(x) + sqrt(1 + x**2)" --show dot --output benchmark_results/tree.dot
```

Show a compact structural summary with variables, level widths, and leaf frequencies:

```bash
uv run eml-compile "sin(x) + sqrt(1 + x**2)" --show summary
```

The same CLI is also available as a workspace script:

```bash
uv run python scripts/eml_compile.py "sin(x)"
```

Supported parser conveniences include:

- `np.` / `numpy.` / `math.` prefixes are ignored during parsing
- `^` is accepted as a power operator and translated to `**`
- `Log[base, value]` is treated as Wolfram-style base-first logarithm
- `log(value, base)` is treated as Python-style value-first logarithm
- Aliases such as `arcsin`, `arccos`, `sigmoid`, `absolute`, `Pow`, `Half`, and `Hypot` are supported

## Public API overview

Constants:

- Uppercase: `ZERO`, `ONE`, `NEG_ONE`, `TWO`, `THREE`, `FOUR`, `FIVE`, `TEN`, `HALF`, `QUARTER`
- Mathematical: `E`, `I`, `PI`, `TAU`, `GOLDEN_RATIO`, `PHI`, `SQRT_TWO`, `SQRT_THREE`, `LN_TWO`, `LN_TEN`, `INV_E`, `INV_PI`
- Lowercase aliases: `e`, `i`, `pi`, `tau`, `phi`

Functions:

- Arithmetic: `add`, `subtract`, `multiply`, `divide`, `negate`, `reciprocal`, `power`
- Exponential and logarithmic: `exp`, `exp2`, `expm1`, `log`, `log_base`, `log2`, `log10`, `log1p`
- Algebraic: `square`, `sqrt`, `cbrt`, `absolute`, `half`, `avg`, `hypot`
- Trigonometric: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`
- Inverse trigonometric: `asin`, `acos`, `atan`, `acot`, `asec`, `acsc`
- Hyperbolic: `sinh`, `cosh`, `tanh`
- Inverse hyperbolic: `asinh`, `acosh`, `atanh`
- Angle conversion: `degrees`, `radians`
- Aliases: `arcsin`, `arccos`, `arctan`, `arcsinh`, `arccosh`, `arctanh`, `sigmoid`, `abs_value`

Compiler and tree:

- `compile_source`, `compile_tree`, `evaluate`
- `EMLTree.pretty`, `EMLTree.to_source`, `EMLTree.to_nested`, `EMLTree.to_mermaid`, `EMLTree.to_dot`, `EMLTree.evaluate`
- `expression_stats`, `parse_expression`, `preprocess_source`

Benchmarking:

- `build_default_benchmark_cases`, `run_benchmark_case`, `run_benchmark_suite`
- `build_group_summaries`, `format_benchmark_table`, `format_group_summary_table`
- `summarize_benchmark_suite`, `write_benchmark_report`, `write_benchmark_csv`, `write_group_summary_csv`, `write_benchmark_bundle`
- `write_benchmark_plots`, `write_benchmark_plots_from_report`, `load_benchmark_report`

## Benchmark CLI

Run the benchmark suite directly from the workspace:

```bash
uv run eml-benchmark
```

Write a JSON report and select a subset of cases:

```bash
uv run eml-benchmark --cases exp log sin --sample-count 8192 --repeats 7 --output benchmark_results/report.json
```

Write a full bundle containing JSON plus CSV summaries grouped by category, domain, and tree complexity:

```bash
uv run eml-benchmark --sample-count 4096 --repeats 5 --output-dir benchmark_results/full_suite
```

Generate PNG charts at the same time:

```bash
uv run eml-benchmark --sample-count 4096 --repeats 5 --output-dir benchmark_results/full_suite --plot-dir benchmark_results/full_suite/plots
```

Create plots later from an existing report bundle:

```bash
uv run eml-benchmark-plot --bundle-dir benchmark_results/full_suite --output-dir benchmark_results/full_suite/plots
```

The legacy wrapper still exists for convenience:

```bash
uv run python scripts/compare_with_numpy.py --output-dir benchmark_results/legacy_wrapper
```

## Notes on complex branches

The EML chain works internally over the principal complex branch. On the negative real axis, the canonical EML logarithm produces the sign convention described in the paper, so the exported constant `I` includes the same branch correction used by the reference implementation.

For functions such as `absolute`, the intended target is the real axis. The implementation uses an EML expression equivalent to `sqrt(x**2)`, which matches `abs(x)` for real input and follows the principal branch for complex input.
