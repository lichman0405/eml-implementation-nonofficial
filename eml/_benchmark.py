from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from . import _functions as functions
from ._compiler import expression_stats

UnaryFunction = Callable[[np.ndarray], np.ndarray | complex | float]
ArrayFactory = Callable[[int], np.ndarray]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    category: str
    domain: str
    sample_range: str
    expression: str
    eml_function: UnaryFunction
    reference_function: UnaryFunction
    sample_factory: ArrayFactory
    notes: str = ""

    def build_samples(self, sample_count: int) -> np.ndarray:
        return np.asarray(self.sample_factory(sample_count), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    category: str
    domain: str
    sample_range: str
    expression: str
    sample_count: int
    max_abs_error: float
    mean_abs_error: float
    rms_abs_error: float
    numpy_seconds: float
    eml_seconds: float
    slowdown: float
    tree_depth: int
    tree_leaf_count: int
    tree_node_count: int
    complexity_band: str
    notes: str = ""

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkGroupSummary:
    group_by: str
    group: str
    case_count: int
    max_abs_error: float
    mean_abs_error: float
    rms_abs_error: float
    mean_slowdown: float
    max_slowdown: float
    mean_tree_depth: float
    max_tree_depth: int
    mean_tree_leaf_count: float
    max_tree_leaf_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _measure(func: UnaryFunction, values: np.ndarray, repeats: int) -> tuple[np.ndarray, float]:
    best = float("inf")
    result: np.ndarray | None = None
    for _ in range(repeats):
        start = perf_counter()
        result = np.asarray(func(values), dtype=np.complex128)
        best = min(best, perf_counter() - start)
    return result, best


def _classify_complexity(*, depth: int, leaf_count: int) -> str:
    if depth <= 12 and leaf_count <= 16:
        return "small"
    if depth <= 32 and leaf_count <= 64:
        return "medium"
    return "large"


def build_default_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="exp",
            category="exponential",
            domain="symmetric_real",
            sample_range="[-5.0, 5.0]",
            expression="exp(x)",
            eml_function=functions.exp,
            reference_function=np.exp,
            sample_factory=lambda n: np.linspace(-5.0, 5.0, n),
            notes="Entire real line sample",
        ),
        BenchmarkCase(
            name="log",
            category="exponential",
            domain="positive_real",
            sample_range="[0.1, 10.0]",
            expression="log(x)",
            eml_function=functions.log,
            reference_function=np.log,
            sample_factory=lambda n: np.linspace(0.1, 10.0, n),
            notes="Positive real axis",
        ),
        BenchmarkCase(
            name="sqrt",
            category="algebraic",
            domain="nonnegative_real",
            sample_range="[0.0, 20.0]",
            expression="sqrt(x)",
            eml_function=functions.sqrt,
            reference_function=np.sqrt,
            sample_factory=lambda n: np.linspace(0.0, 20.0, n),
            notes="Non-negative real axis",
        ),
        BenchmarkCase(
            name="sin",
            category="trigonometric",
            domain="symmetric_real",
            sample_range="[-2.0, 2.0]",
            expression="sin(x)",
            eml_function=functions.sin,
            reference_function=np.sin,
            sample_factory=lambda n: np.linspace(-2.0, 2.0, n),
            notes="Euler-form trigonometric evaluation",
        ),
        BenchmarkCase(
            name="cos",
            category="trigonometric",
            domain="symmetric_real",
            sample_range="[-2.0, 2.0]",
            expression="cos(x)",
            eml_function=functions.cos,
            reference_function=np.cos,
            sample_factory=lambda n: np.linspace(-2.0, 2.0, n),
            notes="Euler-form trigonometric evaluation",
        ),
        BenchmarkCase(
            name="tanh",
            category="hyperbolic",
            domain="symmetric_real",
            sample_range="[-3.0, 3.0]",
            expression="tanh(x)",
            eml_function=functions.tanh,
            reference_function=np.tanh,
            sample_factory=lambda n: np.linspace(-3.0, 3.0, n),
            notes="Hyperbolic ratio",
        ),
        BenchmarkCase(
            name="asin",
            category="inverse",
            domain="principal_interval",
            sample_range="[-0.9, 0.9]",
            expression="asin(x)",
            eml_function=functions.asin,
            reference_function=np.arcsin,
            sample_factory=lambda n: np.linspace(-0.9, 0.9, n),
            notes="Principal inverse branch",
        ),
        BenchmarkCase(
            name="acosh",
            category="inverse",
            domain="principal_inverse_positive",
            sample_range="[1.1, 6.0]",
            expression="acosh(x)",
            eml_function=functions.acosh,
            reference_function=np.arccosh,
            sample_factory=lambda n: np.linspace(1.1, 6.0, n),
            notes="Principal inverse branch",
        ),
        BenchmarkCase(
            name="logistic_sigmoid",
            category="hyperbolic",
            domain="wide_real",
            sample_range="[-10.0, 10.0]",
            expression="logistic_sigmoid(x)",
            eml_function=functions.logistic_sigmoid,
            reference_function=lambda values: 1 / (1 + np.exp(-values)),
            sample_factory=lambda n: np.linspace(-10.0, 10.0, n),
            notes="Stable tanh-based form",
        ),
        BenchmarkCase(
            name="log1p",
            category="stability",
            domain="near_negative_one",
            sample_range="[-0.9, 5.0]",
            expression="log1p(x)",
            eml_function=functions.log1p,
            reference_function=np.log1p,
            sample_factory=lambda n: np.linspace(-0.9, 5.0, n),
            notes="Cancellation-sensitive region",
        ),
        BenchmarkCase(
            name="expm1",
            category="stability",
            domain="near_zero",
            sample_range="[-2.0, 2.0]",
            expression="expm1(x)",
            eml_function=functions.expm1,
            reference_function=np.expm1,
            sample_factory=lambda n: np.linspace(-2.0, 2.0, n),
            notes="Small-input exponential difference",
        ),
    ]


def run_benchmark_case(case: BenchmarkCase, *, sample_count: int = 4096, repeats: int = 5) -> BenchmarkResult:
    values = case.build_samples(sample_count)
    eml_result, eml_seconds = _measure(case.eml_function, values, repeats)
    reference_result, numpy_seconds = _measure(case.reference_function, values, repeats)
    error = np.abs(eml_result - reference_result)
    slowdown = eml_seconds / numpy_seconds if numpy_seconds > 0 else float("inf")
    stats = expression_stats(case.expression)
    complexity_band = _classify_complexity(depth=stats["depth"], leaf_count=stats["leaf_count"])

    return BenchmarkResult(
        name=case.name,
        category=case.category,
        domain=case.domain,
        sample_range=case.sample_range,
        expression=case.expression,
        sample_count=sample_count,
        max_abs_error=float(np.max(error)),
        mean_abs_error=float(np.mean(error)),
        rms_abs_error=float(np.sqrt(np.mean(error**2))),
        numpy_seconds=float(numpy_seconds),
        eml_seconds=float(eml_seconds),
        slowdown=float(slowdown),
        tree_depth=int(stats["depth"]),
        tree_leaf_count=int(stats["leaf_count"]),
        tree_node_count=int(stats["node_count"]),
        complexity_band=complexity_band,
        notes=case.notes,
    )


def run_benchmark_suite(
    cases: Sequence[BenchmarkCase] | None = None,
    *,
    sample_count: int = 4096,
    repeats: int = 5,
) -> list[BenchmarkResult]:
    selected_cases = list(build_default_benchmark_cases() if cases is None else cases)
    return [run_benchmark_case(case, sample_count=sample_count, repeats=repeats) for case in selected_cases]


def summarize_benchmark_suite(results: Sequence[BenchmarkResult]) -> dict[str, float | int]:
    if not results:
        return {
            "case_count": 0,
            "max_error": 0.0,
            "mean_error": 0.0,
            "mean_slowdown": 0.0,
            "max_slowdown": 0.0,
            "mean_tree_depth": 0.0,
            "max_tree_depth": 0,
        }

    return {
        "case_count": len(results),
        "max_error": float(max(result.max_abs_error for result in results)),
        "mean_error": float(np.mean([result.mean_abs_error for result in results])),
        "mean_slowdown": float(np.mean([result.slowdown for result in results])),
        "max_slowdown": float(max(result.slowdown for result in results)),
        "mean_tree_depth": float(np.mean([result.tree_depth for result in results])),
        "max_tree_depth": int(max(result.tree_depth for result in results)),
    }


def build_group_summaries(results: Sequence[BenchmarkResult], group_by: str) -> list[BenchmarkGroupSummary]:
    groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        groups[str(getattr(result, group_by))].append(result)

    summaries: list[BenchmarkGroupSummary] = []
    for group, grouped_results in sorted(groups.items()):
        errors = np.asarray([result.max_abs_error for result in grouped_results], dtype=np.float64)
        mean_errors = np.asarray([result.mean_abs_error for result in grouped_results], dtype=np.float64)
        rms_errors = np.asarray([result.rms_abs_error for result in grouped_results], dtype=np.float64)
        slowdowns = np.asarray([result.slowdown for result in grouped_results], dtype=np.float64)
        depths = np.asarray([result.tree_depth for result in grouped_results], dtype=np.float64)
        leaf_counts = np.asarray([result.tree_leaf_count for result in grouped_results], dtype=np.float64)
        summaries.append(
            BenchmarkGroupSummary(
                group_by=group_by,
                group=group,
                case_count=len(grouped_results),
                max_abs_error=float(np.max(errors)),
                mean_abs_error=float(np.mean(mean_errors)),
                rms_abs_error=float(np.mean(rms_errors)),
                mean_slowdown=float(np.mean(slowdowns)),
                max_slowdown=float(np.max(slowdowns)),
                mean_tree_depth=float(np.mean(depths)),
                max_tree_depth=int(np.max(depths)),
                mean_tree_leaf_count=float(np.mean(leaf_counts)),
                max_tree_leaf_count=int(np.max(leaf_counts)),
            )
        )
    return summaries


def format_benchmark_table(results: Sequence[BenchmarkResult]) -> str:
    header = (
        f"{'name':<18} {'category':<14} {'domain':<24} {'depth':>6} {'leaves':>8} "
        f"{'max_abs_error':>16} {'numpy_s':>12} {'eml_s':>12} {'slowdown':>12}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.name:<18} {result.category:<14} {result.domain:<24} {result.tree_depth:>6d} "
            f"{result.tree_leaf_count:>8d} {result.max_abs_error:>16.6e} {result.numpy_seconds:>12.6f} "
            f"{result.eml_seconds:>12.6f} {result.slowdown:>12.2f}"
        )
    return "\n".join(lines)


def format_group_summary_table(summaries: Sequence[BenchmarkGroupSummary]) -> str:
    header = (
        f"{'group':<24} {'cases':>8} {'max_abs_error':>16} {'mean_abs_error':>16} "
        f"{'mean_depth':>12} {'mean_slowdown':>14} {'max_slowdown':>14}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        lines.append(
            f"{summary.group:<24} {summary.case_count:>8d} {summary.max_abs_error:>16.6e} "
            f"{summary.mean_abs_error:>16.6e} {summary.mean_tree_depth:>12.2f} "
            f"{summary.mean_slowdown:>14.2f} {summary.max_slowdown:>14.2f}"
        )
    return "\n".join(lines)


def write_benchmark_csv(results: Sequence[BenchmarkResult], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_dict())
    return path


def write_group_summary_csv(summaries: Sequence[BenchmarkGroupSummary], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkGroupSummary.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.as_dict())
    return path


def write_benchmark_report(
    results: Sequence[BenchmarkResult],
    destination: str | Path,
    *,
    metadata: dict[str, str | int | float] | None = None,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize_benchmark_suite(results),
        "group_summaries": {
            "category": [summary.as_dict() for summary in build_group_summaries(results, "category")],
            "domain": [summary.as_dict() for summary in build_group_summaries(results, "domain")],
            "complexity_band": [summary.as_dict() for summary in build_group_summaries(results, "complexity_band")],
        },
        "metadata": {} if metadata is None else metadata,
        "results": [result.as_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_benchmark_bundle(
    results: Sequence[BenchmarkResult],
    destination_dir: str | Path,
    *,
    metadata: dict[str, str | int | float] | None = None,
) -> dict[str, Path]:
    directory = Path(destination_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_json": write_benchmark_report(results, directory / "report.json", metadata=metadata),
        "results_csv": write_benchmark_csv(results, directory / "results.csv"),
        "category_summary_csv": write_group_summary_csv(build_group_summaries(results, "category"), directory / "category_summary.csv"),
        "domain_summary_csv": write_group_summary_csv(build_group_summaries(results, "domain"), directory / "domain_summary.csv"),
        "complexity_summary_csv": write_group_summary_csv(build_group_summaries(results, "complexity_band"), directory / "complexity_summary.csv"),
    }
    return paths


__all__ = [
    "BenchmarkCase",
    "BenchmarkGroupSummary",
    "BenchmarkResult",
    "build_default_benchmark_cases",
    "build_group_summaries",
    "format_benchmark_table",
    "format_group_summary_table",
    "run_benchmark_case",
    "run_benchmark_suite",
    "summarize_benchmark_suite",
    "write_benchmark_bundle",
    "write_benchmark_csv",
    "write_benchmark_report",
    "write_group_summary_csv",
]