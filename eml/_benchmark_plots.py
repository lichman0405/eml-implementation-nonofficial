from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ._benchmark import BenchmarkResult


def _coerce_results(results: list[BenchmarkResult] | list[dict[str, Any]]) -> list[BenchmarkResult]:
    coerced: list[BenchmarkResult] = []
    for result in results:
        if isinstance(result, BenchmarkResult):
            coerced.append(result)
        else:
            coerced.append(BenchmarkResult(**result))
    return coerced


def load_benchmark_report(report_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def _finalize_plot(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


def plot_error_vs_slowdown(results: list[BenchmarkResult] | list[dict[str, Any]], output_path: str | Path) -> Path:
    coerced = _coerce_results(results)
    x_values = np.asarray([result.slowdown for result in coerced], dtype=np.float64)
    y_values = np.asarray([max(result.max_abs_error, 1e-18) for result in coerced], dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, s=60, color="#1f77b4")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Slowdown vs NumPy")
    plt.ylabel("Max absolute error")
    plt.title("EML accuracy vs slowdown")
    for result in coerced:
        plt.annotate(result.name, (result.slowdown, max(result.max_abs_error, 1e-18)), textcoords="offset points", xytext=(5, 4), fontsize=8)
    return _finalize_plot(Path(output_path))


def plot_complexity_vs_slowdown(results: list[BenchmarkResult] | list[dict[str, Any]], output_path: str | Path) -> Path:
    coerced = _coerce_results(results)
    categories = sorted({result.category for result in coerced})
    colors = matplotlib.colormaps.get_cmap("tab10")

    plt.figure(figsize=(8, 5))
    for index, category in enumerate(categories):
        subset = [result for result in coerced if result.category == category]
        plt.scatter(
            [result.tree_leaf_count for result in subset],
            [result.slowdown for result in subset],
            s=70,
            label=category,
            color=colors(index / max(len(categories) - 1, 1)),
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Leaf count")
    plt.ylabel("Slowdown vs NumPy")
    plt.title("EML tree complexity vs slowdown")
    plt.legend()
    return _finalize_plot(Path(output_path))


def plot_category_slowdown(results: list[BenchmarkResult] | list[dict[str, Any]], output_path: str | Path) -> Path:
    coerced = _coerce_results(results)
    categories = sorted({result.category for result in coerced})
    mean_slowdowns = []
    for category in categories:
        subset = [result.slowdown for result in coerced if result.category == category]
        mean_slowdowns.append(float(np.mean(subset)))

    plt.figure(figsize=(8, 5))
    plt.bar(categories, mean_slowdowns, color="#ff7f0e")
    plt.ylabel("Mean slowdown vs NumPy")
    plt.title("Benchmark slowdown by category")
    plt.xticks(rotation=20, ha="right")
    return _finalize_plot(Path(output_path))


def plot_depth_vs_error(results: list[BenchmarkResult] | list[dict[str, Any]], output_path: str | Path) -> Path:
    coerced = _coerce_results(results)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        [result.tree_depth for result in coerced],
        [max(result.max_abs_error, 1e-18) for result in coerced],
        s=60,
        color="#2ca02c",
    )
    plt.yscale("log")
    plt.xlabel("Tree depth")
    plt.ylabel("Max absolute error")
    plt.title("Tree depth vs max error")
    for result in coerced:
        plt.annotate(result.name, (result.tree_depth, max(result.max_abs_error, 1e-18)), textcoords="offset points", xytext=(5, 4), fontsize=8)
    return _finalize_plot(Path(output_path))


def write_benchmark_plots(results: list[BenchmarkResult] | list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    coerced = _coerce_results(results)
    return {
        "error_vs_slowdown": plot_error_vs_slowdown(coerced, directory / "error_vs_slowdown.png"),
        "complexity_vs_slowdown": plot_complexity_vs_slowdown(coerced, directory / "complexity_vs_slowdown.png"),
        "category_slowdown": plot_category_slowdown(coerced, directory / "category_slowdown.png"),
        "depth_vs_error": plot_depth_vs_error(coerced, directory / "depth_vs_error.png"),
    }


def write_benchmark_plots_from_report(report_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    report = load_benchmark_report(report_path)
    destination = Path(output_dir) if output_dir is not None else Path(report_path).resolve().parent / "plots"
    return write_benchmark_plots(report["results"], destination)


__all__ = [
    "load_benchmark_report",
    "plot_category_slowdown",
    "plot_complexity_vs_slowdown",
    "plot_depth_vs_error",
    "plot_error_vs_slowdown",
    "write_benchmark_plots",
    "write_benchmark_plots_from_report",
]
