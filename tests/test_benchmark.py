from __future__ import annotations

import json

import eml


def test_run_benchmark_suite_returns_structured_results() -> None:
    results = eml.run_benchmark_suite(sample_count=32, repeats=1)
    assert results
    assert all(result.sample_count == 32 for result in results)
    assert all(result.max_abs_error >= 0 for result in results)
    assert all(result.tree_depth > 0 for result in results)
    assert all(result.category for result in results)
    assert any(result.name == "exp" for result in results)


def test_write_benchmark_report_persists_json(tmp_path) -> None:
    results = eml.run_benchmark_suite(cases=eml.build_default_benchmark_cases()[:2], sample_count=16, repeats=1)
    output_path = tmp_path / "benchmark.json"
    path = eml.write_benchmark_report(results, output_path, metadata={"suite": "test"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] == 2
    assert "group_summaries" in payload
    assert payload["metadata"]["suite"] == "test"
    assert len(payload["results"]) == 2


def test_group_summaries_and_bundle_outputs(tmp_path) -> None:
    results = eml.run_benchmark_suite(cases=eml.build_default_benchmark_cases()[:3], sample_count=16, repeats=1)
    summaries = eml.build_group_summaries(results, "category")
    bundle_paths = eml.write_benchmark_bundle(results, tmp_path, metadata={"suite": "bundle"})

    assert summaries
    assert all(summary.case_count >= 1 for summary in summaries)
    assert bundle_paths["report_json"].exists()
    assert bundle_paths["results_csv"].exists()
    assert bundle_paths["category_summary_csv"].exists()
    assert bundle_paths["domain_summary_csv"].exists()
    assert bundle_paths["complexity_summary_csv"].exists()


def test_plot_generation_from_results_and_report(tmp_path) -> None:
    results = eml.run_benchmark_suite(cases=eml.build_default_benchmark_cases()[:3], sample_count=16, repeats=1)
    plot_paths = eml.write_benchmark_plots(results, tmp_path / "plots")
    bundle_paths = eml.write_benchmark_bundle(results, tmp_path / "bundle", metadata={"suite": "plots"})
    report_plot_paths = eml.write_benchmark_plots_from_report(bundle_paths["report_json"], tmp_path / "report_plots")

    assert plot_paths["error_vs_slowdown"].exists()
    assert plot_paths["complexity_vs_slowdown"].exists()
    assert report_plot_paths["category_slowdown"].exists()
    assert report_plot_paths["depth_vs_error"].exists()
