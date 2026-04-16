from __future__ import annotations

import json

from eml._cli_benchmark import main as benchmark_main
from eml._cli_compile import main as compile_main


def test_compile_cli_all_mode_prints_sections(capsys) -> None:
    compile_main(["sin(x) + sqrt(1 + x**2)", "--show", "all", "--assign", "x=0.5"])
    output = capsys.readouterr().out
    assert "Compiled EML source:" in output
    assert "Tree statistics:" in output
    assert "Evaluation:" in output


def test_compile_cli_json_mode_supports_wolfram_style(capsys) -> None:
    compile_main(["Sin[x] + Log[2, x]", "--show", "json", "--assign", "x=2"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["depth"] > 0
    assert "source" in payload
    assert "evaluation" in payload


def test_compile_cli_mermaid_mode_outputs_graph(capsys) -> None:
    compile_main(["sin(x) + sqrt(1 + x**2)", "--show", "mermaid"])
    output = capsys.readouterr().out
    assert output.startswith("graph TD")


def test_compile_cli_summary_mode_outputs_structure(capsys) -> None:
    compile_main(["sin(x) + sqrt(1 + x**2)", "--show", "summary"])
    output = capsys.readouterr().out
    assert "variables: x" in output
    assert "leaf_frequencies:" in output


def test_benchmark_cli_writes_bundle_and_group_summaries(tmp_path, capsys) -> None:
    benchmark_main(["--cases", "exp", "log", "--sample-count", "16", "--repeats", "1", "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Category summaries" in output
    assert "Domain summaries" in output
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "results.csv").exists()


def test_benchmark_cli_writes_plots(tmp_path, capsys) -> None:
    plot_dir = tmp_path / "plots"
    benchmark_main(["--cases", "exp", "log", "--sample-count", "16", "--repeats", "1", "--plot-dir", str(plot_dir)])
    output = capsys.readouterr().out
    assert "wrote benchmark plots" in output
    assert (plot_dir / "error_vs_slowdown.png").exists()
