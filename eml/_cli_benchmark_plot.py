from __future__ import annotations

import argparse
from pathlib import Path

from ._benchmark_plots import write_benchmark_plots_from_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark plots from an existing benchmark report.")
    parser.add_argument("--report", help="Path to a benchmark report JSON file.")
    parser.add_argument("--bundle-dir", help="Directory containing report.json from eml-benchmark.")
    parser.add_argument("--output-dir", help="Directory where plot PNG files should be written.")
    return parser


def _resolve_report_path(report: str | None, bundle_dir: str | None) -> Path:
    if bool(report) == bool(bundle_dir):
        raise SystemExit("Provide exactly one of --report or --bundle-dir.")
    if report:
        return Path(report)
    return Path(bundle_dir) / "report.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = _resolve_report_path(args.report, args.bundle_dir)
    paths = write_benchmark_plots_from_report(report_path, args.output_dir)
    print(f"generated plots from {report_path}")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
