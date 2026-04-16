from __future__ import annotations

import argparse
from pathlib import Path

from ._benchmark import (
    build_default_benchmark_cases,
    build_group_summaries,
    format_benchmark_table,
    format_group_summary_table,
    run_benchmark_suite,
    summarize_benchmark_suite,
    write_benchmark_bundle,
    write_benchmark_report,
)
from ._benchmark_plots import write_benchmark_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a systematic EML vs NumPy benchmark suite.")
    parser.add_argument("--sample-count", type=int, default=4096, help="Number of samples used for each benchmark case.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing repetitions per case.")
    parser.add_argument("--cases", nargs="*", help="Optional subset of case names to run.")
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--output-dir", help="Optional directory for a full JSON + CSV benchmark bundle.")
    parser.add_argument("--plot-dir", help="Optional directory for generated PNG plots.")
    parser.add_argument("--no-group-summaries", action="store_true", help="Suppress grouped summary tables in console output.")
    return parser


def _filter_cases(names: list[str] | None):
    cases = build_default_benchmark_cases()
    if not names:
        return cases

    selected_names = set(names)
    filtered_cases = [case for case in cases if case.name in selected_names]
    missing = sorted(selected_names - {case.name for case in filtered_cases})
    if missing:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(missing)}")
    return filtered_cases


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = _filter_cases(args.cases)
    results = run_benchmark_suite(cases=cases, sample_count=args.sample_count, repeats=args.repeats)

    print(format_benchmark_table(results))

    summary = summarize_benchmark_suite(results)
    print()
    print(f"cases: {summary['case_count']}")
    print(f"max error: {summary['max_error']:.6e}")
    print(f"mean error: {summary['mean_error']:.6e}")
    print(f"mean slowdown: {summary['mean_slowdown']:.2f}")
    print(f"max slowdown: {summary['max_slowdown']:.2f}")
    print(f"mean tree depth: {summary['mean_tree_depth']:.2f}")
    print(f"max tree depth: {summary['max_tree_depth']}")

    if not args.no_group_summaries:
        for label, attribute in (("Category summaries", "category"), ("Domain summaries", "domain"), ("Complexity summaries", "complexity_band")):
            print()
            print(label)
            print(format_group_summary_table(build_group_summaries(results, attribute)))

    metadata = {
        "sample_count": args.sample_count,
        "repeats": args.repeats,
        "case_names": ",".join(case.name for case in cases),
    }

    if args.output:
        path = write_benchmark_report(results, args.output, metadata=metadata)
        print(f"wrote JSON report to {path}")

    if args.output_dir:
        paths = write_benchmark_bundle(results, args.output_dir, metadata=metadata)
        print(f"wrote benchmark bundle to {Path(args.output_dir)}")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    if args.plot_dir:
        plot_paths = write_benchmark_plots(results, args.plot_dir)
        print(f"wrote benchmark plots to {Path(args.plot_dir)}")
        for key, path in plot_paths.items():
            print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
